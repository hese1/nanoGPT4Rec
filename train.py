"""
Training script for ProductGPT on MovieLens dataset.
Example usage:
$ python train_recsys.py --batch_size=32 --block_size=50 --n_layer=6 --n_head=8
"""

import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from model import ProductGPT, ProductGPTConfig

# -----------------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# default config values
out_dir = "out-recsys"
eval_interval = 100
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True

# data
dataset = "movielens-1m"
batch_size = 32
block_size = 50  # max sequence length
data_dir = os.path.join("data", dataset)

# model
n_layer = 6
n_head = 8
n_embd = 384
dropout = 0.1
bias = True

# optimizer
learning_rate = 1e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 1e-5

# system
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True

# -----------------------------------------------------------------------------


class MovieLensDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        # Load ratings data (UserID::MovieID::Rating::Timestamp)
        ratings_file = os.path.join(data_dir, "ratings.dat")
        self.ratings = pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )

        # Create user and movie mappings from ratings file
        self.user_id_map = {
            id: i for i, id in enumerate(self.ratings["userId"].unique())
        }
        self.movie_id_map = {
            id: i for i, id in enumerate(self.ratings["movieId"].unique())
        }

        # Sort by timestamp
        self.ratings = self.ratings.sort_values("timestamp")

        # Split based on time
        timestamp_threshold = self.ratings["timestamp"].quantile(0.8)
        train_ratings = self.ratings[self.ratings["timestamp"] <= timestamp_threshold]
        val_ratings = self.ratings[self.ratings["timestamp"] > timestamp_threshold]

        # Group by user and create sequences
        ratings_to_use = train_ratings if split == "train" else val_ratings
        user_sequences = ratings_to_use.groupby("userId")

        # Create sequences
        self.sequences = []
        for user_id, group in user_sequences:
            movies = [self.movie_id_map[m] for m in group["movieId"]]
            ratings = group["rating"].values
            if len(movies) >= 3:  # Minimum sequence length
                self.sequences.append(
                    (
                        self.user_id_map[user_id],
                        movies,
                        ratings,
                        group["timestamp"].values,
                    )
                )

        logging.info(f"Loaded {len(self.sequences)} sequences for {split}")
        logging.info(f"Number of users: {len(self.user_id_map)}")
        logging.info(f"Number of movies: {len(self.movie_id_map)}")

        # Create context features for all timestamps
        self.contexts = {}
        all_timestamps = ratings_to_use["timestamp"].values
        for ts in all_timestamps:
            hour = (ts % 86400) / 3600  # Hour of day (0-23)
            day = (ts % 604800) / 86400  # Day of week (0-6)
            self.contexts[ts] = np.array(
                [
                    np.sin(2 * np.pi * hour / 24),  # Hour of day
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * day / 7),  # Day of week
                    np.cos(2 * np.pi * day / 7),
                ]
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user_id, movie_sequence, ratings, timestamps = self.sequences[idx]

        if len(movie_sequence) > block_size:
            # Take the most recent block_size items
            movie_sequence = movie_sequence[-block_size:]
            ratings = ratings[-block_size:]
            timestamps = timestamps[-block_size:]
        else:
            # Pad sequences
            pad_length = block_size - len(movie_sequence)
            movie_sequence = [0] * pad_length + movie_sequence
            ratings = np.pad(ratings, (pad_length, 0), "constant", constant_values=0)
            timestamps = np.pad(timestamps, (pad_length, 0), "edge")

        # Calculate time deltas between interactions
        time_deltas = np.diff(timestamps)
        time_deltas = np.concatenate([[0], time_deltas])

        # Get context features for each timestamp
        context_sequence = np.stack(
            [
                np.concatenate(
                    [
                        self.contexts[ts],
                        [min(delta / 86400, 30.0)],  # Time delta in days, capped at 30
                        [rating / 5.0],  # Normalize rating to 0-1
                    ]
                )
                for ts, delta, rating in zip(timestamps, time_deltas, ratings)
            ]
        )

        # Make sure product_history and targets have the same sequence length
        product_history = torch.tensor(movie_sequence[:-1], dtype=torch.long)
        targets = torch.tensor(movie_sequence[1:], dtype=torch.long)

        # Pad context features to match sequence length
        context_features = torch.tensor(context_sequence[:-1], dtype=torch.float)

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "context_features": context_features,
            "product_history": product_history,
            "targets": targets,
        }


def create_dataloaders():
    train_dataset = MovieLensDataset(data_dir, split="train")
    val_dataset = MovieLensDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Increase if needed
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    return train_loader, val_loader


def get_batch(split, train_loader, val_loader):
    loader = train_loader if split == "train" else val_loader
    return next(iter(loader))


@torch.no_grad()
def evaluate(model, val_loader, ctx):
    model.eval()
    total_loss = 0
    total_hits = 0
    total_ndcg = 0
    n_batches = 0

    for batch in val_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with ctx:
            logits, loss = model(
                context_features=batch["context_features"],
                user_id=batch["user_id"],
                product_history=batch["product_history"],
                targets=batch["targets"],
            )

            # Get top-k recommendations
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            _, top_items = torch.topk(probs, k=10)

            # Calculate metrics
            actual = batch["targets"][:, -1].unsqueeze(-1)
            hits = torch.any(top_items == actual, dim=1).float().mean()

            # NDCG calculation
            pos_in_top = (top_items == actual).nonzero(as_tuple=True)[1]
            ndcg = torch.zeros(1, device=device)
            if pos_in_top.shape[0] > 0:
                ndcg = (1 / torch.log2(pos_in_top.float() + 2)).mean()

        total_loss += loss.item()
        total_hits += hits.item()
        total_ndcg += ndcg.item()
        n_batches += 1

    model.train()
    return {
        "loss": total_loss / n_batches,
        "hit_rate": total_hits / n_batches,
        "ndcg": total_ndcg / n_batches,
    }


def main():
    # Get number of products from dataset
    train_dataset = MovieLensDataset(data_dir, split="train")
    n_products = len(train_dataset.movie_id_map)
    n_users = len(train_dataset.user_id_map)

    logging.info(f"Dataset stats:")
    logging.info(f"  Users: {n_users}")
    logging.info(f"  Movies: {n_products}")

    # Initialize model
    config = ProductGPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        # ProductGPT specific
        max_products=n_products,
        max_users=n_users,
        n_context_features=6,  # 4 time features + time delta + rating
        n_user_metadata=0,  # No user metadata available
        anonymous_user_token=-1,
    )

    model = ProductGPT(config)
    model.to(device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders()

    # Set up training
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device
    )
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    if compile:
        print("Compiling model...")
        model = torch.compile(model)

    logging.info(f"Starting training with device={device}, dtype={dtype}")

    # Training loop
    best_val_loss = float("inf")
    for iter_num in range(max_iters):
        t0 = time.time()

        # Get batch and train
        batch = get_batch("train", train_loader, val_loader)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        with ctx:
            logits, loss = model(
                context_features=batch["context_features"],
                user_id=batch["user_id"],
                product_history=batch["product_history"],
                targets=batch["targets"],
            )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Logging
        if iter_num % log_interval == 0:
            logging.debug(
                f"iter {iter_num}: loss {loss.item():.4f}, time {(time.time()-t0)*1000:.2f}ms"
            )

        # Evaluation
        if iter_num % eval_interval == 0:
            train_metrics = evaluate(model, train_loader, ctx)
            val_metrics = evaluate(model, val_loader, ctx)

            logging.info(f"\nStep {iter_num} Metrics:")
            logging.info("Training:")
            logging.info(f"  Loss: {train_metrics['loss']:.4f}")
            logging.info(f"  Hit@10: {train_metrics['hit_rate']:.4f}")
            logging.info(f"  NDCG@10: {train_metrics['ndcg']:.4f}")
            logging.info("Validation:")
            logging.info(f"  Loss: {val_metrics['loss']:.4f}")
            logging.info(f"  Hit@10: {val_metrics['hit_rate']:.4f}")
            logging.info(f"  NDCG@10: {val_metrics['ndcg']:.4f}")

            # Save checkpoint based on validation loss
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter_num": iter_num,
                    "metrics": {
                        "train": train_metrics,
                        "val": val_metrics,
                    },
                }
                logging.info(f"saving checkpoint to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
