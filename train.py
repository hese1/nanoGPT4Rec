"""
Training script for ProductGPT on MovieLens dataset.
Example usage:
$ python train_recsys.py --batch_size=32 --block_size=50 --n_layer=6 --n_head=8
"""

import argparse
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np

np.random.seed(42)  # Initialize NumPy first
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
n_embd = 256
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
        # Load ratings data
        ratings_path = os.path.join(data_dir, "ratings.dat")
        self.ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        # Create ID mappings
        unique_users = sorted(self.ratings.user_id.unique())
        unique_movies = sorted(self.ratings.movie_id.unique())

        self.user_id_map = {
            old_id: new_id for new_id, old_id in enumerate(unique_users)
        }
        self.movie_id_map = {
            old_id: new_id for new_id, old_id in enumerate(unique_movies)
        }

        # Create reverse mappings
        self.user_id_map_reverse = {v: k for k, v in self.user_id_map.items()}
        self.movie_id_map_reverse = {v: k for k, v in self.movie_id_map.items()}

        # Convert IDs
        self.ratings["user_id"] = self.ratings.user_id.map(self.user_id_map)
        self.ratings["movie_id"] = self.ratings.movie_id.map(self.movie_id_map)

        # Sort by timestamp
        self.ratings = self.ratings.sort_values("timestamp")

        # Split based on time
        timestamp_threshold = self.ratings["timestamp"].quantile(0.8)
        train_ratings = self.ratings[self.ratings["timestamp"] <= timestamp_threshold]
        val_ratings = self.ratings[self.ratings["timestamp"] > timestamp_threshold]

        # Get movies that appear in training set
        train_movies = set(train_ratings["movie_id"].unique())

        # Filter validation to only include movies from training
        val_ratings = val_ratings[val_ratings["movie_id"].isin(train_movies)]

        logging.info(
            f"Removed {len(self.ratings) - len(train_ratings) - len(val_ratings)} cold-start items from validation"
        )

        # Group by user and create sequences
        ratings_to_use = train_ratings if split == "train" else val_ratings
        user_sequences = ratings_to_use.groupby("user_id")

        # Create sequences (using already mapped IDs)
        self.sequences = []
        for user_id, group in user_sequences:
            movies = group["movie_id"].values  # Already mapped IDs
            ratings = group["rating"].values
            if len(movies) >= 3:  # Minimum sequence length
                self.sequences.append(
                    (
                        user_id,  # Already mapped user_id
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

        # Load and process movie genres
        movies_path = os.path.join(data_dir, "movies.dat")
        movies_df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            encoding="ISO-8859-1",
            names=["movie_id", "title", "genres"],
        )

        # Only consider genres from training movies
        train_movies_original_ids = {
            self.movie_id_map_reverse[movie_id] for movie_id in train_movies
        }
        train_movies_df = movies_df[
            movies_df["movie_id"].isin(train_movies_original_ids)
        ]

        # Create genre vocabulary from training movies only
        genres = set()
        for g in train_movies_df.genres.str.split("|"):
            genres.update(g)
        self.genre_list = sorted(list(genres))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_list)}

        # Create genre mapping for each movie (use primary genre)
        self.movie_genres = {}
        for (
            _,
            row,
        ) in movies_df.iterrows():  # Process all movies but use training genres
            primary_genre = row.genres.split("|")[0]
            if primary_genre in self.genre_to_idx:  # Only use genres seen in training
                self.movie_genres[row.movie_id] = self.genre_to_idx[primary_genre]
            else:
                self.movie_genres[row.movie_id] = 0  # Default genre for unseen genres

        logging.info(f"Number of genres in training: {len(self.genre_list)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user_id, movie_sequence, ratings, timestamps = self.sequences[idx]

        # Convert to numpy arrays first
        movie_sequence = np.array(movie_sequence)
        ratings = np.array(ratings)
        timestamps = np.array(timestamps)

        if len(movie_sequence) > block_size:
            # Take the most recent block_size items
            movie_sequence = movie_sequence[-block_size:]
            ratings = ratings[-block_size:]
            timestamps = timestamps[-block_size:]
        else:
            # Pad sequences
            pad_length = block_size - len(movie_sequence)
            movie_sequence = np.pad(
                movie_sequence, (pad_length, 0), "constant", constant_values=0
            )
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
                        self.contexts[ts],  # [4] time features
                        [min(delta / 86400, 30.0)],  # [1] time delta in days
                        [rating / 5.0],  # [1] normalized rating
                        [
                            self.movie_genres.get(  # [1] genre feature
                                self.movie_id_map_reverse[movie_id],
                                0,  # Convert back to original ID for genre lookup
                            )
                        ],
                    ]
                )
                for ts, delta, rating, movie_id in zip(
                    timestamps, time_deltas, ratings, movie_sequence
                )
            ]
        )

        # Make sure sequences have the same length
        product_history = torch.tensor(movie_sequence[:-1], dtype=torch.long)
        targets = torch.tensor(movie_sequence[1:], dtype=torch.long)
        context_features = torch.tensor(context_sequence[:-1], dtype=torch.float)

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "context_features": context_features,
            "product_history": product_history,
            "targets": targets,
        }


def create_dataloaders(train_dataset):
    # Create validation split
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)), test_size=0.1, random_state=42
    )

    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    # Create data loaders with reduced workers
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Reduced to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Reduced to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
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

    # Add progress bar for evaluation
    progress = tqdm(val_loader, desc="Evaluating", leave=False)
    for batch in progress:
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

        # Update progress bar with current metrics
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "hits": f"{hits.item():.4f}",
                "ndcg": f"{ndcg.item():.4f}",
            }
        )

    model.train()
    return {
        "loss": total_loss / n_batches,
        "hit_rate": total_hits / n_batches,
        "ndcg": total_ndcg / n_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProductGPT")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--block_size", type=int, default=50, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global batch_size, block_size, device
    batch_size = args.batch_size
    block_size = args.block_size
    device = args.device

    # Get dataset stats
    train_dataset = MovieLensDataset(data_dir, split="train")
    n_products = len(train_dataset.movie_id_map)
    n_users = len(train_dataset.user_id_map)
    n_genres = len(train_dataset.genre_list)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_dataset)

    logging.info(f"Dataset stats:")
    logging.info(f"  Users: {n_users}")
    logging.info(f"  Movies: {n_products}")
    logging.info(f"  Genres: {n_genres}")
    logging.info(f"Training for {args.epochs} epochs")
    logging.info(f"Steps per epoch: {len(train_loader)}")

    # Initialize model with original context features + 1 for genre
    config = ProductGPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        max_products=n_products,
        max_users=n_users,
        n_context_features=7,  # original 6 + genre
        pad_token=0,
        anonymous_user_token=-1,
    )

    model = ProductGPT(config)
    model.to(device)
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device
    )
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    if compile and device == "cuda":
        print("Compiling model...")
        model = torch.compile(model)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # Training
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
        )

        for step, batch in progress_bar:
            t0 = time.time()
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

            epoch_loss += loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/(step+1):.4f}",
                    "iter/s": f"{1.0/(time.time()-t0):.1f}",
                }
            )

        # End of epoch evaluation
        train_metrics = evaluate(model, train_loader, ctx)
        val_metrics = evaluate(model, val_loader, ctx)

        logging.info(f"\nEpoch {epoch+1} Summary:")
        logging.info(f"Average training loss: {epoch_loss/len(train_loader):.4f}")
        logging.info("Training Metrics:")
        logging.info(f"  Loss: {train_metrics['loss']:.4f}")
        logging.info(f"  Hit@10: {train_metrics['hit_rate']:.4f}")
        logging.info(f"  NDCG@10: {train_metrics['ndcg']:.4f}")
        logging.info("Validation Metrics:")
        logging.info(f"  Loss: {val_metrics['loss']:.4f}")
        logging.info(f"  Hit@10: {val_metrics['hit_rate']:.4f}")
        logging.info(f"  NDCG@10: {val_metrics['ndcg']:.4f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "metrics": {
                    "train": train_metrics,
                    "val": val_metrics,
                },
            }
            logging.info(f"saving checkpoint to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))


if __name__ == "__main__":
    # Add multiprocessing start method
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        main()
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
