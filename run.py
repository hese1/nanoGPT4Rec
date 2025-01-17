import os

import numpy as np
import pandas as pd
import torch

from model import ProductGPT
from train import MovieLensDataset


def load_movie_data(data_dir="data/movielens-1m"):
    """Load movie titles for interpretable recommendations"""
    movies_df = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::",
        engine="python",
        encoding="ISO-8859-1",
        names=["movie_id", "title", "genres"],
    )
    return movies_df


def load_trained_model(checkpoint_path="out-recsys/best_model.pt"):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please train the model first using train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    model = ProductGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, config


def create_context_features(timestamp=None):
    """Create temporal context features"""
    if timestamp is None:
        timestamp = pd.Timestamp.now().timestamp()

    hour = (timestamp % 86400) / 3600  # Hour of day (0-23)
    day = (timestamp % 604800) / 86400  # Day of week (0-6)

    return np.array(
        [
            np.sin(2 * np.pi * hour / 24),  # Hour of day
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),  # Day of week
            np.cos(2 * np.pi * day / 7),
        ]
    )


def get_recommendations(model, movies_df, product_history, ratings):
    """Generate recommendations based on product history"""
    device = next(model.parameters()).device

    # Ensure product_history is 2D [batch_size, seq_len]
    if product_history.dim() == 1:
        product_history = product_history.unsqueeze(0)

    # Create context features
    context = create_context_features()
    context_features = []

    # Create genre mapping like in training
    genres = set()
    for g in movies_df.genres.str.split("|"):
        genres.update(g)
    genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(list(genres)))}

    # Track user's preferred genres and watched movies
    genre_counts = {}
    watched_movies = set(product_history[0].tolist())  # Keep track of training IDs

    # Create reverse mapping for movie IDs
    dataset = MovieLensDataset("data/movielens-1m", split="train")
    movie_id_map_reverse = {v: k for k, v in dataset.movie_id_map.items()}

    # Add time deltas (assume 1 day between ratings for demo)
    seq_len = product_history.size(1)
    time_deltas = [0.0] + [1.0] * (seq_len - 1)  # in days

    for delta, rating, movie_id in zip(time_deltas, ratings, product_history[0]):
        # Convert training ID back to original ID for lookup
        original_movie_id = movie_id_map_reverse[movie_id.item()]
        movie = movies_df[movies_df["movie_id"] == original_movie_id]

        # Count genres from positively rated movies (rating > 3)
        if rating > 3:
            movie_genres = movie.iloc[0]["genres"].split("|")
            for genre in movie_genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        primary_genre = movie.iloc[0]["genres"].split("|")[0]
        genre_id = genre_to_idx.get(primary_genre, 0)

        context_features.append(
            np.concatenate(
                [
                    context,  # Time features [4]
                    [min(delta, 30.0)],  # Time delta in days
                    [(rating - 3.0) / 2.0],  # Normalized rating: maps 1->-1, 3->0, 5->1
                    [genre_id],  # Genre feature
                ]
            )
        )

    context_features = torch.tensor(
        [context_features], dtype=torch.float, device=device
    )

    # Get recommendations
    with torch.no_grad():
        recommendations, scores = model.get_recommendations(
            context_features=context_features,
            product_history=product_history,
            top_k=50,  # Get more recommendations initially for filtering
        )

        # Ensure we have the right shape
        if recommendations.dim() == 1:
            recommendations = recommendations.unsqueeze(0)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

    # Format recommendations with genre-based reranking
    recs = []
    for movie_id, score in zip(recommendations[0], scores[0]):
        training_movie_id = movie_id.item()

        # Skip already watched movies (using training IDs)
        if training_movie_id in watched_movies:
            continue

        # Convert training ID back to original ID for lookup
        original_movie_id = movie_id_map_reverse[training_movie_id]
        movie = movies_df[movies_df["movie_id"] == original_movie_id]

        if not movie.empty:
            movie_genres = movie.iloc[0]["genres"].split("|")

            # Calculate genre similarity score
            genre_similarity = 0
            for genre in movie_genres:
                if genre in genre_counts:
                    genre_similarity += genre_counts[genre]

            # Adjust score based on genre similarity
            adjusted_score = score.item() * (1 + 0.2 * genre_similarity)

            recs.append(
                {
                    "title": movie.iloc[0]["title"],
                    "genres": movie.iloc[0]["genres"],
                    "score": adjusted_score,
                }
            )

            if len(recs) >= 10:  # Only keep top 10 after filtering
                break

    # Sort by adjusted score
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:10]


def main():
    """Test recommendation generation"""
    # Load data and model
    movies_df = load_movie_data()
    model, config = load_trained_model()
    device = next(model.parameters()).device

    # Get movie ID mapping from training data
    dataset = MovieLensDataset("data/movielens-1m", split="train")
    movie_id_map = dataset.movie_id_map

    # Create test input (movie_id=1 with rating 5)
    # Map the original movie ID to training ID
    mapped_movie_id = movie_id_map[1]
    product_history = torch.tensor(
        [[mapped_movie_id]], dtype=torch.long, device=device
    )  # [batch_size=1, seq_len=1]
    ratings = [5.0]  # Single rating for the test movie

    # Get recommendations
    recs = get_recommendations(model, movies_df, product_history, ratings)

    # Print recommendations
    print("\nTop 10 Recommendations:")
    print("-" * 50)
    for rec in recs:
        print(f"Score: {rec['score']:.4f} - {rec['title']} ({rec['genres']})")


if __name__ == "__main__":
    main()
