import torch

import streamlit as st
from run import get_recommendations, load_movie_data, load_trained_model
from train import MovieLensDataset

# Page config
st.set_page_config(page_title="NanoGPT4Rec Movies", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model_and_data():
    """Load and cache model and data"""
    movies_df = load_movie_data()
    model, config = load_trained_model()
    device = next(model.parameters()).device

    # Create movie ID mapping from training data
    dataset = MovieLensDataset("data/movielens-1m", split="train")
    movie_id_map = dataset.movie_id_map

    return model, movies_df, device, movie_id_map


def star_rating_widget(key):
    """Create a row of clickable star buttons"""
    cols = st.columns(5)
    rating = 0

    for i in range(5):
        if cols[i].button("⭐", key=f"{key}_{i+1}"):
            rating = i + 1

    return rating


def main():
    st.title("🎬 NanoGPT4Rec movies")
    st.write("Rate some movies and get personalized recommendations!")

    # Load model and data (cached)
    model, movies_df, device, movie_id_map = load_model_and_data()

    # Initialize session state for storing ratings
    if "rated_movies" not in st.session_state:
        st.session_state.rated_movies = []  # Store mapped movie IDs
        st.session_state.ratings = []  # Store ratings
        st.session_state.movie_titles = []  # Store movie titles for display
        st.session_state.current_rating = 0  # Store temporary rating

    # Create two columns: one for input, one for display
    col1, col2 = st.columns([2, 1])

    with col1:
        # Movie selection and rating
        st.subheader("Rate a Movie")

        # Create a dropdown with all movies
        selected_title = st.selectbox(
            "Select a movie to rate",
            options=[""] + movies_df["title"].tolist(),  # Add empty option at start
            index=0,  # Default to empty option
        )

        if selected_title:  # Only show rating options if a movie is selected
            st.write("Click stars to rate:")
            rating = star_rating_widget(selected_title)

            if rating > 0:  # User clicked a star
                original_movie_id = movies_df[movies_df["title"] == selected_title][
                    "movie_id"
                ].iloc[0]

                # Map to training movie ID
                if original_movie_id not in movie_id_map:
                    st.warning("This movie was not in the training set!")
                    return

                mapped_movie_id = movie_id_map[original_movie_id]

                # Check if movie already rated
                if mapped_movie_id in st.session_state.rated_movies:
                    st.warning("You've already rated this movie!")
                else:
                    # Add to session state
                    st.session_state.rated_movies.append(mapped_movie_id)
                    st.session_state.ratings.append(float(rating))
                    st.session_state.movie_titles.append(selected_title)

                    # Force a rerun to update the display
                    st.rerun()

    with col2:
        # Display rated movies
        st.subheader("📝 Your Ratings")
        if st.session_state.rated_movies:
            for title, rating in zip(
                st.session_state.movie_titles, st.session_state.ratings
            ):
                st.write(f"{'⭐' * int(rating)} {title}")

            # Add a reset button
            if st.button("Reset Ratings"):
                st.session_state.rated_movies = []
                st.session_state.ratings = []
                st.session_state.movie_titles = []
                st.rerun()
        else:
            st.info("Rate some movies to get personalized recommendations!")

    # Generate and display recommendations
    st.subheader("🎯 Recommended Movies")

    if st.session_state.rated_movies:
        # Convert to tensor and get recommendations
        product_history = torch.tensor(
            [st.session_state.rated_movies], dtype=torch.long, device=device
        )

        recommendations = get_recommendations(
            model, movies_df, product_history, st.session_state.ratings
        )

        # Display recommendations in a nice format
        for i, rec in enumerate(recommendations, 1):
            score = rec["score"]
            stars = "⭐" * int(score * 5)  # Convert score to star rating
            st.write(
                f"{i}. {rec['title']} ({rec['genres']})\n"
                f"   {stars} (Score: {score:.3f})"
            )
    else:
        st.info("Rate some movies to get personalized recommendations!")


if __name__ == "__main__":
    main()
