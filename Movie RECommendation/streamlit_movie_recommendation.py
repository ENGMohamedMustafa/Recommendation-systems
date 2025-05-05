import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import SVD, Dataset, Reader

# Load movies and ratings
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎬 Movie Recommendation System")

# Select user ID
user_ids = ratings['userId'].unique()
selected_user = st.selectbox("Select a user", user_ids)

# Recommend top N movies
def recommend_movies(user_id, n=10):
    # Get movies the user hasn't rated
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_movies)]
    
    # Predict ratings
    predictions = []
    for _, row in unrated_movies.iterrows():
        pred = model.predict(user_id, row['movieId']).est
        predictions.append((row['title'], pred))
    
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_movies

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_user)
    st.subheader("Recommended Movies:")
    for title, score in recommendations:
        st.write(f"⭐ {score:.2f} - {title}")
