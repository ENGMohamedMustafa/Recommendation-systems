import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import os
import requests
import zipfile
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
st.markdown("""
<style>
    .main {
        background-color: #F0F4FF;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        color: white;
    }
    .recommendation-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 255, 0, 0.1);
        margin-bottom: 10px;
    }
    .movie-title {
        font-weight: bold;
        font-size: 16px;
        color: #0E3A8A;
    }
    .movie-score {
        color: #2563EB;
        font-weight: bold;
    }
    .movie-genre {
        color: #6B7280;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_data():
    """Load pre-trained recommender system data"""
    # In a real application, this would load your pre-trained models and data
    
    # For demonstration purposes, we'll use MovieLens 100K dataset
    if not os.path.exists('ml-100k'):
        with st.spinner('Downloading MovieLens 100K dataset...'):
            # URL of the MovieLens 100K dataset
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            response = requests.get(url)
            z = zipfile.ZipFile(BytesIO(response.content))
            z.extractall()
            st.success('Dataset downloaded and extracted successfully!')
    
    # Load ratings data
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load movie data
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                 ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                  'Thriller', 'War', 'Western']
    movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')
    
    # Create a user-item matrix
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Extract genres for each movie
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                 'Thriller', 'War', 'Western']
    
    movie_genres = {}
    for idx, row in movies_df.iterrows():
        movie_id = row['movie_id']
        genres = [genre for genre, value in row[genre_cols].items() if value == 1]
        movie_genres[movie_id] = genres
    
    return ratings_df, movies_df, user_item_matrix, user_item_matrix_filled, movie_genres

# Define recommendation functions
def get_similar_movies(movie_id, item_similarity_df, n=10):
    """Get movies similar to a given movie"""
    if movie_id not in item_similarity_df.index:
        return []
    
    similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]
    return similar_movies

def get_user_recommendations(user_id, svd_predictions, user_item_matrix, n=10):
    """Get recommendations for a user using SVD"""
    # Get user's rated items
    if user_id not in user_item_matrix.index:
        return []
        
    user_rated = user_item_matrix.loc[user_id].dropna().index
    
    # Get predictions for unrated items
    if user_id not in svd_predictions.index:
        return []
        
    user_predictions = svd_predictions.loc[user_id].drop(user_rated)
    
    # Get top N recommendations
    top_recommendations = user_predictions.sort_values(ascending=False).head(n)
    return top_recommendations

def get_popular_movies(ratings_df, n=10):
    """Get the most popular movies based on number of ratings and average rating"""
    # Calculate number of ratings and average rating for each movie
    movie_stats = ratings_df.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    })
    movie_stats.columns = ['count', 'mean']
    
    # Only consider movies with a minimum number of ratings
    popular_movies = movie_stats[movie_stats['count'] > 100].sort_values(
        by=['mean', 'count'], ascending=False
    ).head(n)
    
    return popular_movies.index.tolist()

# Pre-compute similarity matrices and SVD predictions
@st.cache_data
def precompute_models(user_item_matrix_filled, user_item_matrix):
    """Pre-compute recommendation models"""
    # Compute item similarity for content-based filtering
    item_similarity = cosine_similarity(user_item_matrix_filled.T)
    item_similarity_df = pd.DataFrame(
        item_similarity, 
        index=user_item_matrix.columns, 
        columns=user_item_matrix.columns
    )
    
    # Compute SVD for matrix factorization (collaborative filtering)
    user_ratings_mean = np.mean(user_item_matrix_filled.values, axis=1)
    ratings_centered = user_item_matrix_filled.values - user_ratings_mean.reshape(-1, 1)
    
    # SVD decomposition
    U, sigma, Vt = svds(ratings_centered, k=50)
    sigma_diag = np.diag(sigma)
    
    # Reconstruct the predictions matrix
    predictions = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
    svd_predictions = pd.DataFrame(
        predictions, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.columns
    )
    
    return item_similarity_df, svd_predictions

# Main function to run the Streamlit application
def main():
    st.title("🎬 Movie Recommender System")

    # Load the data
    ratings_df, movies_df, user_item_matrix, user_item_matrix_filled, movie_genres = load_data()
    
    # Pre-compute models
    item_similarity_df, svd_predictions = precompute_models(user_item_matrix_filled, user_item_matrix)
    
    # Get unique users and movies for dropdowns
    unique_users = sorted(user_item_matrix.index.tolist())
    all_movies = movies_df[['movie_id', 'title']].set_index('movie_id')
    
    # Sidebar
    st.sidebar.header("Recommendation Options")
    
    # Choose recommendation method
    recommendation_method = st.sidebar.radio(
        "Choose a recommendation method:",
        ["User-Based Recommendations", "Movie-Based Recommendations", "Popular Movies"]
    )
    
    if recommendation_method == "User-Based Recommendations":
        st.header("User-Based Movie Recommendations")
        st.write("Get recommendations based on users with similar tastes")
        
        # User selection
        selected_user = st.selectbox("Select a user ID:", unique_users)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                # Get user recommendations
                user_recs = get_user_recommendations(
                    selected_user, 
                    svd_predictions, 
                    user_item_matrix, 
                    n=num_recommendations
                )
                
                if len(user_recs) == 0:
                    st.warning("No recommendations available for this user.")
                else:
                    st.subheader(f"Top {num_recommendations} recommendations for User {selected_user}")
                    
                    # Display recommendations
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        for movie_id, score in user_recs.items():
                            movie_title = all_movies.loc[movie_id, 'title']
                            genres = movie_genres.get(movie_id, [])
                            
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">{movie_title}</div>
                                <div class="movie-genre">Genres: {', '.join(genres)}</div>
                                <div class="movie-score">Score: {score:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
    elif recommendation_method == "Movie-Based Recommendations":
        st.header("Similar Movies Recommendations")
        st.write("Find movies similar to ones you already like")
        
        # Movie selection
        movie_options = movies_df[['movie_id', 'title']].set_index('movie_id')['title']
        movie_dict = {title: idx for idx, title in movie_options.items()}
        selected_movie_title = st.selectbox("Select a movie:", movie_options.tolist())
        selected_movie_id = movie_dict[selected_movie_title]
        
        # Number of recommendations
        num_recommendations = st.slider("Number of similar movies:", 5, 20, 10)
        
        if st.button("Find Similar Movies"):
            with st.spinner("Finding similar movies..."):
                # Get similar movies
                similar_movies = get_similar_movies(
                    selected_movie_id, 
                    item_similarity_df, 
                    n=num_recommendations
                )
                
                if len(similar_movies) == 0:
                    st.warning("No similar movies found.")
                else:
                    st.subheader(f"Movies similar to '{selected_movie_title}'")
                    
                    # Display recommendations
                    for movie_id, similarity in similar_movies.items():
                        movie_title = all_movies.loc[movie_id, 'title']
                        genres = movie_genres.get(movie_id, [])
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="movie-title">{movie_title}</div>
                            <div class="movie-genre">Genres: {', '.join(genres)}</div>
                            <div class="movie-score">Similarity: {similarity:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
    else:  # Popular Movies
        st.header("Most Popular Movies")
        st.write("Discover the highest rated movies with significant viewership")
        
        # Number of recommendations
        num_recommendations = st.slider("Number of popular movies:", 5, 20, 10)
        
        if st.button("Show Popular Movies"):
            with st.spinner("Finding popular movies..."):
                # Get popular movies
                popular_movie_ids = get_popular_movies(ratings_df, n=num_recommendations)
                
                # Display popular movies
                st.subheader(f"Top {num_recommendations} popular movies")
                
                for movie_id in popular_movie_ids:
                    movie_title = all_movies.loc[movie_id, 'title']
                    genres = movie_genres.get(movie_id, [])
                    
                    # Get average rating
                    avg_rating = ratings_df[ratings_df['movie_id'] == movie_id]['rating'].mean()
                    num_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating'].count()
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="movie-title">{movie_title}</div>
                        <div class="movie-genre">Genres: {', '.join(genres)}</div>
                        <div class="movie-score">Rating: {avg_rating:.2f}/5 ({num_ratings} ratings)</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app demonstrates a movie recommender system built with "
        "Streamlit and Python. The recommendations are based on the "
        "MovieLens 100K dataset."
    )

if __name__ == "__main__":
    main()