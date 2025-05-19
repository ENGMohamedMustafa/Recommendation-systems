# 🎬 Movie Recommender System

A web-based movie recommendation application built with Streamlit that provides personalized movie suggestions using collaborative filtering and content-based filtering techniques.

## Features

- **User-Based Recommendations**: Get movie suggestions based on users with similar tastes using Singular Value Decomposition (SVD) for collaborative filtering
- **Movie-Based Recommendations**: Find movies similar to ones you already like using cosine similarity between movies
- **Popular Movies**: Discover the highest-rated movies with significant viewership
- **Interactive UI**: Clean, responsive interface with intuitive controls
- **Automatic Data Loading**: Automatically downloads and processes the MovieLens 100K dataset if not present

## Demo

![Movie Recommender System Demo](https://github.com/ENGMohamedMustafa/Recommendation-systems/blob/main/Moive%20Recommendation-system/Screenshot%206.png)

## Requirements

- Python 3.6+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- SciPy
- Requests

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ENGMohamedMustafa/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run MoiveRecommendation-system.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Choose a recommendation method:
   - **User-Based Recommendations**: Select a user ID and get personalized recommendations
   - **Movie-Based Recommendations**: Select a movie and find similar ones
   - **Popular Movies**: View the most popular movies based on ratings and views

## How It Works

The system uses multiple recommendation approaches:

### Collaborative Filtering
Uses Singular Value Decomposition (SVD) to analyze user-item interactions and identify patterns in user preferences. This allows the system to recommend movies that similar users have enjoyed.

### Content-Based Filtering
Recommends movies based on similarities between movies, calculated using cosine similarity of user ratings patterns.

### Popularity-Based Recommendations
Recommends movies with high average ratings and significant viewership.

## Dataset

The application uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which contains 100,000 movie ratings from 943 users on 1,682 movies. The dataset is automatically downloaded when you first run the application.

## Customization

You can modify various aspects of the recommendation system:

- Change the number of latent factors in SVD by adjusting the `k` parameter in the `svds` function
- Modify the minimum number of ratings required for popular movies
- Adjust the UI styling by modifying the CSS in the `st.markdown` section

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework
