

# 🎬 Movie Recommendation System with Streamlit

This project is a web-based **Movie Recommendation System** built using **Streamlit** and trained using **collaborative filtering (SVD)** on the MovieLens dataset. Users can select their ID and receive personalized movie recommendations instantly.



## 📌 Features

- Select any user from the dropdown.
- Instantly get **Top N recommended movies** for that user.
- Powered by a trained `SVD` model using the Surprise library.
- Simple and interactive UI using Streamlit.


## 📁 Project Structure
---
recommendation_app/
├── app.py                # Streamlit web app

├── model.pkl             # Trained SVD model

├── movies.csv            # Movie metadata

├── ratings.csv           # User ratings

├── requirements.txt      # Python dependencies

├── train_model.py        # (Optional) script to retrain the model

---

## 🧠 Model Details

- **Algorithm**: SVD (Singular Value Decomposition)
- **Library**: `scikit-surprise`
- **Input**: `userId`, `movieId`, `rating`
- **Output**: Predicted rating for unseen movies


## 🚀 Getting Started


### 1. Install Dependencies

pip install -r requirements.txt


### 2. (Optional) Retrain the Model

python train_model.py


### 3. Run the Web App

streamlit run app.py



## 📊 Dataset

- **Source**: [MovieLens](https://grouplens.org/datasets/movielens/)
- `movies.csv`: Contains `movieId`, `title`, `genres`
- `ratings.csv`: Contains `userId`, `movieId`, `rating`, `timestamp`


## 📎 License

This project is licensed under the MIT License.
