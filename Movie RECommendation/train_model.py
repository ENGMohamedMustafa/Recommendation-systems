from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import pandas as pd

# Load ratings
ratings = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
