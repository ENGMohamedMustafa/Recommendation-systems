import pytest
from app.model import load_data, preprocess_data, train_model

def test_training_pipeline():
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")
