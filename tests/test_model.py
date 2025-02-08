"""
test_model.py

Unit tests for the logistic regression model.
"""

import pytest
import pandas as pd
from src.model import LogisticRegressionModel
from src.data import load_data


@pytest.fixture(scope="module")
def model():
    lr_model = LogisticRegressionModel()
    lr_model.train()
    return lr_model


def test_model_accuracy(model):
    # Ensure that the model achieves a reasonable accuracy on training data
    X, y = load_data()
    predictions = model.predict(X)
    accuracy = sum(predictions == y) / len(y)
    assert accuracy > 0.7, "Model accuracy should be greater than 70%"


def test_model_predict_proba(model):
    # Check that predict_proba returns probabilities that sum to 1
    X, _ = load_data()
    proba = model.predict_proba(X)
    sums = proba.sum(axis=1)
    for s in sums:
        assert abs(s - 1.0) < 1e-5, "Probabilities do not sum to 1"
