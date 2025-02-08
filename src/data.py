"""
data.py

Module for loading and preprocessing the Iris dataset.
"""

from sklearn.datasets import load_iris
import pandas as pd


def load_data():
    """
    Load the Iris dataset and return as a tuple (X, y).

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y
