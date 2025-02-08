"""
model.py

Module for training and using a logistic regression model on the Iris dataset.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .data import load_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """
    A class to train, save, load, and use a logistic regression model.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = LogisticRegression(random_state=self.random_state, max_iter=200)

    def train(self, test_size=0.2):
        """
        Train the logistic regression model on the Iris dataset.

        Args:
            test_size (float): Fraction of the data to be used as test set.

        Returns:
            dict: Dictionary containing training and testing accuracy.
        """
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        logger.info(f"Training accuracy: {train_acc:.4f}, Testing accuracy: {test_acc:.4f}")
        return {"train_accuracy": train_acc, "test_accuracy": test_acc}

    def predict(self, X):
        """
        Predict the target for a given feature set.

        Args:
            X (pd.DataFrame or array-like): Feature set.

        Returns:
            array-like: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict the class probabilities for a given feature set.

        Args:
            X (pd.DataFrame or array-like): Feature set.

        Returns:
            array-like: Predicted probabilities.
        """
        return self.model.predict_proba(X)

    def save(self, filepath='model.joblib'):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model.
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath='model.joblib'):
        """
        Load a model from disk.

        Args:
            filepath (str): Path to the model file.
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
