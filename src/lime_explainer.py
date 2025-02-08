"""
lime_explainer.py

Module for generating LIME explanations for model predictions.
"""

import numpy as np
import lime
import lime.lime_tabular
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LimeExplainer:
    """
    Class for creating LIME explanations for a given classifier.
    """

    def __init__(self, training_data, feature_names, class_names, discretize_continuous=True):
        """
        Initialize the LimeExplainer.

        Args:
            training_data (array-like): Data used for fitting the explainer.
            feature_names (list): Names of the features.
            class_names (list): Names of the classes.
            discretize_continuous (bool): Whether to discretize continuous features.
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=discretize_continuous,
            verbose=True,
            mode='classification'
        )

    def explain_instance(self, instance, predict_fn, num_features=4, num_samples=1000):
        """
        Generate a LIME explanation for a single instance.

        Args:
            instance (array-like): A single data instance to explain.
            predict_fn (callable): Prediction function that takes an array-like input.
            num_features (int): Number of features to include in explanation.
            num_samples (int): Number of samples to generate for explanation.

        Returns:
            Explanation object from LIME.
        """
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        logger.info("Generated LIME explanation.")
        return explanation

    def save_explanation(self, explanation, filepath='lime_explanation.html'):
        """
        Save the generated explanation as an HTML file.

        Args:
            explanation: LIME explanation object.
            filepath (str): File path to save the explanation.
        """
        html_data = explanation.as_html()
        with open(filepath, 'w') as f:
            f.write(html_data)
        logger.info(f"LIME explanation saved to {filepath}")
