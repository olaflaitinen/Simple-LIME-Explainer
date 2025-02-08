"""
utils.py

Utility functions for the project.
"""

import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_feature_importance(explanation):
    """
    Plot the feature importance from a LIME explanation.

    Args:
        explanation: LIME explanation object.
    """
    # Extract feature contributions from the explanation.
    local_exp = explanation.local_exp[explanation.available_labels()[0]]
    features = [feat for feat, weight in local_exp]
    weights = [weight for feat, weight in local_exp]

    plt.figure(figsize=(8, 4))
    plt.barh(range(len(features)), weights, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Weight')
    plt.title('Feature Importance from LIME Explanation')
    plt.tight_layout()
    plt.show()
    logger.info("Displayed feature importance plot.")
