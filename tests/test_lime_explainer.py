"""
test_lime_explainer.py

Unit tests for the LIME explainer functionality.
"""

import numpy as np
import pandas as pd
import pytest
from src.lime_explainer import LimeExplainer
from src.data import load_data
from src.model import LogisticRegressionModel


@pytest.fixture(scope="module")
def explainer():
    X, y = load_data()
    feature_names = list(X.columns)
    class_names = [str(i) for i in sorted(y.unique())]
    # Initialize explainer using full training data
    return LimeExplainer(training_data=X.values, feature_names=feature_names, class_names=class_names)


@pytest.fixture(scope="module")
def trained_model():
    model = LogisticRegressionModel()
    model.train()
    return model


def test_explain_instance_output(explainer, trained_model):
    X, _ = load_data()
    # Take a sample instance (convert to 1d array)
    instance = X.values[0]
    explanation = explainer.explain_instance(instance, predict_fn=trained_model.predict_proba)
    # Check that explanation contains a list of feature contributions
    label = explanation.available_labels()[0]
    local_exp = explanation.local_exp[label]
    assert isinstance(local_exp, list) and len(local_exp) > 0, "Explanation should have non-empty feature contributions"
