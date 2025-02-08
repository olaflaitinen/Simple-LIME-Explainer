"""
run_explainer.py

Example script to train the logistic regression model and generate a LIME explanation for a sample instance.
"""

import pandas as pd
from src.model import LogisticRegressionModel
from src.lime_explainer import LimeExplainer
from src.data import load_data
from src.utils import plot_feature_importance

def main():
    # Load data and train model
    X, y = load_data()
    model = LogisticRegressionModel()
    metrics = model.train()
    print(f"Model training metrics: {metrics}")

    # Initialize LIME explainer
    feature_names = list(X.columns)
    class_names = [str(i) for i in sorted(y.unique())]
    explainer = LimeExplainer(training_data=X.values, feature_names=feature_names, class_names=class_names)

    # Select a sample instance for explanation
    sample_instance = X.values[0]
    print("Explaining instance:", sample_instance)

    # Generate LIME explanation
    explanation = explainer.explain_instance(sample_instance, predict_fn=model.predict_proba)
    
    # Save explanation as HTML
    explainer.save_explanation(explanation, filepath='lime_explanation.html')
    print("LIME explanation saved as lime_explanation.html")

    # Optionally, plot the feature importance
    plot_feature_importance(explanation)


if __name__ == "__main__":
    main()
