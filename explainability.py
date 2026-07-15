import shap
import numpy as np
import pandas as pd

def get_explainer(model, X_train):
    """
    Initializes a SHAP TreeExplainer for the given XGBoost model.
    """
    explainer = shap.TreeExplainer(model, X_train)
    return explainer

def explain_alert(shap_values, feature_names):
    """
    Given pre-calculated SHAP values for a single row, generates a plain-English explanation.
    """
    # shap_values could be a list if multi-class, but it's single array for binary XGBoost
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Take positive class
    
    vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    top_indices = np.argsort(np.abs(vals))[-3:][::-1]
    
    top_features = []
    for idx in top_indices:
        feature_name = feature_names[idx]
        contribution = vals[idx]
        
        # Format the name a bit nicely if it's one-hot encoded
        clean_name = feature_name.replace('_', ' ')
        
        if contribution > 0:
            top_features.append(f"{clean_name} (+{contribution:.2f})")
    
    if not top_features:
        return "Flagged due to a combination of minor anomalous factors."
        
    explanation = f"Flagged: {', '.join(top_features)}."
    return explanation
