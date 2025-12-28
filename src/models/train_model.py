import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(X, y, model_type='logistic'):
    """
    Trains a model.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        model_type (str): 'logistic' or 'decision_tree'.
        
    Returns:
        model: Trained sklearn model.
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'logistic' or 'decision_tree'.")
    
    model.fit(X, y)
    return model

def save_model(model, filepath):
    """
    Saves the trained model to a file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
