from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns metrics.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("Model Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }
