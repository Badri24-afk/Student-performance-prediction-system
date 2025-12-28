import joblib
import os
import numpy as np

def load_trained_model(filepath):
    """
    Loads a trained model from a .pkl file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)

def predict_single(model, input_features):
    """
    Predicts for a single instance.
    
    Args:
        model: Trained model.
        input_features (list or np.array): [Attendance, Marks, Activities_Encoded]
        
    Returns:
        int: Prediction (1 for Pass, 0 for Fail)
        float: Probability of class 1 (Pass)
    """
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    # Check if model supports proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_array)[0][1]
    else:
        probability = float(prediction) # Fallback
        
    return prediction, probability

def explain_prediction(prediction, attendance, marks, activities):
    """
    Generates a human-readable explanation for the prediction.
    """
    result = "PASS" if prediction == 1 else "FAIL"
    reasons = []
    
    # Simple heuristic rules for explanation
    if result == "FAIL":
        if attendance < 75:
            reasons.append("low attendance (< 75%)")
        if marks < 40:
            reasons.append("low internal marks (< 40)")
        if activities == 0 and not reasons:
             reasons.append("lack of extracurricular activities")
        
        if not reasons:
            explanation = "The student is predicted to FAIL, possibly due to a borderline combination of factors."
        else:
            explanation = f"The student is predicted to FAIL due to {', '.join(reasons)}."
            
    else: # PASS
        if attendance >= 75 and marks >= 40:
            explanation = "The student is predicted to PASS due to good attendance and sufficient marks."
        elif attendance >= 90:
             explanation = "The student is predicted to PASS, significantly aided by high attendance."
        else:
             explanation = "The student is predicted to PASS based on the overall feature profile."

    return explanation
