import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    """
    Transforms the cleaned data into ML-ready features.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        pd.DataFrame: DataFrame with encoded features.
        dict: Dictionary of encoders or mappings used (for reproducibility).
    """
    df_processed = df.copy()
    
    # Mapping for binary categorical features
    # Explicit mapping is safer for simple cases than LabelEncoder which preserves state
    
    activity_map = {'Yes': 1, 'No': 0}
    result_map = {'Pass': 1, 'Fail': 0}
    
    # Apply mappings safe way
    if 'Activities (Yes/No)' in df_processed.columns:
        df_processed['Activities_Encoded'] = df_processed['Activities (Yes/No)'].map(activity_map)
        # Drop original? Or keep? Usually we keep numeric for model. 
        # Let's drop original to avoid confusion in model training, 
        # but in a real pipeline we might keep them for analysis. 
        # For training, we need X only numeric usually.
        
    if 'Final Result (Pass/Fail)' in df_processed.columns:
        df_processed['Target'] = df_processed['Final Result (Pass/Fail)'].map(result_map)
        
    # Select features for model
    # Features: Attendance (%), Internal Marks (100), Activities_Encoded
    features = ['Attendance (%)', 'Internal Marks (100)', 'Activities_Encoded']
    
    return df_processed, features
