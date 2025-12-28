import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the raw dataframe:
    - Removes duplicates
    - Handles missing values (simple drop or impute - for now drop as per typical simple pipeline, or mean)
    - Returns cleaned df
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    initial_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values - for this specific dataset structure
    # 'Attendance (%)', 'Internal Marks (100)' should be numeric.
    # 'Activities (Yes/No)' should be categorical.
    
    # Basic cleanup if needed (e.g. converting string to numbers if dirty)
    # Check for nulls
    df = df.dropna()
    
    print(f"Data cleaned. Rows removed: {initial_shape[0] - df.shape[0]}")
    return df

def save_data(df, filepath):
    """
    Saves dataframe to csv.
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
