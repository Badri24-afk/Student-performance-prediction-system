import pandas as pd
import os

def load_data(filepath):
    """
    Loads data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
