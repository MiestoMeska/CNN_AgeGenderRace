import pandas as pd

def get_class_ranges(df):
    """
    Get the ranges (min, max) of the 'age', 'gender', and 'race' columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the columns 'age', 'gender', and 'race'.
    
    Returns:
        dict: A dictionary with the ranges of 'age', 'gender', and 'race'.
    """
    ranges = {}

    df.loc[:, 'age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[:, 'gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df.loc[:, 'race'] = pd.to_numeric(df['race'], errors='coerce')

    ranges['age'] = (df['age'].min(), df['age'].max())
    ranges['gender'] = (df['gender'].min(), df['gender'].max())
    ranges['race'] = (df['race'].min(), df['race'].max())
    
    return ranges