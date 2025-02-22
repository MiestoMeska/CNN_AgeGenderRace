from sklearn.cluster import DBSCAN
import pandas as pd

def find_outliers_dbscan(df, x_column, y_column, eps=0.05, min_samples=5):
    """
    Identify outliers in a DataFrame using DBSCAN clustering based on x and y coordinates.
    
    Args:
        df (pd.DataFrame): The DataFrame containing normalized landmark coordinates.
        x_column (str): The column name for horizontal (x) coordinates.
        y_column (str): The column name for vertical (y) coordinates.
        eps (float): The maximum distance between two points for them to be considered in the same neighborhood.
        min_samples (int): The minimum number of points to form a dense region.
    
    Returns:
        pd.DataFrame: A DataFrame containing rows that are outliers (labeled as -1 by DBSCAN).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    
    df['cluster'] = db.fit_predict(df[[x_column, y_column]])
    
    return df[df['cluster'] == -1]

def gather_all_outliers(df, columns, eps=0.05, min_samples=5):
    """
    Run DBSCAN on multiple pairs of coordinates and return a combined DataFrame of all outliers.
    
    Args:
        df (pd.DataFrame): The DataFrame containing normalized landmark coordinates.
        columns (list): List of column names (each entry contains (x, y) tuple for coordinates).
        eps (float): The maximum distance between two points for them to be considered in the same neighborhood.
        min_samples (int): The minimum number of points to form a dense region.
    
    Returns:
        pd.DataFrame: A DataFrame containing all the unique outliers found across all coordinate pairs.
    """
    all_outliers = pd.DataFrame()

    for col in columns:
        df[f'{col}_x'] = df[col].apply(lambda x: x[0])
        df[f'{col}_y'] = df[col].apply(lambda x: x[1])

        outliers = find_outliers_dbscan(df, x_column=f'{col}_x', y_column=f'{col}_y', eps=eps, min_samples=min_samples)

        all_outliers = pd.concat([all_outliers, outliers])

    all_outliers = all_outliers.drop_duplicates(subset='filename')

    return all_outliers