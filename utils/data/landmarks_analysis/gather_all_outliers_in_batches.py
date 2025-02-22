from .gather_all_outliers import gather_all_outliers
import pandas as pd

def gather_all_outliers_in_batches(df, columns, batch_size=10000, eps=0.05, min_samples=5):
    """
    Process the DataFrame in batches and apply DBSCAN to each batch using the existing functions.
    
    Args:
        df (pd.DataFrame): The DataFrame containing normalized landmark coordinates.
        columns (list): List of column names (each entry contains (x, y) tuple for coordinates).
        batch_size (int): The size of each batch.
        eps (float): The maximum distance between two points for them to be considered in the same neighborhood.
        min_samples (int): The minimum number of points to form a dense region.
    
    Returns:
        pd.DataFrame: A DataFrame containing all the unique outliers found across all batches.
    """
    all_outliers = pd.DataFrame()
    
    num_batches = len(df) // batch_size + 1

    for i in range(num_batches):
        batch_df = df[i * batch_size:(i + 1) * batch_size].copy()
        
        for col in columns:
            batch_df.loc[:, f'{col}_x'] = batch_df[col].apply(lambda x: x[0])
            batch_df.loc[:, f'{col}_y'] = batch_df[col].apply(lambda x: x[1])

            outliers = gather_all_outliers(batch_df, columns=[col], eps=eps, min_samples=min_samples)
        
            all_outliers = pd.concat([all_outliers, outliers], ignore_index=True)

    all_outliers = all_outliers.drop_duplicates(subset='filename')

    return all_outliers