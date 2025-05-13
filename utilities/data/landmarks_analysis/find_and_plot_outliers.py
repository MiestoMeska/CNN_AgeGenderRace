from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_and_plot_outliers(df, column, threshold=3):
    """
    Find and plot outliers using the Z-score method for a specified column in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to analyze for outliers (e.g., 'eye_distance').
        threshold (float): Z-score threshold to define outliers (default is 3).
    
    Returns:
        pd.DataFrame: DataFrame containing the rows where the outliers were found (including the filename).
    """
    data = df[column].values

    z_scores = np.abs(stats.zscore(data))

    outliers_mask = z_scores > threshold
    outliers_df = df[outliers_mask]

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, color='purple', alpha=0.7, label=column)
    
    plt.axvline(np.mean(data) - threshold * np.std(data), color='r', linestyle='--', label='Lower Bound')
    plt.axvline(np.mean(data) + threshold * np.std(data), color='r', linestyle='--', label='Upper Bound')
    
    plt.xlabel(f"{column.replace('_', ' ').title()} (pixels)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column.replace('_', ' ').title()} with Z-score Outliers")
    plt.legend()
    plt.show()

    return outliers_df[['filename', column]]
