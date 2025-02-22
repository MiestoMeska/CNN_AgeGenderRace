import matplotlib.pyplot as plt
import numpy as np

def plot_pie_chart(df, column, label_dict=None):
    """
    Plots a pie chart for the distribution of values in a specified DataFrame column.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name for which to plot the pie chart.
        label_dict (dict): A dictionary to map numeric labels to meaningful names (optional).
    
    Returns:
        None
    """
    df_copy = df[column].copy()
    
    if label_dict:
        df_copy = df_copy.map(label_dict)
    
    value_counts = df_copy.value_counts().sort_index()

    plt.figure(figsize=(8, 8))
    colors = plt.cm.Paired(np.arange(len(value_counts)))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f"Distribution of {column}")
    plt.axis('equal')
    plt.show()