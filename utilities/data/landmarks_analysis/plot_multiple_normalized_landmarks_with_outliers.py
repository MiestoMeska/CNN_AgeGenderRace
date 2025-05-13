import matplotlib.pyplot as plt

def plot_multiple_normalized_landmarks_with_outliers(df, outliers_df, columns, labels=None):
    """
    Function to plot scatterplots of multiple normalized landmark positions in a 3-wide layout, with outliers shown in red.
    
    Args:
        df (pd.DataFrame): DataFrame containing the normalized landmark data.
        outliers_df (pd.DataFrame): DataFrame containing the outlier rows.
        columns (list): List of column names containing the normalized landmark data.
        labels (list): Optional list of labels for each landmark being plotted. If not provided, column names will be used as labels.
    
    Returns:
        None
    """
    def safe_eval(x):
        try:
            if isinstance(x, str):
                return eval(x)
            elif isinstance(x, (list, tuple)):
                return x
            else:
                return [None, None]
        except Exception as e:
            print(f"Error evaluating: {x} - {e}")
            return [None, None]
    
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(13, num_rows * 5))
    axes = axes.ravel()
    
    if labels is None:
        labels = columns

    for i, column in enumerate(columns):
        ax = axes[i]
        
        # Plot normal data points in blue
        ax.scatter(df[column].apply(lambda x: safe_eval(x)[0]), 
                   df[column].apply(lambda x: safe_eval(x)[1]), 
                   c='blue', label=labels[i])

        # Plot outliers in red
        ax.scatter(outliers_df[column].apply(lambda x: safe_eval(x)[0]), 
                   outliers_df[column].apply(lambda x: safe_eval(x)[1]), 
                   c='red', label=f'{labels[i]} - Outliers')

        ax.set_title(f"Scatter plot of {labels[i]} positions")
        ax.set_xlabel("Horizontal position (normalized)")
        ax.set_ylabel("Vertical position (normalized)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(df[column].apply(lambda x: safe_eval(x)[1]).mean(), color='red', linestyle='--', label='Average Y')
        ax.axvline(df[column].apply(lambda x: safe_eval(x)[0]).mean(), color='red', linestyle='--', label='Average X')
        ax.grid(True)
        ax.legend()

    # Remove any extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()