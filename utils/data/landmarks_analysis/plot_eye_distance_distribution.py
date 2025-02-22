import matplotlib.pyplot as plt

def plot_eye_distance_distribution(df, column='eye_distance', bins=30, color='purple'):
    """
    Function to plot the distribution of the eye distance (or any numeric column) 
    and print the mean and standard deviation.
    
    Args:
        df (pd.DataFrame): DataFrame containing the column to be plotted.
        column (str): Name of the column representing the eye distance.
        bins (int): Number of bins for the histogram.
        color (str): Color of the histogram bars.
    
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=bins, color=color)
    plt.title(f"Distribution of {column.replace('_', ' ').title()} (Zoom Level)")
    plt.xlabel(f"{column.replace('_', ' ').title()} (pixels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print(f"Mean {column.replace('_', ' ').title()}: {df[column].mean():.2f}")
    print(f"Standard Deviation of {column.replace('_', ' ').title()}: {df[column].std():.2f}")