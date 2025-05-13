import matplotlib.pyplot as plt

def plot_aspect_ratio_distribution(df):
    """
    Function to calculate and plot the distribution of bounding box aspect ratios.
    
    Args:
        df (pd.DataFrame): DataFrame containing the bounding box information.
    
    Returns:
        None
    """
    df['aspect_ratio'] = df['bounding_box'].apply(lambda x: (x[2] - x[0]) / (x[3] - x[1]))

    plt.figure(figsize=(8, 6))
    plt.hist(df['aspect_ratio'], bins=30, color='green')
    plt.title("Distribution of Bounding Box Aspect Ratios")
    plt.xlabel("Aspect Ratio (width/height)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print("Mean Aspect Ratio:", df['aspect_ratio'].mean())
    print("Standard Deviation of Aspect Ratio:", df['aspect_ratio'].std())
