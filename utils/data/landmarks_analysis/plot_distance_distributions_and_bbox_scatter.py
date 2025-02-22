import matplotlib.pyplot as plt

def plot_distance_distributions_and_bbox_scatter(df):
    """
    Function to plot two histograms for left-right and top-bottom distances, 
    followed by a scatter plot of bounding box sizes.
    
    Args:
        df (pd.DataFrame): DataFrame containing the distances and bounding box information.
    
    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(df['distance_left'], bins=20, alpha=0.5, label='Left Distance', color='blue')
    axs[0].hist(df['distance_right'], bins=20, alpha=0.5, label='Right Distance', color='green')
    axs[0].set_title("Left and Right Distances Distribution")
    axs[0].set_xlabel("Distance (normalized)")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].hist(df['distance_top'], bins=20, alpha=0.5, label='Top Distance', color='red')
    axs[1].hist(df['distance_bottom'], bins=20, alpha=0.5, label='Bottom Distance', color='orange')
    axs[1].set_title("Top and Bottom Distances Distribution")
    axs[1].set_xlabel("Distance (normalized)")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    df['bbox_width'] = df['bounding_box'].apply(lambda x: x[2] - x[0])
    df['bbox_height'] = df['bounding_box'].apply(lambda x: x[3] - x[1])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df['bbox_width'], df['bbox_height'], alpha=0.7, color='purple')
    plt.title("Scatter plot of Bounding Box Sizes")
    plt.xlabel("Bounding Box Width (pixels)")
    plt.ylabel("Bounding Box Height (pixels)")
    plt.grid(True)
    plt.show()