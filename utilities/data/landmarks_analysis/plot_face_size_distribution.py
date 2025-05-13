import matplotlib.pyplot as plt

def plot_face_size_distribution(df):
    """
    Function to calculate and plot the distribution of face sizes (bounding box area).
    
    Args:
        df (pd.DataFrame): DataFrame containing the bounding box information.
    
    Returns:
        None
    """
    df['face_size'] = df['bounding_box'].apply(lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    plt.figure(figsize=(8, 6))
    plt.hist(df['face_size'], bins=30, color='orange')
    plt.title("Distribution of Face Sizes (Bounding Box Area)")
    plt.xlabel("Face Size (Bounding Box Area)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print("Mean Face Size:", df['face_size'].mean())
    print("Standard Deviation of Face Size:", df['face_size'].std())
