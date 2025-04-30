import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_image_dimensions(folder_path, df_labels):
    """
    Plot a scatter plot of image dimensions (width vs height) for images in the specified folder.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
        df_labels (pd.DataFrame): The DataFrame containing the 'filename' column.
    
    Returns:
        None: Displays a scatter plot of image dimensions.
    """
    widths = []
    heights = []

    for file_name in tqdm(df_labels['filename'], desc="Gathering image dimensions"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except Exception as e:
            print(f"Could not open {file_name}: {e}")

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5, edgecolor='k')
    plt.title("Scatter Plot of Image Dimensions (Width vs Height)")
    plt.xlabel("Image Width")
    plt.ylabel("Image Height")
    plt.grid(True)
    plt.show()