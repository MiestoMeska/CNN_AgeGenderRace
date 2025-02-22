import os
import matplotlib.pyplot as plt
from PIL import Image
import math

def display_images_for_outliers(outliers_df, folder_path, filename_column='filename', images_per_row=8):
    """
    Display images for the rows identified as outliers in a grid layout with 8 images per row.
    
    Args:
        outliers_df (pd.DataFrame): DataFrame containing outlier rows.
        folder_path (str): The folder where the images are stored.
        filename_column (str): The column containing the filenames.
        images_per_row (int): The number of images to display per row in the grid layout.
    
    Returns:
        None
    """
    num_images = len(outliers_df)
    
    if num_images == 0:
        print("No outliers found.")
        return

    num_rows = math.ceil(num_images / images_per_row)

    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, (index, row) in enumerate(outliers_df.iterrows()):
        image_path = os.path.join(folder_path, row[filename_column])
        
        try:
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Outlier {i+1}")
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading image {row[filename_column]}: {e}")
            axes[i].axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()