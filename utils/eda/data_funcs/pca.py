import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import random

def load_images_as_flattened_array(folder_path, df_labels, num_images=None):
    """
    Load random images from the folder and flatten them into a 1D array for PCA.
    
    Args:
        folder_path (str): Path to the folder containing images.
        df_labels (pd.DataFrame): DataFrame containing the 'filename' column.
        num_images (int): Number of random images to load. If None, all images will be loaded.
        
    Returns:
        np.ndarray: Array of flattened images.
    """
    images = []
    
    if num_images:
        sampled_filenames = df_labels['filename'].sample(n=num_images, random_state=42).tolist()
    else:
        sampled_filenames = df_labels['filename'].tolist()
    
    for file_name in sampled_filenames:
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path).convert('L')
        img_array = np.asarray(img).flatten()
        images.append(img_array)
    
    return np.array(images)

def plot_principal_components_as_images(pca, num_components=5):
    """
    Plot the principal components as images, arranging them in rows of 5 images max.
    
    Args:
        pca (PCA): PCA model.
        num_components (int): Number of principal components to plot.
    """
    images_per_row = 5
    num_rows = (num_components + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
    axes = axes.flatten()
    
    for i in range(num_components):
        component = pca.components_[i]
        img_shape = int(np.sqrt(component.shape[0]))
        axes[i].imshow(component.reshape(img_shape, img_shape), cmap='gray')
        axes[i].set_title(f"PC {i+1}")
        axes[i].axis('off')
    
    for j in range(num_components, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()