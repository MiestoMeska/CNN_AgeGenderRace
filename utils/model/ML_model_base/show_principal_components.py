import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

def show_principal_components(pca, num_components=5):
    """
    Show the first few principal components.
    
    Args:
    - ipca: Fitted IPCA model.
    - n_components: Number of principal components to show.
    
    Prints:
    - The first `n_components` principal components.
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
