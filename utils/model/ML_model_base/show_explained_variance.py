import numpy as np
import matplotlib.pyplot as plt

def show_explained_variance(ipca):
    """
    Show a plot of the cumulative explained variance for each principal component.
    
    Args:
    - ipca: Fitted IPCA model.
    
    Plots:
    - A cumulative explained variance plot for each principal component.
    """
    explained_variance = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linestyle='-')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    total_explained_variance = np.sum(explained_variance)
    print(f"Total Explained Variance: {total_explained_variance:.4f}")