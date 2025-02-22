import matplotlib.pyplot as plt

def show_scree_plot(ipca):
    """
    Show a scree plot of the explained variance.
    
    Args:
    - ipca: Fitted IPCA model.
    
    Plots:
    - A scree plot of the explained variance for each component without markers.
    """
    explained_variance = ipca.explained_variance_ratio_
    n_components = len(explained_variance)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), explained_variance, linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()