from sklearn.decomposition import IncrementalPCA

def fit_ipca_with_batches(batch_loader, n_components=100, batch_size=100):
    """
    Fit Incremental PCA on large dataset using batch loading.
    
    Args:
    - batch_loader: A generator that yields batches of data.
    - n_components: Number of components to retain.
    - batch_size: Batch size for fitting IPCA.
    
    Returns:
    - Fitted IncrementalPCA model.
    """
    ipca = IncrementalPCA(n_components=n_components)
    
    for X_batch, _, _, _ in batch_loader:
        ipca.partial_fit(X_batch)
    
    return ipca
