import numpy as np

def transform_with_ipca(batch_loader, ipca):
    X_pca = []

    for X_batch, _, _, _ in batch_loader:
        if X_batch.size > 0:
            X_pca_batch = ipca.transform(X_batch)
            X_pca.append(X_pca_batch)

    if X_pca:
        X_pca = np.vstack(X_pca)
    else:
        print("No batches were loaded.")
        return None

    return X_pca