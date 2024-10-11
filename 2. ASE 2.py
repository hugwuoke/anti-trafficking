import pickle
import numpy as np
from scipy.linalg import eigh  # For symmetric matrices


with open('adjacency_matrices.pkl', 'rb') as f:
    Adjacency_matrices = pickle.load(f)

def compute_ase(A, d):
    """
    Compute the adjacency spectral embedding (ASE) of an adjacency matrix A.

    Parameters:
    A: numpy.ndarray
        The adjacency matrix (n x n).
    d: int
        The embedding dimension.

    Returns:
    X_hat: numpy.ndarray
        The ASE embedding matrix (n x d).
    """
    # Ensure A is a NumPy array
    A = np.array(A)

    # Compute the eigenvalues and eigenvectors
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = eigh(A)

    # Select the top d largest eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]  # Indices for sorting eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Handle negative eigenvalues
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    num_positive = len(positive_eigenvalues)
    
    if num_positive == 0:
        raise ValueError("No positive eigenvalues found. Cannot compute ASE.")
    
    if num_positive < d:
        print(f"Only {num_positive} positive eigenvalues available. Adjusting dimension to {num_positive}.")
        d = num_positive

    eigenvalues_d = positive_eigenvalues[:d]
    eigenvectors_d = eigenvectors[:, :d]

    # Compute the embedding
    sqrt_eigenvalues = np.sqrt(eigenvalues_d)
    X_hat = eigenvectors_d * sqrt_eigenvalues

    return X_hat

# Assume Adjacency_matrices is your dictionary of adjacency matrices
d = 2  # Desired embedding dimension
Embeddings = {}

for year, adjacency_matrix in Adjacency_matrices.items():
    try:
        X_hat = compute_ase(adjacency_matrix, d)
        Embeddings[year] = X_hat
        print(f"Year {year}: Computed ASE with embedding shape {X_hat.shape}.")
    except ValueError as e:
        print(f"Year {year}: Error computing ASE: {e}")

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(Embeddings, f)