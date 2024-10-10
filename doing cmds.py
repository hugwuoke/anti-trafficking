import numpy as np
import pickle

with open('D_phi.pkl', 'rb') as f:
    D_phi = pickle.load(f)

with open('embeddings.pkl', 'rb') as f:
    Embeddings = pickle.load(f)

def classical_mds(D, k=1):
    """
    Perform Classical Multidimensional Scaling (MDS).
    
    Parameters:
    D : numpy.ndarray
        The symmetric distance matrix (n x n).
    k : int
        The number of dimensions for the embedding.
    
    Returns:
    X : numpy.ndarray
        The coordinates of the points in k-dimensional space (n x k).
    """
    # Number of items
    n = D.shape[0]
    
    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Compute the squared distances
    D_squared = D ** 2
    
    # Double center the squared distance matrix
    B = -0.5 * H @ D_squared @ H
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top k positive eigenvalues
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    num_positive = len(positive_eigenvalues)
    
    if num_positive < k:
        print(f"Only {num_positive} positive eigenvalues available. Adjusting k to {num_positive}.")
        k = num_positive
    
    L_k = np.diag(np.sqrt(eigenvalues[:k]))
    V_k = eigenvectors[:, :k]
    
    # Compute the coordinates
    X = V_k @ L_k
    
    return X

# Assuming D_phi is your distance matrix
X_mds = classical_mds(D_phi, k=1)

import matplotlib.pyplot as plt
time_points = sorted(Embeddings.keys())

plt.figure(figsize=(10, 6))

# Plot the first dimension of the MDS embeddings against time
plt.scatter(time_points, X_mds, color='blue', zorder=2)
plt.plot(time_points, X_mds, color='red', linestyle='-', linewidth=2, zorder=1)

plt.title('Mirror of CTDC Data')
plt.xlabel('Time')
plt.ylabel('CMDS Dimension 1')
plt.grid(True)
plt.show()

