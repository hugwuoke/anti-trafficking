import numpy as np
import pickle
from scipy.linalg import svd, orthogonal_procrustes

with open('embeddings.pkl', 'rb') as f:
    Embeddings = pickle.load(f)

def compute_dMV(X_t, X_s):
    """
    Compute the estimated pairwise distance dMV between two embeddings X_t and X_s.

    Parameters:
    X_t: numpy.ndarray
        The embedding at time t (n x d).
    X_s: numpy.ndarray
        The embedding at time s (n x d).

    Returns:
    dMV: float
        The estimated pairwise distance between X_t and X_s.
    W: numpy.ndarray
        The optimal orthogonal matrix aligning X_s to X_t.
    """

    n, d = X_t.shape

    # Step 1: Compute the optimal orthogonal matrix W
    # Using the Orthogonal Procrustes problem with Frobenius norm
    W = orthogonal_procrustes(X_s, X_t)

    # Step 2: Compute the spectral norm of the difference
    D = X_t - X_s @ W
    # Compute the largest singular value (spectral norm)
    singular_values = svd(D, compute_uv=False)
    spectral_norm = singular_values[0]

    # Step 3: Compute the distance dMV
    dMV = (1 / np.sqrt(n)) * spectral_norm

    return dMV, W

time_points = sorted(Embeddings.keys())
m = len(time_points)
D_phi = np.zeros((m, m))

for i in range(m):
    for j in range(i, m):
        t_i = time_points[i]
        t_j = time_points[j]
        X_ti = Embeddings[t_i]
        X_tj = Embeddings[t_j]
        print(f"Time {t_i} embedding shape: {X_ti.shape}")
        print(f"Time {t_j} embedding shape: {X_tj.shape}")

        # Ensure embeddings have the same number of nodes
        if X_ti.shape != X_tj.shape:
            print(f"Time points {t_i} and {t_j} have embeddings with different shapes.")

        # Compute the distance
        try:
            dMV, _ = compute_dMV(X_ti, X_tj)
            D_phi[i, j] = dMV
            D_phi[j, i] = dMV  # Symmetric matrix
        except Exception as e:
            print(f"Error computing distance between {t_i} and {t_j}: {e}")
            D_phi[i, j] = np.nan
            D_phi[j, i] = np.nan

with open('D_phi.pkl', 'wb') as f:
    pickle.dump(D_phi, f)

# Visualizing the distance matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(D_phi, annot=True, xticklabels=time_points, yticklabels=time_points, cmap='viridis')
plt.title('Pairwise Distance Matrix D_phi')
plt.xlabel('Time Points')
plt.ylabel('Time Points')
plt.show()