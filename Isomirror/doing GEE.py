import numpy as np
import pickle

def adjacency_to_edge_list(adjacency_matrix):
    """
    Convert an adjacency matrix to an edge list with weights.

    Parameters:
    adjacency_matrix: numpy.ndarray
        A square adjacency matrix of shape (n, n).

    Returns:
    numpy.ndarray
        Edge list of shape (s, 3), where each row represents [node_a, node_b, weight].
    """
    edge_list = []
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                edge_list.append([i, j, adjacency_matrix[i, j]])

    return np.array(edge_list)

def graph_encoder_embedding(E, Y, n, K, Lap=True):
    """
    Function to compute graph encoder embedding (GEE).
    
    Parameters:
    E: numpy.ndarray
        Edge list of shape (s, 3), where each row contains [node_a, node_b, weight]
    Y: numpy.ndarray
        Labels of nodes of shape (n,)
    n: int
        Number of nodes
    K: int
        Number of classes
    Lap: bool
        Whether to use Laplacian-based normalization (default: True)
    
    Returns:
    Z: numpy.ndarray
        Encoder embedding of shape (n, K)
    W: numpy.ndarray
        Projection matrix of shape (n, K)
    """
    # Initialize the projection matrix
    W = np.zeros((n, K))

    # If Laplacian-based normalization is enabled
    if Lap:
        D = np.zeros((n,))
        
        # Compute degree vector D based on edge list
        for i in range(len(E)):
            D[int(E[i, 0])] += 1
            D[int(E[i, 1])] += 1

        # Entry-wise square root of the degree vector
        D = np.sqrt(D)

        # Update weights in edge list based on degree
        for i in range(len(E)):
            E[i, 2] = E[i, 2] / (D[int(E[i, 0])] * D[int(E[i, 1])])

    # Compute the projection matrix W
    for k in range(K):
        ind = np.where(Y == k)[0]
        nk = len(ind)
        if nk > 0:
            W[ind, k] = 1.0 / nk

    # Initialize embedding Z
    Z = np.zeros((n, K))

    # Calculate encoder embedding Z using W and edge list E
    for i in range(len(E)):
        a = int(E[i, 0])
        b = int(E[i, 1])
        e = E[i, 2]
        c = int(Y[a])
        d = int(Y[b])
        
        Z[a, d] += W[b, d] * e
        Z[b, c] += W[a, c] * e

    return Z, W
with open('adjacency_matrices.pkl', 'rb') as f:
    Adjacency_matrices = pickle.load(f)

Embeddings_G = {}
for year, matrix in Adjacency_matrices.items():
    edge_list = adjacency_to_edge_list(matrix)
    n = matrix.shape[0]
    K=2
    node_labels = np.random.randint(0, K, size=n)
    # Compute graph encoder embedding
    Z, W = graph_encoder_embedding(edge_list, node_labels, n, K, Lap=True)
    Embeddings_G[year] = Z
    print(year)

with open('embeddings_g.pkl', 'wb') as f:
    pickle.dump(Embeddings_G, f)