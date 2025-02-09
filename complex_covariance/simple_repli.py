import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_data_into_timesteps(data: pd.DataFrame) -> dict:
    # Split the data into time steps
    time_steps = data["t"].unique()
    data_by_time = {t: data[data["t"] == t].drop(columns=["t"]).to_numpy() for t in time_steps}

    return data_by_time

def build_A_matrix(data: np.ndarray) -> np.ndarray:
    # get total number occurances
    n = data.shape[0]

    A = data.T @ data

    return A / n

def get_graph_covariance(A: np.ndarray) -> np.ndarray:
    # Compute the graph covariance matrix
    # get main diagonal of A
    n = np.diag(A) 
    graph_covariance = A - np.outer(n, n)

    return graph_covariance


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv("simple_gen_data.csv")
    data = split_data_into_timesteps(data)

    # Build the adjacency matrix for each time step
    adjacency_matrices = {t: build_A_matrix(data[t]) for t in data}

    # Compute the graph covariance matrix for each time step
    graph_covariances = {t: get_graph_covariance(adjacency_matrices[t]) for t in adjacency_matrices}
    # for t in graph_covariances:
    #     # only print the first 4 digits after the decimal point
    #     np.set_printoptions(precision=4)
    #     print(f"Time step {t}:")
    #     print(graph_covariances[t])
    #     print()
    # plot the first row of the graph covariance matrix for each time step in one plot

    graph_covariances_over_time = np.array([graph_covariances[t][0, :] for t in sorted(list(graph_covariances.keys()))])
    # print(graph_covariances_over_time)
    fig, ax = plt.subplots()
    for feature in graph_covariances_over_time.T:
        ax.plot(feature, label=f"Feature {len(ax.lines)}", alpha=0.5)

    ax.legend()
    plt.show()
