import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_data_into_timesteps(data: pd.DataFrame) -> dict:
    # Split the data into time steps
    time_steps = data["t"].unique()
    data_by_time = {t: data[data["t"] == t].drop(columns=["t"]).to_numpy() for t in time_steps}

    return data_by_time

def get_graph_covariance(data: np.ndarray) -> np.ndarray:
    # Compute the graph covariance matrix
    n = data.shape[0]
    mean = np.mean(data, axis=0).reshape(1, -1)
    data_centered = data - mean
    # normalize the data
    data_centered = data_centered / data_centered.std(axis=0).reshape(1, -1)
    graph_covariance = (data_centered.T @ data_centered) / (n-1)

    return graph_covariance

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv("simple_gen_data.csv")
    data = split_data_into_timesteps(data)

    # Build the adjacency matrix for each time step
    # adjacency_matrices = {t: build_A_matrix(data[t]) for t in data}

    # Compute the graph covariance matrix for each time step
    graph_covariances = {t: get_graph_covariance(data[t]) for t in data}
    # for t in graph_covariances:
    #     # only print the first 4 digits after the decimal point
    #     np.set_printoptions(precision=4)
    #     print(f"Time step {t}:")
    #     print(graph_covariances[t])
    #     print()
    # plot the first row of the graph covariance matrix for each time step in one plot

    graph_covariances_over_time = np.array([graph_covariances[t][0, :] for t in sorted(list(graph_covariances.keys()))])[:, 1:]

    # Custom legend labels for each feature
    legend_labels = [
        "(1,2) Independent", 
        "(1,3) Dependent", 
        "(1,4) Increasing", 
        "(1,5) Decreasing", 
        "(1,6) Shifting", 
        "(1,7) Spike"
    ]

    # Use a built-in matplotlib style that doesn't require seaborn
    plt.style.use('ggplot')

    # Create a figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Define increased font sizes
    title_fontsize = 20
    label_fontsize = 18
    legend_fontsize = 18
    tick_fontsize = 18

    # Plot the first 3 features in the left subplot
    for i in range(3):
        ax1.plot(graph_covariances_over_time[:, i],
                label=legend_labels[i],
                linewidth=2,
                alpha=0.85)
    ax1.set_title("Features 1-3", fontsize=title_fontsize)
    ax1.set_xlabel("Time", fontsize=label_fontsize)
    ax1.set_ylabel("Covariance", fontsize=label_fontsize)
    ax1.legend(loc="best", fontsize=legend_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True)

    # Plot the last 3 features in the right subplot
    for i in range(3, 6):
        ax2.plot(graph_covariances_over_time[:, i],
                label=legend_labels[i],
                linewidth=2,
                alpha=0.85)
    ax2.set_title("Features 4-6", fontsize=title_fontsize)
    ax2.set_xlabel("Time", fontsize=label_fontsize)
    ax2.set_ylabel("Covariance", fontsize=label_fontsize)
    ax2.legend(loc="best", fontsize=legend_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()