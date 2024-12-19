import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------- Entropy Calculation --------------------------- #

def compute_entropy(data_window):
    """Compute the entropy of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_entropy_over_data(data, window_size=256):
    """Slide a window of size `window_size` across data and compute entropy for each window."""
    entropies = []
    data_len = len(data)
    for start_idx in range(0, data_len - window_size + 1, window_size):
        window = data[start_idx:start_idx + window_size]
        ent = compute_entropy(window)
        entropies.append(ent)
    return entropies


# --------------------------- Data Splitting --------------------------- #

def split_bytes_into_components(byte_array, component_sizes):
    """
    Split the byte array into individual components based on specified sizes.

    :param byte_array: The original byte array.
    :param component_sizes: List indicating the size of each component.
    :return: List of numpy arrays, each representing a component.
    """
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    num_elements = len(byte_array) // total_bytes

    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        component = byte_array[offset::total_bytes]
        components.append(component)
    return components


# --------------------------- Correlation Analysis --------------------------- #

def plot_correlation_matrix(components_entropy, dataset_name, save_path):
    """
    Plot the correlation matrix of component entropies.

    :param components_entropy: List of entropy lists for each component.
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
    """
    df = pd.DataFrame(components_entropy).T
    df.columns = [f'Component {i + 1}' for i in range(len(components_entropy))]
    correlation_matrix = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix of Component Entropies for {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation matrix plot saved as {save_path}")


def plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size=256, save_fig=True):
    """Plot entropy profiles for full data and each component."""
    num_components = len(components_entropy)

    fig, axes = plt.subplots(num_components + 1, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'Entropy Profiles (Window Size = {window_size}) for {dataset_name}')

    # Plot full data entropy
    axes[0].plot(full_data_entropy, label='Full Data Entropy', color='blue')
    axes[0].set_ylabel('Entropy (bits)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Plot each component entropy
    for i, comp_ent in enumerate(components_entropy):
        axes[i + 1].plot(comp_ent, label=f'Component {i + 1} Entropy', color='red')
        axes[i + 1].set_ylabel('Entropy (bits)')
        axes[i + 1].legend(loc='upper right')
        axes[i + 1].grid(True)

    axes[-1].set_xlabel('Window Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        fig_filename = f'/home/jamalids/Documents/{dataset_name}_entropy_profiles.png'
        plt.savefig(fig_filename)
        print(f"Entropy profiles plot saved as {fig_filename}")
    else:
        plt.show()


def plot_entropy_distances(dataset_name, components_entropy, window_size=256, save_path=None):
    """
    Plot the distances between entropy profiles of each component.

    :param dataset_name: Name of the dataset for titling.
    :param components_entropy: List of entropy lists for each component.
    :param window_size: Size of the sliding window used for entropy calculation.
    :param save_path: File path to save the plot. If None, the plot is shown.
    """
    num_components = len(components_entropy)
    if num_components < 2:
        print("Not enough components to calculate distances.")
        return

    # Calculate pairwise distances between components
    distance_dict = {}
    for i in range(num_components):
        for j in range(i + 1, num_components):
            # Using absolute difference (Manhattan distance) per window
            distance = np.abs(np.array(components_entropy[i]) - np.array(components_entropy[j]))
           # distance = (np.array(components_entropy[i]) - np.array(components_entropy[j]))
            distance_dict[f'Component {i + 1} vs Component {j + 1}'] = distance

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title(f'Entropy Distance Profiles (Window Size = {window_size}) for {dataset_name}')

    for label, distance in distance_dict.items():
        plt.plot(distance, label=label)

    plt.xlabel('Window Index')
    plt.ylabel('Entropy Distance (bits)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()


    plt.savefig(f'/home/jamalids/Documents/{dataset_name}_entropy_distance.png')
    print(f"Entropy distance plot saved as {save_path}")

    plt.show()


# --------------------------- Groupings Definition --------------------------- #

def get_all_groupings(n=4):
    """
    Generate all possible groupings (set partitions) for n components.

    :param n: Number of components.
    :return: List of groupings, where each grouping is a list of lists.
    """
    if n != 4:
        raise NotImplementedError("This function currently supports only n=4 components.")

    groupings = [
        [[0, 1, 2, 3]],
        [[0, 1, 2], [3]],
        [[0, 1, 3], [2]],
        [[0, 2, 3], [1]],
        [[1, 2, 3], [0]],
        [[0, 1], [2, 3]],
        [[0, 2], [1, 3]],
        [[0, 3], [1, 2]],
        [[0, 1], [2], [3]],
        [[0, 2], [1], [3]],
        [[0, 3], [1], [2]],
        [[1, 2], [0], [3]],
        [[1, 3], [0], [2]],
        [[2, 3], [0], [1]],
        [[0], [1], [2], [3]]
    ]
    return groupings


# --------------------------- Main Function --------------------------- #

def run_analysis():
    # --------------------------- Configuration --------------------------- #
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/High-Entropy/32/turbulence_f32.tsv"
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    save_dir = f"/home/jamalids/Documents/{dataset_name}_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    window_size = 65536
    correlation_threshold = 0.8  # Threshold for considering components as highly correlated (optional)

    # --------------------------- Data Loading --------------------------- #
    print(f"Loading dataset from {dataset_path}...")
    ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
    print("Dataset loaded successfully.\n")

    # --------------------------- Data Preparation --------------------------- #
    print("Preparing data...")
    group = ts_data.drop(ts_data.columns[0], axis=1).astype(float).T
    byte_array = group.to_numpy().astype(np.float32).tobytes()
    component_sizes = [1, 1, 1, 1]  # Assuming four components, each of size 1 byte
    components = split_bytes_into_components(byte_array, component_sizes)
    print("Data preparation completed.\n")

    # --------------------------- Entropy Calculation --------------------------- #
    print("Calculating entropy profiles...")
    full_data_entropy = calculate_entropy_over_data(byte_array, window_size=window_size)
    print("Full data entropy calculation completed.\n")

    # Calculate entropy for each component
    components_entropy = []
    for i, comp in enumerate(components):
        comp_data = comp.tobytes()
        comp_entropy = calculate_entropy_over_data(comp_data, window_size=window_size)
        components_entropy.append(comp_entropy)
        print(f"Component {i + 1} entropy calculation completed.\n")

    # --------------------------- Correlation Analysis --------------------------- #
    print("Analyzing correlations between component entropies...")
    correlation_plot_path = os.path.join(save_dir, f"{dataset_name}_correlation_matrix.png")
    plot_correlation_matrix(components_entropy, dataset_name, correlation_plot_path)

    # --------------------------- Visualization --------------------------- #
    # Plot entropy profiles
    plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size=window_size, save_fig=True)

    # Plot entropy distances
    entropy_distance_plot_path = os.path.join(save_dir, f"{dataset_name}_entropy_distances.png")
    plot_entropy_distances(dataset_name, components_entropy, window_size=window_size,
                           save_path=entropy_distance_plot_path)

    print("\nAnalysis completed successfully.")


# --------------------------- Execute Script --------------------------- #

if __name__ == "__main__":
    run_analysis()
