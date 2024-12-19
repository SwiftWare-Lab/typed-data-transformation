import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- Entropy Calculation --------------------------- #
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np  # Ensure numpy is imported

def plot_entropy_distances3(dataset_name, components_entropy, window_size, save_path=None):
    """
    Plot the distances between entropy profiles of each component in separate subplots,
    avoiding redundancy by only plotting distances for component i with components j where j > i.
    """
    num_components = len(components_entropy)
    if num_components < 2:
        print("Not enough components to calculate distances.")
        return

    # Set up the subplot grid
    fig, axes = plt.subplots(num_components, 1, figsize=(10, 5 * num_components))
    if num_components == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    # Plot distances only once
    for i in range(num_components):
        ax = axes[i]
        for j in range(i + 1, num_components):  # Start from i+1 to avoid redundancy
            distance = np.abs(np.array(components_entropy[i]) - np.array(components_entropy[j]))
            ax.plot(distance, label=f'Component {i + 1} vs Component {j + 1}')

        ax.set_title(f'Component {i + 1} Distance Profiles', fontsize=14)
        ax.set_xlabel('Window Index', fontsize=12)
        ax.set_ylabel('Entropy Distance (bits)', fontsize=12)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Distance plots saved to {save_path}")
    else:
        plt.show()

def plot_entropy_distances(dataset_name, components_entropy, window_size, save_path=None):
    """
    Plot the distances between entropy profiles of each component in separate subplots.
    Each subplot will display the distances of one component with all others.
    """
    num_components = len(components_entropy)
    if num_components < 2:
        print("Not enough components to calculate distances.")
        return

    # Set up the subplot grid
    fig, axes = plt.subplots(num_components, 1, figsize=(30, 10 * num_components))
    if num_components == 1:
        axes = [axes]  # Make sure axes is iterable for a single subplot scenario

    # Plot each component's distance to others
    for i in range(num_components):
        ax = axes[i]
        for j in range(num_components):
            if i != j:
                distance = np.abs(np.array(components_entropy[i]) - np.array(components_entropy[j]))
                ax.plot(distance, label=f'Component {i + 1} vs Component {j + 1}')

        ax.set_title(f'Component {i + 1} Distance Profiles', fontsize=14)
        ax.set_xlabel('Window Index', fontsize=12)
        ax.set_ylabel('Entropy Distance (bits)', fontsize=12)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Distance plots saved to {save_path}")
    else:
        plt.show()
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_entropy_distances1(dataset_name, components_entropy, window_size, save_path=None):
    """
    Plot the distances between entropy profiles of each component with enhanced y-axis details and distinctly different colors.
    If save_path is provided, the plot will be saved to that location.
    Otherwise, the plot will be displayed.
    """
    num_components = len(components_entropy)
    if num_components < 2:
        print("Not enough components to calculate distances.")
        return

    # Define a custom color palette with distinct colors
    color_palette = [
        'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
    ]

    # Calculate pairwise distances between components
    distance_dict = {}
    color_idx = 0  # Start indexing from 0 for the color palette
    for i in range(num_components):
        for j in range(i + 1, num_components):
            if color_idx >= len(color_palette):  # Reset the color index if it exceeds the palette length
                color_idx = 0
            distance = np.abs(np.array(components_entropy[i]) - np.array(components_entropy[j]))
            distance_dict[f'Component {i + 1} vs Component {j + 1}'] = (distance, color_palette[color_idx])
            color_idx += 1  # Move to the next color for the next line

    # Plotting configurations
    plt.figure(figsize=(20, 20), dpi=300)  # Increased dpi for higher resolution
    plt.title(f'Entropy Distance Profiles (Window Size = {window_size}) for {dataset_name}', fontsize=20)

    for label, (distance, color) in distance_dict.items():
        plt.plot(distance, label=label, color=color, marker='o', linestyle='-', markersize=5)

    plt.xlabel('Window Index', fontsize=16)
    plt.ylabel('Entropy Distance (bits)', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Entropy distance plot saved as {save_path}")
    else:
        plt.show()

# Example use of the function would go here

# Example use of the function would go here

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

def plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size, save_fig=True):
    """Plot entropy profiles for full data and each component."""
    num_components = len(components_entropy)

    fig, axes = plt.subplots(num_components + 1, 1, figsize=(30, 30), sharex=True)
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


# --------------------------- Main Function --------------------------- #

def run_analysis():
    # Configuration
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/High-Entropy/64/tpch_order_f64.tsv"
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    save_dir = f"/home/jamalids/Documents/{dataset_name}_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    window_size = 65536  # Increase if needed to cover more bytes

    # Data Loading
    print(f"Loading dataset from {dataset_path}...")
    ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
    print("Dataset loaded successfully.\n")

    # Data Preparation
    print("Preparing data...")
    group = ts_data.drop(ts_data.columns[0], axis=1).astype(float).T
    #group=group.iloc[:100,:]
    byte_array = group.to_numpy().astype(np.float64).tobytes()
    component_sizes = [8] * 8  # 8 components, each 8 bytes
    components = split_bytes_into_components(byte_array, component_sizes)
    print("Data preparation completed.\n")

    # Entropy Calculation
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

    # Correlation Analysis
    print("Analyzing correlations between component entropies...")
    correlation_plot_path = os.path.join(save_dir, f"{dataset_name}_correlation_matrix.png")
    plot_correlation_matrix(components_entropy, dataset_name, correlation_plot_path)

    # Distance Analysis
    print("Analyzing distances between component entropies...")
    entropy_distance_plot_path = os.path.join(save_dir, f"{dataset_name}_entropy_distances.png")
    plot_entropy_distances(dataset_name, components_entropy, window_size=window_size,
                           save_path=entropy_distance_plot_path)



    # Visualization
    # Plot entropy profiles
    plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size=window_size, save_fig=True)

    print("\nAnalysis completed successfully.")

# Execute Script
if __name__ == "__main__":
    run_analysis()
