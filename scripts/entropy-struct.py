import os
import sys
import math
import numpy as np
import pandas as pd
import zstandard as zstd
from collections import Counter
import matplotlib.pyplot as plt

def compute_entropy(data_window):
    """Compute the entropy of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
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

def split_bytes_into_components(byte_array, component_sizes):
    # Convert the byte array to a numpy array for easier manipulation
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)  # Total bytes per element
    num_elements = len(byte_array) // total_bytes  # Number of complete elements in the array

    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])  # Offset for the current component
        component = np.zeros((num_elements, size), dtype=np.uint8)

        for j in range(num_elements):
            start_idx = j * total_bytes + offset
            end_idx = start_idx + size
            component[j, :] = byte_array[start_idx:end_idx]

        components.append(component.flatten())

    return components

def compress_with_zstd(data, level=3):
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    comp_ratio = len(data) / len(compressed)
    return compressed, comp_ratio

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)

    # Compress each component and calculate ratios
    compressed_components = []
    total_compressed_size = 0

    for i, comp in enumerate(components):
        compressed, comp_ratio = compress_with_zstd(comp.tobytes(), level=3)
        compressed_components.append(compressed)
        total_compressed_size += len(compressed)
        print(f"Component {i + 1} compression ratio: {comp_ratio}")

    combined_compressed_data = b"".join(compressed_components)
    # Overall compression ratio
    overall_ratio = len(byte_array) / total_compressed_size
    print(f"Overall compression ratio: {overall_ratio}")

    return components, combined_compressed_data

def plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size=256, save_fig=False):
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
        axes[i+1].plot(comp_ent, label=f'Component {i+1} Entropy', color='red')
        axes[i+1].set_ylabel('Entropy (bits)')
        axes[i+1].legend(loc='upper right')
        axes[i+1].grid(True)

    axes[-1].set_xlabel('Window Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        fig_filename = f'{dataset_name}_entropy_profiles.png'
        plt.savefig(fig_filename)
        print(f"Entropy plot saved as {fig_filename}")
    else:
        plt.show()

def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/tpch_lineitem_f32.tsv"
    datasets = [dataset_path]

    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")

        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T
        byte_array = group.to_numpy().astype(np.float32).tobytes()

        component_sizes = [1,1, 1,1]
        components, combined_compressed_data = decomposition_based_compression(byte_array, component_sizes)

        # Entropy analysis
        window_size = 65536

        # Calculate entropy for the full data
        full_data_entropy = calculate_entropy_over_data(byte_array, window_size=window_size)
        print(f"Full data entropy values (window size = {window_size}): {full_data_entropy}")

        # Calculate entropy for each component
        components_entropy = []
        for i, comp in enumerate(components):
            comp_data = comp.tobytes()
            comp_entropy = calculate_entropy_over_data(comp_data, window_size=window_size)
            components_entropy.append(comp_entropy)
            print(f"Component {i + 1} entropy values (window size = {window_size}): {comp_entropy}")

        # Plot the entropy profiles
        plot_entropy_profiles(dataset_name, full_data_entropy, components_entropy, window_size=window_size, save_fig=False)

if __name__ == "__main__":
    run_and_collect_data()
