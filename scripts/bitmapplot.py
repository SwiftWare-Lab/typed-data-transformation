import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import os
import zstandard as zstd


def split_bytes_into_components(byte_array, component_sizes):
    """
    Splits a byte array into multiple components based on the given sizes.

    Args:
        byte_array (bytes): Input byte array to be split.
        component_sizes (list of int): Sizes of each component.

    Returns:
        List of numpy arrays, where each corresponds to a component.
    """
    # Convert the byte array to a numpy array for easier manipulation
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)  # Total bytes per element
    num_elements = len(byte_array) // total_bytes  # Number of complete elements in the array

    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])  # Offset for the current component
        component = np.zeros((num_elements, size), dtype=np.uint8)

        # Extract data for the current component
        for j in range(num_elements):
            start_idx = j * total_bytes + offset
            end_idx = start_idx + size
            component[j, :] = byte_array[start_idx:end_idx]

        # Flatten the component if needed (to match the 1D structure expected)
        components.append(component.flatten())

    return components


def compress_with_zstd(data, level=3):
    """
    Compress data using Zstd and calculate the compression ratio.

    Args:
        data (bytes): Data to compress.
        level (int): Compression level (default: 3).

    Returns:
        Tuple: Compressed data and compression ratio.
    """
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)

    # Calculate compression ratio
    comp_ratio = len(data) / len(compressed)
    return compressed, comp_ratio


def decomposition_based_compression(byte_array, component_sizes):
    """
    Perform decomposition-based compression.

    Args:
        byte_array (bytes): Input byte array to compress.
        component_sizes (list): Sizes of the components for splitting.
    """
    # Split the byte array into components
    components = split_bytes_into_components(byte_array, component_sizes)

    # Compress each component and calculate ratios
    compressed_components = []
    total_compressed_size = 0

    for i, comp in enumerate(components):
        compressed, comp_ratio = compress_with_zstd(comp.tobytes(), level=3)
        compressed_components.append(compressed)
        total_compressed_size += len(compressed)
        print(f"Component {i + 1} compression ratio: {comp_ratio}")

    # Combine compressed components
    combined_compressed_data = b"".join(compressed_components)

    # Overall compression ratio
    overall_ratio = len(byte_array) / total_compressed_size
    print(f"Overall compression ratio: {overall_ratio}")


def run_and_collect_data():
    """
    Main function to read, process, and compress the dataset.
    """
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]

    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")

        # Drop the first column and transpose
        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T

        # Convert the transposed data to bytes
        byte_array = group.to_numpy().astype(np.float32).tobytes()

        # Define component sizes (e.g., leading, content, trailing)
        component_sizes = [1, 1, 1,1]  # Example: 1 byte leading, 2 bytes content, 1 byte trailing

        # Perform decomposition-based compression
        decomposition_based_compression(byte_array, component_sizes)


if __name__ == "__main__":
    run_and_collect_data()
