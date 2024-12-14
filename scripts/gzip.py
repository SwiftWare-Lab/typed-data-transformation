import numpy as np
import pandas as pd
import os
import gzip


def split_bytes_into_components(byte_array, component_sizes):
    """
    Splits a byte array into multiple components based on the given sizes.

    Args:
        byte_array (bytes): Input byte array to be split.
        component_sizes (list of int): Sizes of each component.

    Returns:
        List of numpy arrays, where each corresponds to a component.
    """
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    num_elements = len(byte_array) // total_bytes

    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        component = np.zeros((num_elements, size), dtype=np.uint8)

        for j in range(num_elements):
            start_idx = j * total_bytes + offset
            end_idx = start_idx + size
            component[j, :] = byte_array[start_idx:end_idx]

        components.append(component.flatten())

    return components


def compress_with_gzip(data, output_path):
    """
    Compress data using Gzip and calculate the compression ratio.

    Args:
        data (bytes): Data to compress.
        output_path (str): Path to save the compressed file.

    Returns:
        float: Compression ratio.
    """
    original_size = len(data)

    # Compress data using Gzip
    with gzip.open(output_path, 'wb') as f:
        f.write(data)

    # Get the size of the compressed file
    compressed_size = os.path.getsize(output_path)

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size
    return compression_ratio, compressed_size


def decomposition_based_compression(byte_array, component_sizes, output_dir):
    """
    Perform decomposition-based compression and save results with Gzip.

    Args:
        byte_array (bytes): Input byte array to compress.
        component_sizes (list): Sizes of the components for splitting.
        output_dir (str): Directory to save compressed components.
    """
    components = split_bytes_into_components(byte_array, component_sizes)

    total_original_size = 0
    total_compressed_size = 0

    for i, comp in enumerate(components):
        original_size = comp.nbytes
        total_original_size += original_size

        output_path = os.path.join(output_dir, f"component_{i + 1}.gz")
        comp_ratio, compressed_size = compress_with_gzip(comp.tobytes(), output_path)
        total_compressed_size += compressed_size

        print(f"Component {i + 1}: Original size = {original_size} bytes, Compressed size = {compressed_size} bytes, Compression ratio = {comp_ratio:.2f}")

    overall_ratio = total_original_size / total_compressed_size
    print(f"Overall compression ratio: {overall_ratio:.2f}")


def run_and_collect_data():
    """
    Main function to read, process, and compress the dataset using Gzip.
    """
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]
    output_dir = "compressed_gzip"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")

        # Drop the first column and transpose
        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T

        # Convert the transposed data to bytes

         = group.to_numpy().astype(np.float32).tobytes()
        comp_ratio, compressed_size = compress_with_gzip(comp.tobytes(), output_path)

        # Define component sizes (e.g., leading, content, trailing)
        component_sizes = [1, 1, 1, 1]  # Example: 1 byte leading, 1 byte content, etc.

        # Perform decomposition-based compression
        decomposition_based_compression(byte_array, component_sizes, output_dir)

    print(f"Compressed data saved to {output_dir}")


if __name__ == "__main__":
    run_and_collect_data()
