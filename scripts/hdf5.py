import numpy as np
import pandas as pd
import os
import h5py


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


def save_with_gzip(comp, hdf5_file, dataset_name, component_index):
    """
    Save compressed data in HDF5 format using Gzip compression and show the compression ratio.

    Args:
        comp (numpy.ndarray): Data to save.
        hdf5_file (h5py.File): HDF5 file handle to store data.
        dataset_name (str): Base name of the dataset.
        component_index (int): Index of the component.

    Returns:
        float: Compression ratio for the component.
    """
    # Original size in bytes
    original_size = comp.nbytes

    # Save the component with Gzip compression
    dataset = hdf5_file.create_dataset(
        f"{dataset_name}_component_{component_index}",
        data=comp,
        compression="gzip",  # Use Gzip compression
        compression_opts=9   # Compression level
    )

    # Compressed size
    compressed_size = dataset.id.get_storage_size()

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size
    print(f"Component {component_index} compression ratio: {compression_ratio:.2f}")

    return compression_ratio


def decomposition_based_compression(byte_array, component_sizes, hdf5_file, dataset_name):
    """
    Perform decomposition-based compression and save results in HDF5.

    Args:
        byte_array (bytes): Input byte array to compress.
        component_sizes (list): Sizes of the components for splitting.
        hdf5_file (h5py.File): HDF5 file handle to store results.
        dataset_name (str): Name of the dataset for storing compressed data.
    """
    components = split_bytes_into_components(byte_array, component_sizes)

    total_original_size = 0
    total_compressed_size = 0

    for i, comp in enumerate(components):
        total_original_size += comp.nbytes
        compression_ratio = save_with_gzip(comp, hdf5_file, dataset_name, i + 1)
        compressed_size = hdf5_file[f"{dataset_name}_component_{i + 1}"].id.get_storage_size()
        total_compressed_size += compressed_size

    # Overall compression ratio
    overall_ratio = total_original_size / total_compressed_size
    print(f"Overall compression ratio: {overall_ratio:.2f}")


def run_and_collect_data():
    """
    Main function to read, process, and compress the dataset using HDF5 and Gzip.
    """
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]
    hdf5_output_path = "compressed_data_gzip.h5"

    with h5py.File(hdf5_output_path, "w") as hdf5_file:
        for dataset_path in datasets:
            ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
            dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
            print(f"Processing dataset: {dataset_name}")

            # Drop the first column and transpose
            group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T

            # Convert the transposed data to bytes
            byte_array = group.to_numpy().astype(np.float32).tobytes()

            # Define component sizes (e.g., leading, content, trailing)
            component_sizes = [1, 1, 1, 1]  # Example: 1 byte leading, 1 byte content, etc.

            # Perform decomposition-based compression
            decomposition_based_compression(byte_array, component_sizes, hdf5_file, dataset_name)

    print(f"Compressed data saved to {hdf5_output_path}")


if __name__ == "__main__":
    run_and_collect_data()
