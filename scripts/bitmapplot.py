
import numpy as np
import pandas as pd
import os
import zstandard as zstd


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


def run_and_collect_data():

    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]

    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")


        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T


        byte_array = group.to_numpy().astype(np.float32).tobytes()


        component_sizes = [2, 1,1]


        decomposition_based_compression(byte_array, component_sizes)


if __name__ == "__main__":
    run_and_collect_data()
