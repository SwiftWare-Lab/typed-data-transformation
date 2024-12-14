import os

import numpy as np
import pandas as pd
import zlib
from numpy.lib.stride_tricks import as_strided

def split_bytes_into_components(byte_array, component_sizes):
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    num_elements = len(byte_array) // total_bytes
    components = []
    offset = 0
    for size in component_sizes:
        component_view = as_strided(byte_array[offset:], shape=(num_elements, size), strides=(total_bytes, 1))
        components.append(component_view.flatten())
        offset += size
    return components

def lz77_compress(data):
    compressed = zlib.compress(data)
    compression_ratio = len(data) / len(compressed)
    return compressed, compression_ratio

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes = []
    total_original_size = len(byte_array)

    for i, comp in enumerate(components):
        compressed_comp, lz77_ratio = lz77_compress(comp)
        compressed_sizes.append(len(compressed_comp))
        print(f"Component {i + 1} zlib compression ratio: {lz77_ratio:.2f}")
        print(f"Component {i + 1} compressed size: {len(compressed_comp)} bytes")

    # Compress the whole array
    compressed_full, full_ratio = lz77_compress(byte_array)
    print(f"Overall zlib compression ratio for full data: {full_ratio:.2f}")
    print(f"Compressed size of the full data: {len(compressed_full)} bytes")

    # Calculate the total compression ratio after decomposing
    total_compressed_size = sum(compressed_sizes)
    decomposition_ratio = total_original_size / total_compressed_size
    print(f"Overall compression ratio after decomposition: {decomposition_ratio:.2f}")

def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    print(f"Processing dataset: {dataset_name}")
    group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T
    byte_array = group.to_numpy().astype(np.float32).tobytes()
    component_sizes = [1,1, 1, 1]  # Adjusted to reflect full 32-bit floats
    decomposition_based_compression(byte_array, component_sizes)

if __name__ == "__main__":
    run_and_collect_data()
