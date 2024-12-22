import numpy as np
import pandas as pd
import os
import zstandard as zstd
import brotli

def split_bytes_into_components(byte_array, component_sizes):
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

def compress_with_zstd(data, level=3):
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    comp_ratio = len(data) / len(compressed)
    return compressed, comp_ratio

def compress_with_brotli(data, quality=11):
    compressed = brotli.compress(data, quality=quality)
    comp_ratio = len(data) / len(compressed)
    return compressed, comp_ratio

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    total_compressed_size_zstd = 0
    total_compressed_size_brotli = 0

    for i, comp in enumerate(components):
        comp_bytes = comp.tobytes()
        compressed_zstd, comp_ratio_zstd = compress_with_zstd(comp_bytes, level=3)
        compressed_brotli, comp_ratio_brotli = compress_with_brotli(comp_bytes, quality=11)

        total_compressed_size_zstd += len(compressed_zstd)
        total_compressed_size_brotli += len(compressed_brotli)

        print(f"Component {i + 1} Zstd compression ratio: {comp_ratio_zstd}")
        print(f"Component {i + 1} Brotli compression ratio: {comp_ratio_brotli}")

    overall_ratio_zstd = len(byte_array) / total_compressed_size_zstd
    overall_ratio_brotli = len(byte_array) / total_compressed_size_brotli
    print(f"decompose Zstd compression ratio: {overall_ratio_zstd}")
    print(f"decompose Brotli compression ratio: {overall_ratio_brotli}")

    compressed_zstd1, ratio_zstd = compress_with_zstd(byte_array, level=3)
    compressed_brotli1, ratio_brotli = compress_with_brotli(byte_array, quality=11)
    # ratio_zstd = len(byte_array) / total_compressed_size_zstd
    # ratio_brotli = len(byte_array) / total_compressed_size_brotli
    print(f" Zstd compression ratio : {ratio_zstd}")
    print(f" Brotli compression ratio : {ratio_brotli}")


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]

    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")

        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T
        byte_array = group.to_numpy().astype(np.float32).tobytes()
        component_sizes = [2, 1, 1]
        decomposition_based_compression(byte_array, component_sizes)

if __name__ == "__main__":
    run_and_collect_data()
