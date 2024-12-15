import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import lz4.frame as fastlz
from collections import Counter, defaultdict

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

def fastlz_compress(data):
    compressed = fastlz.compress(data)
    compression_ratio = len(data) / len(compressed)
    return compressed, compression_ratio

def rle_compress(data):
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    if not data:
        return b'', 0
    compressed_data = []
    count = 1
    last = data[0]
    for current in data[1:]:
        if current == last and count < 255:
            count += 1
        else:
            compressed_data.extend([count, last])
            last = current
            count = 1
    compressed_data.extend([count, last])
    compressed_bytes = bytes(compressed_data)
    compression_ratio = len(data) / len(compressed_bytes) if compressed_bytes else 1
    return compressed_bytes, compression_ratio

def calculate_entropy(data):
    freq = Counter(data)
    total = len(data)
    probabilities = [count / total for count in freq.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def unique_ratio(data):
    unique_count = len(set(data))
    total_count = len(data)
    return unique_count / total_count

def chunk_data(data, chunk_size):
    """Divide data into chunks of specified size."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def analyze_components(components, chunk_size=65536):
    """Analyze each component in chunks."""
    results = []
    for component in components:
        chunks = chunk_data(component, chunk_size)
        total_component_size = len(component)
        weighted_entropy = 0
        weighted_ur=0

        for chunk in chunks:
            compressed, compression_ratio = fastlz_compress(chunk)
            entropy = calculate_entropy(chunk)
            ur = unique_ratio(chunk)
            chunk_size = len(chunk)

            weighted_entropy += (chunk_size / total_component_size) * entropy
            weighted_ur += (chunk_size / total_component_size) * ur
            #
            # results.append({
            #     "Compression Ratio": compression_ratio,
            #     "Compressed Size": len(compressed),
            #     "Entropy": entropy,
            #     "Unique Ratio": ur,
            #     "Chunk Size": chunk_size
            # })

        results.append({
            "Component Summary": "Weighted Entropy",
            "Weighted Entropy": weighted_entropy,
            "Weighted Unique Ratio":weighted_ur
        })

    return results


def analyze_components1(components, chunk_size=65536):
    """Analyze each component in chunks and add debugging output."""
    results = []
    for index, component in enumerate(components):
        chunks = chunk_data(component, chunk_size)
        total_component_size = len(component)
        weighted_entropy = 0
        weighted_ur = 0

        for chunk in chunks:
            compressed, compression_ratio = fastlz_compress(chunk)
            entropy = calculate_entropy(chunk)
            ur = unique_ratio(chunk)
            chunk_real_size = len(chunk)

            weighted_entropy += (chunk_real_size / total_component_size) * entropy
            weighted_ur += (chunk_real_size / total_component_size) * ur

            print(f"Debug Info - Component {index}, Chunk Size: {chunk_real_size}, Entropy: {entropy}, UR: {ur}")

        results.append({
            "Component Index": index,
            "Component Summary": "Weighted Metrics",
            "Weighted Entropy": weighted_entropy,
            "Weighted Unique Ratio": weighted_ur
        })

    return results


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    ts_data1 = ts_data1.iloc[0:16384, :]
    data_without_first_col = ts_data1.drop(ts_data1.columns[0], axis=1)
    data_float32 = data_without_first_col.astype(float).T.to_numpy().astype(np.float32)
    byte_array = data_float32.flatten().tobytes()
    component_configurations = [
        [1, 1, 1, 1],
        [2, 1, 1],
    ]
    all_results = []
    for config in component_configurations:
        components = split_bytes_into_components(byte_array, config)
        result = analyze_components(components)
        for res in result:
            res.update({"Configuration": "-".join(map(str, config))})
        all_results.extend(result)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("/home/jamalids/Documents/solar_wind_f32.csv", index=False)
    print("Saved results to compression_results.csv")

if __name__ == "__main__":
    run_and_collect_data()
