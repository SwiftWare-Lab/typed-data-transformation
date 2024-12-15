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


def unique_ratio_pairs(data):
    # Ensure data is an iterable of bytes
    if isinstance(data, np.ndarray):
        data = data.tobytes()

    # Pair the bytes and create a list of tuples
    byte_pairs = list(zip(data[0::2], data[1::2])) if len(data) % 2 == 0 else list(
        zip(data[0::2], data[1::2] + (None,)))

    # Calculate unique count and total count
    unique_count = len(set(byte_pairs))
    total_count = len(byte_pairs)
    return unique_count / total_count if total_count > 0 else 0

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes_fastlz = []
    compressed_sizes_rle = []
    total_original_size = len(byte_array)
    results = []

    entropy_full_data = calculate_entropy(byte_array)
    unique_ratio_full_data = unique_ratio(byte_array)
    unique_ratio_pairs_full_data = unique_ratio_pairs(byte_array)

    weighted_entropy = 0
    weighted_unique_ratio = 0
    weighted_unique_ratio_pairs = 0

    for comp in components:
        comp_size = len(comp)
        compressed_comp, lz77_ratio = fastlz_compress(comp)
        compressed_sizes_fastlz.append(len(compressed_comp))
        rle_comp, rle_ratio = rle_compress(comp)
        compressed_sizes_rle.append(len(rle_comp))

        comp_entropy = calculate_entropy(comp)
        comp_unique_ratio = unique_ratio(comp)
        comp_unique_ratio_pairs = unique_ratio_pairs(comp)

        weighted_entropy += (comp_size / total_original_size) * comp_entropy
        weighted_unique_ratio += (comp_size / total_original_size) * comp_unique_ratio
        weighted_unique_ratio_pairs += (comp_size / total_original_size) * comp_unique_ratio_pairs

        results.append({
            "Type": "Component",
            "Compression Ratio FastLZ": lz77_ratio,
            "Compressed Size FastLZ": len(compressed_comp),
            "Compression Ratio RLE": rle_ratio,
            "Compressed Size RLE": len(rle_comp),
            "Unique Ratio": comp_unique_ratio,
            "Unique Ratio Pairs": comp_unique_ratio_pairs,
            "Entropy": comp_entropy,
        })

    compressed_full, full_ratio = fastlz_compress(byte_array)
    rle_full, rle_full_ratio = rle_compress(byte_array)

    total_compressed_size_fastlz = sum(compressed_sizes_fastlz)
    total_compressed_size_rle = sum(compressed_sizes_rle)

    results.append({
        "Type": "Full Data",
        "Compression Ratio FastLZ": full_ratio,
        "Compressed Size FastLZ": len(compressed_full),
        "Compression Ratio RLE": rle_full_ratio,
        "Compressed Size RLE": len(rle_full),
        "Unique Ratio": unique_ratio_full_data,
        "Unique Ratio Pairs": unique_ratio_pairs_full_data,
        "Entropy": entropy_full_data
    })

    # Add decompression results back with enhanced details
    results.append({
        "Type": "Decompression",
        "Compression Ratio FastLZ": total_original_size / total_compressed_size_fastlz,
        "Compressed Size FastLZ": total_compressed_size_fastlz,
        "Compression Ratio RLE": total_original_size / total_compressed_size_rle,
        "Compressed Size RLE": total_compressed_size_rle,
        "Weighted Entropy": weighted_entropy,
        "Weighted Unique Ratio": weighted_unique_ratio,
        "Weighted Unique Ratio Pairs": weighted_unique_ratio_pairs
    })

    return results

def decomposition_based_compression1(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes_fastlz = []
    compressed_sizes_rle = []
    total_original_size = len(byte_array)
    results = []

    entropy_full_data = calculate_entropy(byte_array)
    unique_ratio_full_data = unique_ratio(byte_array)

    weighted_entropy = 0
    weighted_unique_ratio = 0

    for comp in components:
        comp_size = len(comp)
        compressed_comp, lz77_ratio = fastlz_compress(comp)
        compressed_sizes_fastlz.append(len(compressed_comp))
        rle_comp, rle_ratio = rle_compress(comp)
        compressed_sizes_rle.append(len(rle_comp))

        comp_entropy = calculate_entropy(comp)
        comp_unique_ratio = unique_ratio(comp)

        weighted_entropy += (comp_size / total_original_size) * comp_entropy
        weighted_unique_ratio += (comp_size / total_original_size) * comp_unique_ratio

        results.append({
            "Type": "Component",
            "Compression Ratio FastLZ": lz77_ratio,
            "Compressed Size FastLZ": len(compressed_comp),
            "Compression Ratio RLE": rle_ratio,
            "Compressed Size RLE": len(rle_comp),
            "Unique Ratio": comp_unique_ratio,
            "Entropy": comp_entropy,
        })

    compressed_full, full_ratio = fastlz_compress(byte_array)
    rle_full, rle_full_ratio = rle_compress(byte_array)
    total_compressed_size_fastlz = sum(compressed_sizes_fastlz)
    total_compressed_size_rle = sum(compressed_sizes_rle)

    results.append({
        "Type": "Full Data",
        "Compression Ratio FastLZ": full_ratio,
        "Compressed Size FastLZ": len(compressed_full),
        "Compression Ratio RLE": rle_full_ratio,
        "Compressed Size RLE": len(rle_full),
        "Unique Ratio": unique_ratio_full_data,
        "Entropy": entropy_full_data
    })
    results.append({
        "Type": "Decomposition",
        "Compression Ratio FastLZ": total_original_size / total_compressed_size_fastlz,
        "Compressed Size FastLZ": total_compressed_size_fastlz,
        "Compression Ratio RLE": total_original_size / total_compressed_size_rle,
        "Compressed Size RLE": total_compressed_size_rle,
        "Weighted Entropy": weighted_entropy,
        "Weighted Unique Ratio": weighted_unique_ratio
    })
    return results


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    data_without_first_col = ts_data1.drop(ts_data1.columns[0], axis=1)
    data_float32 = data_without_first_col.astype(float).T.to_numpy().astype(np.float32)
    byte_array = data_float32.flatten().tobytes()
    component_configurations = [
        [1, 1, 1, 1],
        [2, 1, 1],

    ]
    all_results = []
    for config in component_configurations:
        result = decomposition_based_compression(byte_array, config)
        for res in result:
            res.update({"Configuration": "-".join(map(str, config))})
        all_results.extend(result)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("/home/jamalids/Documents/solar_wind_f32.csv", index=False)
    print("Saved results to compression_results.csv")

if __name__ == "__main__":
    run_and_collect_data()
