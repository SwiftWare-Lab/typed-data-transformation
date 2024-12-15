import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import lz4.frame as fastlz
#import pylzma as fastlz
import heapq
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
    # First, ensure the data is in bytes format if it's a numpy array
    if isinstance(data, np.ndarray):
        data = data.tobytes()

    # Check if the bytes object is empty
    if not data:
        return b'', 0

    compressed_data = []
    count = 1
    last = data[0]

    # Iterate over each byte in the data starting from the second byte
    for current in data[1:]:
        if current == last and count < 255:
            count += 1
        else:
            compressed_data.extend([count, last])
            last = current
            count = 1

    # Append the last run
    compressed_data.extend([count, last])

    # Convert list to bytes
    compressed_bytes = bytes(compressed_data)
    compression_ratio = len(data) / len(compressed_bytes) if compressed_bytes else 1

    return compressed_bytes, compression_ratio




def decomposition_based_compression2(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes_rle = []
    total_original_size = len(byte_array)
    results = []

    for i, comp in enumerate(components):
        rle_comp, rle_ratio = rle_compress(comp)
        compressed_sizes_rle.append(len(rle_comp))
        print(f"Component {i}: Compressed size = {len(rle_comp)}")  # Debugging output

    total_compressed_size_rle = sum(compressed_sizes_rle)
    print(f"Total compressed size RLE: {total_compressed_size_rle}")  # Sum check

    return results
def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes_fastlz = []
    compressed_sizes_rle = []
    total_original_size = len(byte_array)
    results = []

    # Loop through each component and perform compression
    for comp in components:
        # FastLZ Compression
        compressed_comp, lz77_ratio = fastlz_compress(comp)
        compressed_sizes_fastlz.append(len(compressed_comp))

        # RLE Compression
        rle_comp, rle_ratio = rle_compress(comp)
        compressed_sizes_rle.append(len(rle_comp))


    # Compress the whole array with FastLZ, RLE, and Huffman
    compressed_full, full_ratio = fastlz_compress(byte_array)
    rle_full, rle_full_ratio = rle_compress(byte_array)

    # Calculate total compressed sizes for FastLZ, RLE, and Huffman
    total_compressed_size_fastlz = sum(compressed_sizes_fastlz)
    total_compressed_size_rle = sum(compressed_sizes_rle)

    # Append overall results for full data
    results.append({
        "Type": "Full Data",
        "Compression Ratio FastLZ": full_ratio,
        "Compressed Size FastLZ": len(compressed_full),
        "Compression Ratio RLE": rle_full_ratio,
        "Compressed Size RLE": len(rle_full),

    })

    # Append decomposition ratios
    results.append({
        "Type": "Decompression",
        "Compression Ratio FastLZ": total_original_size / total_compressed_size_fastlz,
        "Compressed Size FastLZ": total_compressed_size_fastlz,
        "Compression Ratio RLE": total_original_size / total_compressed_size_rle,
        "Compressed Size RLE": total_compressed_size_rle,

    })

    return results


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    ts_data1= pd.read_csv(dataset_path, delimiter='\t', header=None)
   # ts_data1  = ts_data1.iloc [0:2000, :]
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    data_without_first_col = ts_data1.drop(ts_data1.columns[0], axis=1)

    data_float32 = data_without_first_col.astype(float).T.to_numpy().astype(np.float32)
    byte_array = data_float32.flatten().tobytes()

    component_configurations = [
        [1, 1, 1, 1],
        [1, 1, 2],
        [1, 2, 1],
        [1, 3],
        [2, 1, 1],
        [2, 2],
        [3, 1]
    ]

    all_results = []
    for config in component_configurations:
        result = decomposition_based_compression(byte_array, config)
        for res in result:
            res.update({"Configuration": "-".join(map(str, config))})
        all_results.extend(result)

    # Convert results to DataFrame and save to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("/home/jamalids/Documents/solar_wind_f32.csv", index=False)
    print("Saved results to compression_results.csv")

if __name__ == "__main__":
    run_and_collect_data()
