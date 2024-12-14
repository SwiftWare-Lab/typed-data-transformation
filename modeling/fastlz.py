import os
import numpy as np
import pandas as pd
import zlib
from numpy.lib.stride_tricks import as_strided
import lz4.frame as fastlz  # Using lz4 as an alternative example

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

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes = []
    total_original_size = len(byte_array)
    results = []

    for i, comp in enumerate(components):
        compressed_comp, lz77_ratio = fastlz_compress(comp)
        compressed_sizes.append(len(compressed_comp))
       # results.append({
           # "Component": i + 1,
           # "Compression Ratio": lz77_ratio,
           # "Compressed Size": len(compressed_comp)
        #})

    # Compress the whole array
    compressed_full, full_ratio = fastlz_compress(byte_array)
    total_compressed_size = sum(compressed_sizes)
    decomposition_ratio = total_original_size / total_compressed_size

    results.append({
        "Component": "Full Data",
        "Compression Ratio": full_ratio,
        "Compressed Size": len(compressed_full)
    })
    results.append({
        "Component": "decomposition_ratio",
        "Compression Ratio": decomposition_ratio,
        "Compressed Size": total_compressed_size
    })

    return results

def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/solar_wind_f32.tsv"
    # Step 1: Read the TSV file without headers
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

    # Extract the dataset name from the path
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    # Step 2: Drop the first column
    data_without_first_col = ts_data1.drop(ts_data1.columns[0], axis=1)

    # Step 3: Convert the data to float and transpose
    # This results in a 2D NumPy array of type float32
    data_float32 = data_without_first_col.astype(float).T.to_numpy().astype(np.float32)

    # Step 4: Flatten the array to 1D if it's multi-dimensional
    # This ensures that slicing works correctly
    data_flat = data_float32.flatten()

    # Step 5: Slice the array to keep only the first 2048 float32 values
    # Ensure that the data has at least 2048 values to avoid IndexError
    num_floats = 2048
    if len(data_flat) < num_floats:
        raise ValueError(f"Insufficient data: Expected at least {num_floats} float32 values, but got {len(data_flat)}.")
    data_limited = data_flat[:num_floats]

    # Step 6: Convert the sliced array to bytes (8 KB)
    byte_array = data_limited.tobytes()

    # Step 6: Convert the sliced array to bytes (8 KB)
    byte_array = data_limited.tobytes()
    #byte_array = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T.to_numpy().astype(np.float32).tobytes()

    # Different component sizes to try
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
