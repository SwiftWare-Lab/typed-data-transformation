import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import lz4.frame as fastlz
import lzma
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

def lzma_compression(data):
    compressed = lzma.compress(data)  # Use lzma.compress directly
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


def huffman_coding(data):
    """Create a Huffman tree and encode the data."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()

    # Frequency dictionary of the data
    frequency = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = defaultdict(str)
    for pair in heap[0][1:]:
        huffman_dict[pair[0]] = pair[1]

    # Encode data
    encoded_data = ''.join(huffman_dict[byte] for byte in data)
    compressed_bytes = bytes(int(encoded_data[i:i+8], 2) for i in range(0, len(encoded_data), 8))
    compression_ratio = len(data) / len(compressed_bytes)

    return compressed_bytes, compression_ratio, huffman_dict


def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    compressed_sizes_fastlz = []
    compressed_sizes_rle = []
    compressed_sizes_huffman = []
    compressed_sizes_lzma=[]

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

        # # Huffman Compression
        # huffman_comp, huffman_ratio, _ = huffman_coding(comp)
        # compressed_sizes_huffman.append(len(huffman_comp))
        #lzma

        compressed_comp, lzma_ratio = lzma_compression(comp)
        compressed_sizes_lzma.append(len(compressed_comp))

    # Compress the whole array with FastLZ, RLE, and Huffman
    compressed_full, full_ratio = fastlz_compress(byte_array)
    lzma_full, lzma_ratio = lzma_compression(byte_array)
    rle_full, rle_full_ratio = rle_compress(byte_array)
    huffman_full, huffman_full_ratio, _ = huffman_coding(byte_array)  # Adjusted unpacking here

    # Calculate total compressed sizes for FastLZ, RLE, and Huffman
    total_compressed_size_fastlz = sum(compressed_sizes_fastlz)
    total_compressed_size_lzma = sum(compressed_sizes_lzma)
    total_compressed_size_rle = sum(compressed_sizes_rle)
    total_compressed_size_huff = sum(compressed_sizes_huffman)

    # Append overall results for full data
    results.append({
        "Type": "Full Data",
        "Compression Ratio FastLZ": full_ratio,
        "Compressed Size FastLZ": len(compressed_full),
        "Compression Ratio Lzma": lzma_ratio,
        "Compressed Size Lzma": len(lzma_full),
        "Compression Ratio RLE": rle_full_ratio,
        "Compressed Size RLE": len(rle_full),
        # "Compression Ratio Huffman": huffman_full_ratio,
        # "Compressed Size Huffman": len(huffman_full)
    })

    # Append decomposition ratios
    results.append({
        "Type": "Decompression",
        "Compression Ratio FastLZ": total_original_size / total_compressed_size_fastlz,
        "Compressed Size FastLZ": total_compressed_size_fastlz,
        "Compression Ratio Lzma": total_original_size / total_compressed_size_lzma,
        "Compressed Size Lzma": total_compressed_size_lzma,
        "Compression Ratio RLE": total_original_size / total_compressed_size_rle,
        "Compressed Size RLE": total_compressed_size_rle,
        # "Compression Ratio Huffman": total_original_size / total_compressed_size_huff,
        # "Compressed Size Huffman": total_compressed_size_huff
    })

    return results


def run_and_collect_data():
    dataset_path = 'C:\\Users\\jamalids\\Documents\\db\\hst_wfc3_ir_f32.tsv'

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
            res.update({"Configuration": "_".join(map(str, config))})
        all_results.extend(result)

    # Convert results to DataFrame and save to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("C:\\Users\\jamalids\\Documents\\hst_wfc3_ir_f32.csv")
    print("Saved results to compression_results.csv")

if __name__ == "__main__":
    run_and_collect_data()
