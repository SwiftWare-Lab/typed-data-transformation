import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
import math
#from matplotlib import pyplot as plt

from utils import binary_to_int
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data,decode_decompose,concat_decompose


def float_to_ieee754(f):
    """Convert a float or a numpy array of floats to their IEEE 754 binary representation and return as an integer array."""
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its IEEE 754 binary form."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        # Apply the conversion to each element in the numpy array
        return np.array([float_to_binary_array(single_f) for single_f in f.ravel()]).reshape(f.shape + (32,))
    else:
        # Apply the conversion to a single float
        return float_to_binary_array(f)
def binary_array_to_float(binary_array):
    """Convert an integer array representing an IEEE 754 binary form back to a float or numpy array of floats."""
    def binary_array_to_single_float(binary_bits):
        """Convert a single binary array to a float."""
        # Convert the binary array to a string
        binary_str = ''.join(binary_bits.astype(str))
        # Convert the binary string to an unsigned 32-bit integer
        int_value = int(binary_str, 2)
        # Use numpy to view the integer as a float
        return np.float32(np.uint32(int_value).view(np.float32))

    if binary_array.ndim > 1:
        # Apply the conversion to each element in the numpy array
        return np.array([binary_array_to_single_float(bits) for bits in binary_array.reshape(-1, 32)]).reshape(binary_array.shape[:-1])
    else:
        # Apply the conversion to a single binary array
        return binary_array_to_single_float(binary_array)
def calculate_entropy_float(data):
    """Calculate the Shannon entropy of quantized data."""
    value, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_entropy(binary_data):
    # Calculate the frequency of each unique binary pattern
    frequencies = {}
    for binary in binary_data:
        # Convert the numpy array to a tuple to use it as a dictionary key
        binary_tuple = tuple(binary.flatten())
        if binary_tuple in frequencies:
            frequencies[binary_tuple] += 1
        else:
            frequencies[binary_tuple] = 1

    # Calculate probabilities and entropy
    entropy = 0
    total_count = len(binary_data)
    for freq in frequencies.values():
        probability = freq / total_count
        entropy -= probability * math.log2(probability)

    return entropy
def compress_with_zstd(data, level=3):
    print("level",level)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    #parse_zstd_compressed_data(compressed)
    # comp ratio
    print("data",len(data.tobytes()))
    print("compressed size",len(compressed))
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio
def xor_ieee754_consecutive(ieee_754_arr):
    """Perform XOR between consecutive IEEE 754 binary representations in an ndarray."""
    # Convert each float in the array to IEEE 754 binary representation
    #ieee_754_arr = np.array([float_to_ieee754(num) for num in arr], dtype=np.uint64)

    # Perform XOR between consecutive elements
    xor_results = np.bitwise_xor(ieee_754_arr[1:], ieee_754_arr[:-1])

    return xor_results



def run_and_collect_data():
    results=[]

    dataset_path = "/home/jamalids/Documents/2D/data1/"
    datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if
                f.endswith('.tsv')]



    for dataset_path in datasets:
        result_row = {}
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        group = ts_data1.drop(ts_data1.columns[0], axis=1)
        group = group.T
        group = group.iloc[:, 0:10000000]
        #group = group.iloc[:, 0:10]
        group = group.astype(np.float32).to_numpy().reshape(-1)
        entropy_float = calculate_entropy_float(group)
        print("entropy_float=", entropy_float)

        bool_array_zstd = float_to_ieee754(group)
        entropy_IEE754 = calculate_entropy(bool_array_zstd)

        zstd_IEE = xor_ieee754_consecutive(bool_array_zstd)
        entropy_IEE754_XOR = calculate_entropy(zstd_IEE)



        group_zstd = binary_array_to_float(zstd_IEE)
        entropy_float_XOR = calculate_entropy_float(group_zstd)
        #group_zstd1 = binary_array_to_float(bool_array_zstd)

        # Zstd_XOR
        zstd_compressed_ts, comp_ratio_zstd_default_XOR = compress_with_zstd(group_zstd)
        zstd_compressed_ts_l22, comp_ratio_l22_XOR = compress_with_zstd(group_zstd, 22)
        #ZSTD
        zstd_compressed, comp_ratio_zstd_default = compress_with_zstd(group)
        zstd_compressed_l22, comp_ratio_l22 = compress_with_zstd(group, 22)

        result_row["dataset_name"] = dataset_name
        result_row["XOR_comp_ratio_zstd_default"] = comp_ratio_zstd_default_XOR
        result_row["XOR_comp_ratio_zstd_22"] = comp_ratio_l22_XOR
        result_row["comp_ratio_zstd_default"] = comp_ratio_zstd_default
        result_row["comp_ratio_zstd_22"] = comp_ratio_l22
        result_row["entropy_float"] = entropy_float
        result_row["entropy_float_XOR"] = entropy_float_XOR
        result_row["entropy_IEE754"] = entropy_IEE754
        result_row["entropy_IEE754_XOR"] = entropy_IEE754_XOR



        results.append(result_row)

    return (pd.DataFrame(results))
if __name__ == "__main__":
    df_results = run_and_collect_data()
    df_results.to_csv("zstd.csv")