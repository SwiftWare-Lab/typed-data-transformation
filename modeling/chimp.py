import math
import os
import sys

import pandas as pd
import numpy as np
import struct

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
def xor_bin(bin1, bin2):
    """ Perform XOR operation between two binary strings. """
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bin1, bin2))
def count_leading_trailing_zeros(bin_str):
    """ Count leading and trailing zeros in a binary string. """
    leading_zeros = len(bin_str) - len(bin_str.lstrip('0'))
    trailing_zeros = len(bin_str) - len(bin_str.rstrip('0'))
    return leading_zeros, trailing_zeros
def chimp_encode1(xor_result, leading_zeros, trailing_zeros):
    """ Encode using a simplified version of the Chimp algorithm. """
    if xor_result.count('1') == 0:
        return '0'  # No change
    # Encode leading zeros and the length of the meaningful bits
    meaningful_bits = xor_result.strip('0')
    return f'1{leading_zeros:05b}{len(meaningful_bits):06b}{meaningful_bits}'


def chimp_encode(xor_array):
    """ Encode using a simplified version of the Chimp algorithm and return the size of the encoded data. """
    # Convert XOR array back to binary string
    xor_str = ''.join(map(str, xor_array))
    # Measure leading zeros, trailing zeros, and significant bits
    leading_zeros = len(xor_str) - len(xor_str.lstrip('0'))
    trailing_zeros = len(xor_str) - len(xor_str.rstrip('0'))
    center_bits = len(xor_str) - leading_zeros - trailing_zeros

    if center_bits == 0:  # No change, encode as a single bit
        return 1
    else:
        # Encode the number of leading zeros, center bits, and the center bits themselves
        # This is a simplified model: leading zeros (5 bits), length of center bits (5 bits), center bits
        return 1 + 5 + 5 + center_bits


dataset_path ="/home/jamalids/Documents/2D/data1/HPC/H/wave_f32.tsv"
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
ts_data1=ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1=ts_data1.T
ts_data1 = ts_data1.iloc[0:1,75:83]
group = ts_data1.astype(np.float32).to_numpy().reshape(-1)


binary_numbers = float_to_ieee754(group)

print(binary_numbers)
xor_results = [xor_bin(binary_numbers[i], binary_numbers[i+1]) for i in range(len(binary_numbers)-1)]
print(xor_results)

zeros_count = [count_leading_trailing_zeros(xor) for xor in xor_results]
print(zeros_count)

# Encode each XOR result

encoded_sizes = [chimp_encode(xor) for xor in xor_results]
total_encoded_size = sum(encoded_sizes)
print("Encoded sizes for each XOR result:", encoded_sizes)
print("Total encoded size in bits:", total_encoded_size)