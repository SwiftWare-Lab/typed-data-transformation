import pandas as pd
import numpy as np
import math

def float_to_ieee754(f):
    """Convert a float or a numpy array of floats to their IEEE 754 binary representation."""
    def float_to_binary_array(single_f):
        """Convert a single float to a binary string."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        return np.array([float_to_binary_array(x) for x in f.ravel()]).reshape(-1, 32)  # Ensure it's reshaped properly for the entire dataset
    else:
        return float_to_binary_array(f)

def calculate_entropy(binary_data):
    """Calculate the entropy of binary data."""
    frequencies = {}
    for binary in binary_data:
        binary_tuple = tuple(binary)
        if binary_tuple in frequencies:
            frequencies[binary_tuple] += 1
        else:
            frequencies[binary_tuple] = 1

    entropy = 0
    total_count = len(binary_data)
    for freq in frequencies.values():
        probability = freq / total_count
        entropy -= probability * math.log2(probability)
    return entropy

def entropy_of_blocks(data, block_size_bytes):
    """Calculate the entropy for each block of binary data, treating data as bits within those blocks."""
    if data.size == 0:
        return []  # Return an empty list if there is no data to process

    # Convert block size from bytes to bits
    bits_per_block = block_size_bytes * 8

    # Since each element in the data array represents 32 bits of the IEEE 754 binary representation:
    elements_per_block = bits_per_block // 32  # Calculate how many 32-bit elements fit in one block

    block_entropies = []
    # Process the data block by block
    for start in range(0, data.shape[0], elements_per_block):
        end = min(start + elements_per_block, data.shape[0])
        block_data = data[start:end]
        if block_data.size > 0:
            # Calculate entropy by flattening the block to treat all bits collectively
            flat_block_data = block_data.flatten()
            entropy = calculate_entropy(flat_block_data.reshape(-1, 32))  # Reshape if necessary to handle the bits correctly
            block_entropies.append(entropy)
    return block_entropies
def decompose_array_three(max_lead, min_tail, array):
    leading_zero_array = array[:, :max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_zero_array = array[:, min_tail:]
    return leading_zero_array, content_array, trailing_zero_array

# Load and preprocess data
dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/msg_bt_f64.tsv"
data = pd.read_csv(dataset_path, delimiter='\t', header=None, dtype=np.float32)
data = data.drop(data.columns[0], axis=1)
data = data.iloc[:4000000, :]
data = data.T
data = data.values.flatten()

# Convert data to IEEE 754 binary form
binary_data = float_to_ieee754(data)

# Calculate entropy for 128 KB blocks
block_size_bytes = 256 * 1024  # 128 KB
block_entropies = entropy_of_blocks(binary_data, block_size_bytes)
print("Block Entropies:", block_entropies)

# Decompose and calculate entropies for each component
max_lead = 8  # Adjust these values as needed
min_tail = 8  # Adjust these values as needed
leading_zeros, content, trailing_zeros = decompose_array_three(max_lead, min_tail, binary_data)
leading_entropies = entropy_of_blocks(leading_zeros, block_size_bytes)
content_entropies = entropy_of_blocks(content, block_size_bytes)
trailing_entropies = entropy_of_blocks(trailing_zeros, block_size_bytes)

print("Leading Zero Entropies:", leading_entropies)
print("Content Entropies:", content_entropies)
print("Trailing Zero Entropies:", trailing_entropies)
