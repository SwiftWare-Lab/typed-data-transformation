import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip

from matplotlib import pyplot as plt
def decompose_array_three(max_lead, min_tail, array):
    leading_zero_array = array[:, :max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_zero_array = array[:, min_tail:]
    return leading_zero_array, content_array, trailing_zero_array
def compress_with_zstd(data, level=3):
    print("level",level)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)

    # comp ratio
    print("data",len(data.tobytes()))
    print("compressed size",len(compressed))
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio
def decomposition_based_compression(image_ts, data):
    bnd1 = 16
    bnd2 = 32 - 8
    print("Bnd1: ", bnd1, "Bnd2:", bnd2)


    leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2,image_ts)
    leading_zero_array_int8 = leading_zero_array_orig.astype(np.int8)
    content_array_int8 = content_array_orig.astype(np.int8)
    trailing_mixed_array_int8 = trailing_mixed_array_orig.astype(np.int8)
    comp_zstd_leading, leading_zstd_ratio = compress_with_zstd( leading_zero_array_int8, level=3)


    comp_zstd_content, content_zstd_ratio = compress_with_zstd(content_array_int8, level=3)


    comp_zstd_trailing, trailing_zstd_ratio = compress_with_zstd(trailing_mixed_array_int8, level=3)
    comp_data=comp_zstd_trailing+comp_zstd_content+comp_zstd_leading
    comp_ratio = len(data.tobytes()) / len(comp_data)
    print("comp-ratio",comp_ratio)
    #print("comp-zstd",comp_data)

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

def float_to_ieee754_64(f):
    """
    Convert a float or a numpy array of floats to their IEEE 754 64-bit binary representation.

    Args:
        f: A float or a numpy array of floats.

    Returns:
        Numpy array with binary representation of shape (..., 64).
    """
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its 64-bit IEEE 754 binary form."""
        single_f = np.float64(single_f)  # Ensure it's a 64-bit float
        binary_str = format(single_f.view(np.uint64), '064b')  # Convert to 64-bit binary
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        # Flatten the array, convert each float, and reshape back to original dimensions
        original_shape = f.shape
        binary_arrays = [float_to_binary_array(single_f) for single_f in f.ravel()]
        return np.array(binary_arrays).reshape(original_shape + (64,))
    else:
        # Apply the conversion to a single float
        return float_to_binary_array(f)

def plot_bitmap_standalone(bool_array, name):
    """
    Plots a standalone bitmap visualization of the boolean array representing IEEE 754 64-bit binary format.

    Args:
        bool_array: Numpy array of shape (n, 64), where n is the number of floats and 64 is the bit length.
        name: The name of the file to save the bitmap plot.
    """
    plt.figure(figsize=(12, 8))  # Adjust figure size for better visualization

    # Create the bitmap plot
    plt.imshow(bool_array, cmap='gray_r', aspect='auto')

    # Add labels and title
    plt.xlabel('Bit Position (0-63)')
    plt.ylabel('Float Index')

    # Add a color bar to indicate 0 and 1 mapping
    plt.colorbar(label='Bit Value')

    # Save the figure
    plt.savefig(name)
    plt.show()

def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]
    results = []
    for dataset_path in datasets:
        result_row = {}

        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print("datasetname##################################", dataset_name)

        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float)  # Ensure group is numeric
        group = group.T
        #group = group.iloc[:, 0:128000]
        group = group.astype(np.float64).to_numpy().reshape(-1)
        #bool_array = float_to_ieee754_64(group)
        bool_array =float_to_ieee754(group)

       # filename = f"{dataset_name}_all128.png"
      #  plot_bitmap_standalone(bool_array , filename)
        decomposition_based_compression(bool_array, group)


if __name__ == "__main__":

    df_results = run_and_collect_data()