import bz2
import math
import os
import sys
import zlib

import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip
import fpzip
from utils import binary_to_int
import argparse
#from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data,decode_decompose,concat_decompose
##################################
#########################
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




def compute_leading_tailing_zeros(array):
    leading_zeros_array = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        leading_zeros = 0
        # find first non zero in array[i]
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                leading_zeros += 1
            else:
                break
        leading_zeros_array[i] = leading_zeros

    trailing_zeros_array = np.zeros(array.shape[0])
    for i in range(array.shape[0]-1, -1, -1):
        trailing_zeros = 0
        for j in range(array.shape[1]-1, -1, -1):
            if array[i, j] == 0:
                trailing_zeros += 1
            else:
                break
        trailing_zeros_array[i] =  trailing_zeros
    return leading_zeros_array, trailing_zeros_array
def decompose_array(min_lead, max_lead, min_tail, max_tail, array):
    leading_zero_array = array[:, :min_lead]
    assert np.sum(leading_zero_array) == 0
    leading_mixed_array = array[:, min_lead:max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_mixed_array = array[:, min_tail:max_tail]
    trailing_zero_array = array[:, max_tail:]
    assert np.sum(trailing_zero_array) == 0
    return leading_zero_array, leading_mixed_array, content_array, trailing_mixed_array, trailing_zero_array


def decompose_array_three(max_lead, min_tail, array):
    leading_zero_array = array[:, :max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_zero_array = array[:, min_tail:]
    return leading_zero_array, content_array, trailing_zero_array

def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos, funct_name):
    # Calculate min, max, and avg for leading and tail zeros
    min_lead, max_lead, avg_lead = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), int(
        np.mean(leading_zero_pos))
    min_tail, max_tail, avg_tail = int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))

    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead,
          "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)

    # Set bounds based on ad-hoc conditions
    #bnd1 = max_lead if max_lead < 28 else avg_lead
    #bnd2 = min_tail if min_tail >= 4 else 32-avg_tail
    bnd1=8
    bnd2=32-8
    print("Bnd1: ", bnd1, "Bnd2:",bnd2 )

    # Tune decomposition steps
    tune_decomp = [0, 8]

    # Initialize lists to store compressed sizes and dictionaries

    lead_entropy, tail_entropy, content_entropy = [], [], []
    leading_zero_array_orig1, content_array_orig1, trailing_mixed_array_orig1=[],[],[]
    leading, content, tailing = [], [], []
    leading_R, content_R, tailing_R = [], [], []


    for i in tune_decomp:
        print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
        print("Tune Decomp: ", i)

        # Adjust bounds based on tuning step
        bnd1 = bnd1 + i
        bnd2 = bnd2 - i
        print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
        print("Tune Decomp: ", i)

        # Decompose the array into three parts
        leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2,
                                                                                                       image_ts)

        # Compress leading zero array
        ts_m_l, ts_n_l = leading_zero_array_orig.shape
        if ts_n_l != 0:

            leading_entropy = calculate_entropy(leading_zero_array_orig)
            leadinf_float = bits_to_float32(leading_zero_array_orig)
            if(funct_name=='zstd'):
                comp_leading, leading_ratio = compress_with_zstd(leadinf_float, level=3)
            elif(funct_name=='zstd_22'):
                comp_leading, leading_ratio = compress_with_zstd(leadinf_float, level=3)
            elif (funct_name == 'gzip'):
                comp_leading, leading_ratio=compress_with_gzip(leadinf_float)

        else:

            leading_entropy = 0
            comp_leading, leading_ratio=0,0

        # Compress content array
        ts_m_c, ts_n_c = content_array_orig.shape
        if ts_n_c != 0:

            contents_entropy = calculate_entropy(content_array_orig)

            content_float = bits_to_float32(content_array_orig)

            if (funct_name == 'zstd'):
                comp_content, content_ratio = compress_with_zstd(content_float, level=3)
            elif (funct_name == 'zstd_22'):
                comp_content, content_ratio = compress_with_zstd(content_float, level=3)
            elif (funct_name == 'gzip'):
                comp_content, content_ratio = compress_with_gzip(content_float)


        else:

            contents_entropy = 0
            comp_content, content_ratio=0,0

            # Compress trailing mixed array
        ts_m_t, ts_n_t = trailing_mixed_array_orig.shape
        if ts_n_t != 0:

            trailing_entropy = calculate_entropy(trailing_mixed_array_orig)

            trailing_float = bits_to_float32(trailing_mixed_array_orig)
            if (funct_name == 'zstd'):
                comp_trailing, trailing_ratio = compress_with_zstd(trailing_float, level=3)
            elif (funct_name == 'zstd_22'):
                comp_trailing, trailing_ratio = compress_with_zstd(trailing_float, level=3)
            elif (funct_name == 'gzip'):
                comp_trailing, trailing_ratio = compress_with_gzip(trailing_float)



        else:

            trailing_entropy = 0
            comp_trailing, trailing_ratio=0,0
        # Store compressed sizes and dictionaries

        leading.append(comp_leading)
        content.append(comp_content)
        tailing.append(comp_trailing)
        leading_R.append(leading_ratio)
        content_R.append(content_ratio)
        tailing_R.append(trailing_ratio)

        lead_entropy.append(leading_entropy)
        tail_entropy.append(contents_entropy)
        content_entropy.append(trailing_entropy)

        leading_zero_array_orig1.append(leading_zero_array_orig)
        content_array_orig1.append(content_array_orig)
        trailing_mixed_array_orig1.append(trailing_mixed_array_orig)


    return (leading,content,tailing,leading_R,content_R,tailing_R,lead_entropy, tail_entropy, content_entropy)

def bits_to_float32(bit_array):


    # Ensure it's a 1D array
    bit_array = np.array(bit_array).flatten()

    # Calculate how many bits are needed to make the array divisible by 32
    remainder = len(bit_array) % 32
    if remainder != 0:
        padding_needed = 32 - remainder
        # Add zero bits to the end of the array
        bit_array = np.concatenate([bit_array, np.zeros(padding_needed, dtype=np.uint8)])

    # Now the bit_array length is divisible by 32
    num_floats = len(bit_array) // 32
    floats = np.empty(num_floats, dtype=np.float32)

    for i in range(num_floats):
        # Convert 32 bits to a binary string
        binary_str = ''.join(str(int(bit)) for bit in bit_array[i * 32:(i + 1) * 32])
        # Convert the binary string to an integer, then to float32
        floats[i] = np.float32(np.uint32(int(binary_str, 2)).view(np.float32))

    return floats


def measure_total_compressed_size_RLE( huffman_codes):
    dic_size_bits = 0

    # Calculate size of dictionary
    for value,key in huffman_codes:
        key_size_bits = max(1, key.bit_length())  # Number of bits required to represent the integer key
        value_size_bits = len(value)             # Number of bits in the Huffman code string
        dic_size_bits += key_size_bits + value_size_bits


    return  dic_size_bits

# RLE encoding function for values
def rle_encode(value_data):
    encoding = []
    i = 0

    while i < len(value_data):
        count = 1

        # Use np.array_equal to compare subarrays
        while i + 1 < len(value_data) and np.array_equal(value_data[i], value_data[i + 1]):
            count += 1
            i += 1

        # Append the value and its count to the encoding
        encoding.append((value_data[i], count))
        i += 1

    return encoding
def measure_total_compressed_size( encoded_string, huffman_codes):
    dic_size_bits = 0
    #original_size = sys.getsizeof(original_data) * 8  # in bits
    encoded_size = len(encoded_string)  # encoded string is already in bits
    huffman_dict_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in huffman_codes.items()) * 8  # in bits
    for key, value in huffman_codes.items():
        key_size_bits = max(1, key.bit_length())  # Number of bits required to represent the integer key
        value_size_bits = len(value)      # Number of bits in the Huffman code string
        dic_size_bits += key_size_bits + value_size_bits

    total_compressed_size = encoded_size + dic_size_bits
    # print("huffman_dict_size: ", huffman_dict_size)
    #print("encoded_size: ", encoded_size)
    return total_compressed_size,encoded_size,dic_size_bits
def compress_with_zstd(data, level=3):
    print("level",level)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)

    # comp ratio
    print("data",len(data.tobytes()))
    print("compressed size",len(compressed))
    comp_ratio = len(data.tobytes()) / len(compressed)
    compressed_size= len(compressed)
    return compressed_size, comp_ratio
#########################################################################
def compress_with_gzip1(array):
    # Step 2: Convert the array to a byte representation
    array_bytes = array.tobytes()

    # Step 3: Compress the byte representation using gzip
    compressed_data = gzip.compress(array_bytes)

    # Step 4: Calculate original and compressed sizes
    original_size = len(array_bytes)  # Size of the original array in bytes
    compressed_size = len(compressed_data)  # Size of the compressed data in bytes

    # Step 5: Calculate the compression ratio
    compression_ratio = original_size / compressed_size
    return  compressed_size ,compression_ratio


def compress_with_bz2(array, compresslevel=9):

    # Step 1: Convert the array to bytes
    array_bytes = array.tobytes()

    # Step 2: Compress the byte representation using bz2
    compressed_data = bz2.compress(array_bytes, compresslevel=compresslevel)

    # Step 3: Calculate the sizes
    original_size = len(array_bytes)  # Size of the original array in bytes
    compressed_size = len(compressed_data)  # Size of the compressed data in bytes

    # Step 4: Calculate the compression ratio
    compression_ratio = original_size / compressed_size

    # Return compressed size and compression ratio
    return compressed_size, compression_ratio
####################################################
#def compress_with_zlib(array, compression_level=6):
def compress_with_gzip(array):
    compression_level = 6
    # Step 1: Convert the array to a byte representation
    array_bytes = array.tobytes()

    # Step 2: Compress the byte representation using zlib with specified compression level
    compressed_data = zlib.compress(array_bytes, level=compression_level)

    # Step 3: Calculate original and compressed sizes
    original_size = len(array_bytes)  # Size of the original array in bytes
    compressed_size = len(compressed_data)  # Size of the compressed data in bytes

    # Step 4: Calculate the compression ratio
    compression_ratio = original_size / compressed_size

    return compressed_size, compression_ratio
####################################################

def are_equal(val1, val2):
    # Check if two values are equal, treating NaN as equal to NaN
    if np.isnan(val1) and np.isnan(val2):
        return True
    return val1 == val2

def split_array_on_multiple_consecutive_values(data, threshold_percentage=9):
    total_length = len(data)
    threshold = total_length * (threshold_percentage / 100.0)
    print("threshold",threshold)
    threshold=3


    total_length = len(data)
    consecutive_count = 1
    start_idx = 0
    metadata = []
    non_consecutive_array = []

    for i in range(1, total_length):
        if are_equal(data[i], data[i - 1]):
            consecutive_count += 1
        else:
            if consecutive_count > threshold:
                metadata.append({
                    'start_index': i - consecutive_count,  # The first repetitive value index
                    'value': data[i - 1],
                    'count': consecutive_count
                })
            else:
                non_consecutive_array.extend(data[start_idx:i])
            start_idx = i
            consecutive_count = 1

        # Handle the last segment
    if consecutive_count > threshold:
        metadata.append({
                'start_index': total_length - consecutive_count,
                # The first repetitive value index for the last segment
                'value': data[-1],
                'count': consecutive_count
        })
    else:
        non_consecutive_array.extend(data[start_idx:])
    non_consecutive_array = np.array(non_consecutive_array, dtype=np.float32).reshape(-1)

    return non_consecutive_array, metadata


def convert_RLE(metadata):
    if isinstance(metadata, list):
        # Initialize an empty list to store all the consecutive values
        consecutive_values = []

        # Loop through each dictionary in the metadata list
        for item in metadata:
            # Extract 'value', 'count', and 'start_index' and append them consecutively to the list
            consecutive_values.extend([item['start_index'], item['value'], item['count']])


        # Convert the list to a float32 NumPy array
        metadata1 = np.array(consecutive_values, dtype=np.float32).reshape(-1)
    else:
        # If metadata is not a list of dictionaries, handle it differently as needed
        metadata1 = np.array(metadata, dtype=np.float32).reshape(-1)
    return metadata1



def calculate_exact_metadata_size(metadata):
    total_bits = 0

    for entry in metadata:
        start_index = entry['start_index']
        value = entry['value']
        count = entry['count']

        # Calculate the number of bits required for each field
        start_index_bits = math.ceil(math.log2(start_index +1))
        if math.isnan(value):
            value = -99  # Treat NaN as -99
        if value >= 0:
            # For positive values, use log2(value + 1)
            value_bits = math.ceil(math.log2(value + 1)) if value > 0 else 1
        else:
            # For negative values, use two's complement
            value_bits = math.ceil(math.log2(abs(value) + 1)) + 1  # +1 for the sign bit

        #value_bits =  math.ceil(math.log2(value +1))
        count_bits = math.ceil(math.log2(count + 1))  # Number of bits for the count

        # Total bits for this metadata entry
        total_bits += start_index_bits + value_bits + count_bits

    return total_bits
def run_and_collect_data(dataset_path):
    #dataset_path = "/home/jamalids/Documents/2D/data1/other/"
    dataset_path ="/home/jamalids/Documents/2D/data1/Fcbench/TS/L/citytemp_f32.tsv"
    datasets = [dataset_path]
    #datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if
               # f.endswith('.tsv')]
    results = []
    for dataset_path in datasets:
        result_row = {}

        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print("datasetname##################################",dataset_name)
        group = ts_data1.drop(ts_data1.columns[0], axis=1)
        group=group.iloc[:400,:]
        group = group.T
        #group = group.iloc[:, 0:4000000]
        verify_flag_final = False
        m, n = 8, 1
        ts_n = 32

        group = group.astype(np.float32).to_numpy().reshape(-1)

        entropy_float = calculate_entropy_float(group)
        print("entropy_float=", entropy_float)

        # Calculate the total number of elements
        total_elements = len(group)
        positive_values = np.sum(group > 0)
        negative_values = np.sum(group < 0)

        # Zstd
        zstd_compressed_ts, comp_ratio_zstd_default = compress_with_zstd(group)
        zstd_compressed_ts_l22, comp_ratio_l22 = compress_with_zstd(group, 22)
        #gzip
        gzip_compressed_ts_l22, comp_ratio_gzip=compress_with_gzip(group)
        #fpzip
        fpzip_compressed_ts_l22, comp_ratio_fpzip = compress_with_bz2(group)
        bool_array = float_to_ieee754(group)
        bool_array_size_bits = bool_array.nbytes  # Size in bits

        # Split array and apply RLE
        non_consecutive_array, metadata = split_array_on_multiple_consecutive_values(group, threshold_percentage=1)
        metadata1 = convert_RLE(metadata)
        metadata_array = float_to_ieee754(metadata1)
        group2 = group
        #size_metadata = len(metadata) * 96  # Example size calculation
        size_metadata = calculate_exact_metadata_size(metadata)
        group3 = non_consecutive_array


        new_array_size = group3.shape[0] - group3.shape[0] % m
        group3 = group3[:new_array_size]
        ts_m1 = group3.shape[0]
        m1 = 1
        n1 = 32
        bool_array3 = float_to_ieee754(group3)



        compress_list = ['zstd','gzip']


        for func_name  in compress_list:
            group = non_consecutive_array

            bool_array = float_to_ieee754(group)
            entropy_all = calculate_entropy(bool_array)
            print("entropy_all", entropy_all)

            # Decomposition-based compression
            l_z_array, t_z_array = compute_leading_tailing_zeros(bool_array)
            (leading, content, tailing, leading_R, content_R, tailing_R, lead_entropy, tail_entropy,
             content_entropy) = decomposition_based_compression(bool_array, l_z_array, t_z_array, func_name)

            result_row = {"Original Size (bits)": bool_array_size_bits,"func_name": func_name}
            zstd_encoded_b = {}
            zstd_encoded_b_R = {}

            for idx, leading_zstd1 in enumerate(leading, start=1):
                if leading_zstd1 == 0:
                    result_row[f"b{idx}_eading_size"] = 0
                else:
                    result_row[f"b{idx}_leading_size"] = leading_zstd1 * 8
                    zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + leading_zstd1 * 8

            for idx, content_zstd1 in enumerate(content, start=1):
                if content_zstd1 == 0:
                    result_row[f"b{idx}_content_size"] = 0
                else:
                    result_row[f"b{idx}_content_size"] = content_zstd1 * 8
                    zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + content_zstd1 * 8

            for idx, tailing_zstd1 in enumerate(tailing, start=1):
                if tailing_zstd1 == 0:
                    result_row[f"b{idx}_tailing_size"] = 0
                else:
                    result_row[f"b{idx}_tailing_size"] = tailing_zstd1 * 8
                    zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + tailing_zstd1 * 8

            ############################################################
            for idx, leading_zstd1 in enumerate(leading_R, start=1):
                result_row[f"b{idx}_leading_comp_ratio"] = leading_zstd1
                zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + leading_zstd1

            for idx, content_zstd1 in enumerate(content_R, start=1):
                result_row[f"b{idx}_content_comp_ratio"] = content_zstd1
                zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + content_zstd1

            for idx, tailing_zstd1 in enumerate(tailing_R, start=1):
                result_row[f"b{idx}_tailing_comp_ratio"] = tailing_zstd1
                zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + tailing_zstd1

            #######################################################################

            for idx in zstd_encoded_b:
                result_row[f"com_ratio_b{idx}"] = bool_array_size_bits / (
                        zstd_encoded_b[idx] + size_metadata) if \
                    zstd_encoded_b[idx] > 0 else None

            # Store Zstd and Huffman results
            result_row["comp_ratio_zstd_default"] = comp_ratio_zstd_default
            result_row["comp_ratio_l22"] = comp_ratio_l22
            result_row["comp_ratio_zlib"] = comp_ratio_gzip
            result_row["comp_ratio_bz2"] = comp_ratio_fpzip

            result_row["bool_array_size_bits"] = bool_array_size_bits
            result_row["entropy_remainig"] = entropy_all
            result_row["entropy_float"] = entropy_float
            result_row["verify_flag_final"] = verify_flag_final
            result_row["len(metadata)"] = len(metadata)
            result_row["len(non_consecutive_array)"] = len(non_consecutive_array)
            result_row["dataset_name"] = dataset_name

            results.append(result_row)

    save_results(pd.DataFrame(results), dataset_name, func_name)


    return pd.DataFrame(results)




def save_results(df_results, name_dataset,func_name):

   com_ratio_cols = [col for col in df_results.columns if col.startswith("com_ratio_b")]
   if  com_ratio_cols:  # Ensure the list is not empty
       df_results[f"max_Decom_com_ratio"] = df_results[com_ratio_cols ].max(axis=1)
   entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]
   if entropy_cols:  # Ensure the list is not empty
       entropy_full_data = df_results[entropy_cols].max().max()
   else:
       entropy_full_data = 0  # Fallback value if no columns exist
   df_results.to_csv("Decom+zstd+gzip.csv")
   return df_results

def arg_parser():
    parser = argparse.ArgumentParser(description='Compress one dataset and store the log.')
    parser.add_argument('--dataset', dest='dataset_path', help='Path to the UCR dataset tsv/csv.')
    parser.add_argument('--variant', dest='variant', default="dictionary", help='Variant of the algorithm.')
    parser.add_argument('--pattern', dest='pattern', default="10*16", help='Pattern to match the files.')
    parser.add_argument('--outcsv', dest='log_file', default="./log_out.csv", help='Output directory for the sbatch scripts.')
    parser.add_argument('--nthreads', dest='num_threads', default=1, type=int, help='Number of threads to use.')
    parser.add_argument('--mode', dest='mode',default="signal", help='run mode.')


    return parser

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    dataset_path = args.dataset_path
    comp_variant = args.variant
    pattern = args.pattern
    log_file = args.log_file
    num_threads = args.num_threads
    mode = args.mode
    df_results = run_and_collect_data(dataset_path)