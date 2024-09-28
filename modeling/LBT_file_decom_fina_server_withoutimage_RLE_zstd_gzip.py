
import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip
from utils import binary_to_int
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data,decode_decompose,concat_decompose
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

def get_dict(bool_array, m, n,ts_m, ts_n):
    rectangles = {}
    rectangles1 = {}
    for i in range(0, ts_n, n):
        for j in range(0, ts_m, m):
            rect = bool_array[j:j + m, i:i + n]
            rect_int=binary_to_int(rect)
            # increment the number of times we have seen this rectangle
            rectangles[rect_int] = rectangles.get(rect_int, 0) + 1

    return rectangles


def compute_huffman_code(dict_code,original_data_bool,m, n):
    # Extract pattern counts from dict_code
    pattern_count = list(dict_code.values())

    # Create Huffman tree from dict_code
    root = create_huffman_tree(dict_code)

    # Create Huffman codes dictionary
    codes = {}
    create_huffman_codes(root, "", codes)
    encode_text = encode_data(original_data_bool, m, n, codes)


    # Create a mapping of keys from dict_code to indices in pattern_count
    key_to_index = {key: index for index, key in enumerate(dict_code)}

    estimated_size = 0
    for key in codes:
        # Convert key to match dict_code keys and find corresponding index
        if key in key_to_index:
            index = key_to_index[key]
            estimated_size += len(codes[key]) * pattern_count[index]
        else:
            # Handle cases where key does not match
            raise KeyError(f"Key {key} not found in dict_code")

    dict_size = 0
    for value in codes.values():
        dict_size += len(value)
    tree_size = calculate_size_of_huffman_tree(root)

    return estimated_size, estimated_size + dict_size,codes, root,tree_size,encode_text

def pattern_based_compressor(original_data_bool, m, n,ts_m, ts_n):
    # for each rectangle 4x8, convert it to an integer and udpate the dictionary
    rectangles = get_dict(original_data_bool, m, n,ts_m, ts_n)
    # create a dictionary of code words
    estimated_size, estimated_all,cw_dict,root,tree_size,encode_text=compute_huffman_code(rectangles,original_data_bool,m, n)

    return cw_dict,root,tree_size,encode_text
def pattern_based_decompressor(inverse_cw_dict,encoded_text ,m, n,ts_m, ts_n):
    root1 = create_huffman_tree_from_dict(inverse_cw_dict)
    Decodedata = decode(encoded_text, root1, m, n, (ts_m, ts_n))
    return Decodedata
def pattern_based_decompressor_compose(inverse_cw_dict,encoded_text ,m, n,ts_m, ts_n):
    root1 = create_huffman_tree_from_dict(inverse_cw_dict)
    Decodedata = decode_decompose(encoded_text, root1, m, n, (ts_m, ts_n))
    return Decodedata
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


def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos, m, n):
    # Calculate min, max, and avg for leading and tail zeros
    min_lead, max_lead, avg_lead = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), int(
        np.mean(leading_zero_pos))
    min_tail, max_tail, avg_tail = int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))

    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead,
          "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)

    # Set bounds based on ad-hoc conditions
    bnd1 = max_lead if max_lead < 28 else avg_lead
    bnd2 = min_tail if min_tail >= 4 else 32-avg_tail
    bnd1=8
    bnd2=32-8
    print("Bnd1: ", bnd1, "Bnd2:",bnd2 )

    # Tune decomposition steps
    tune_decomp = [0, 8,16]

    # Initialize lists to store compressed sizes and dictionaries
    lead_comp_size, tail_comp_size, content_comp_size = [], [], []
    lead_comp_size_d, tail_comp_size_d, content_comp_size_d = [], [], []
    lead_entropy, tail_entropy, content_entropy = [], [], []
    lead_shape_m, tail_shap_m, content_shap_m = [], [], []
    lead_shape_n, tail_shap_n, content_shap_n = [], [], []
    leading_zero_array_orig1, content_array_orig1, trailing_mixed_array_orig1=[],[],[]
    leading_RlE,content_RlE,tailing_RlE=[],[],[]
    leading_zstd, content_zstd, tailing_zstd = [], [], []
    leading_zstd_22, content_zstd_22, tailing_zstd_22 = [], [], []
    leading_zstd_R, content_zstd_R, tailing_zstd_R = [], [], []
    leading_zstd_22_R, content_zstd_22_R, tailing_zstd_22_R = [], [], []
    leading_gzip_R, content_gzip_R, tailing_gzip_R = [], [], []
    leading_gzip, content_gzip, tailing_gzip = [], [], []

    for i in tune_decomp:
        print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
        print("Tune Decomp: ", i)

        # Adjust bounds based on tuning step
        bnd1 = bnd1 + i
        bnd2 = bnd2 - i

        if bnd1 % n != 0 or bnd2 % n != 0:
            continue
        else:
            print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
            print("Tune Decomp: ", i)

            # Decompose the array into three parts
            leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2,
                                                                                                           image_ts)
            tl_m, tl_n = leading_zero_array_orig.shape
            tc_m, tc_n = content_array_orig.shape
            tt_m, tt_n = trailing_mixed_array_orig.shape

            # Compress leading zero array
            ts_m_l, ts_n_l = leading_zero_array_orig.shape
            if ts_n_l != 0:
                dict_leading, root_leading, tree_size_leading, encoded_text_leading, = pattern_based_compressor(
                    leading_zero_array_orig, m, bnd1, ts_m_l, ts_n_l)
                leading_entropy = calculate_entropy(leading_zero_array_orig)
                leading_RlE1=rle_encode(leading_zero_array_orig)
                #RLE_size_L1=len(leading_RlE1)*(bnd1+32)+32
                RLE_size_L=measure_total_compressed_size_RLE(leading_RlE1)
                leadinf_float=bits_to_float32(leading_zero_array_orig)
                comp_zstd_leading,leading_zstd_ratio=compress_with_zstd(leadinf_float, level=3)
                comp_zstd_leading_22, leading_zstd_ratio_22 = compress_with_zstd(leadinf_float, level=22)
                comp_gzip_leading, leading_gzip_ratio = compress_with_gzip(leadinf_float)

            else:
                dict_leading, root_leading, tree_size_leading, encoded_text_leading = {}, None, 0, ''
                leading_entropy = 0
                RLE_size_L=0
                comp_zstd_leading, leading_zstd_ratio=0,0
                comp_zstd_leading_22, leading_zstd_ratio_22 = 0, 0
                comp_gzip_leading, leading_gzip_ratio=0,0

            # Compress content array
            ts_m_c, ts_n_c = content_array_orig.shape
            if ts_n_c != 0:
                dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(
                    content_array_orig, m, n, ts_m_c, ts_n_c)
                contents_entropy = calculate_entropy(content_array_orig)
                content_RlE1=rle_encode(content_array_orig)
                #RLE_size_c=len(content_RlE1)*(bnd2-bnd1+32)+32
                RLE_size_c=measure_total_compressed_size_RLE(content_RlE1)
                content_float = bits_to_float32(content_array_orig)
                comp_zstd_content, content_zstd_ratio = compress_with_zstd(content_float, level=3)
                comp_zstd_content_22, content_zstd_ratio_22 = compress_with_zstd(content_float, level=22)
                comp_gzip_content, content_gzip_ratio = compress_with_gzip(content_float)

            else:
                dict_content, root_content, tree_size_content, encoded_text_content = {}, None, 0, ''
                contents_entropy = 0
                RLE_size_c=0
                comp_zstd_content, content_zstd_ratio=0,0
                comp_zstd_content_22, content_zstd_ratio_22=0,0
                comp_gzip_content, content_gzip_ratio =0,0


                # Compress trailing mixed array
            ts_m_t, ts_n_t = trailing_mixed_array_orig.shape
            if ts_n_t != 0:
                dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(
                    trailing_mixed_array_orig, m,n, ts_m_t, ts_n_t)
                trailing_entropy = calculate_entropy(trailing_mixed_array_orig)
                tailing_RlE1=rle_encode(trailing_mixed_array_orig)
                #RLE_size_t=len(tailing_RlE1)*(32-bnd2+32)+32
                RLE_size_t=measure_total_compressed_size_RLE(tailing_RlE1)
                trailing_float = bits_to_float32(trailing_mixed_array_orig)
                comp_zstd_trailing, trailing_zstd_ratio = compress_with_zstd(trailing_float, level=3)
                comp_zstd_trailing_22, trailing_zstd_ratio_22 = compress_with_zstd(trailing_float, level=22)
                comp_gzip_trailing, trailing_gzip_ratio = compress_with_gzip(trailing_float)



            else:
                dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = {}, None, 0, ''
                trailing_entropy = 0
                RLE_size_t=0
                comp_zstd_trailing, trailing_zstd_ratio=0,0
                comp_zstd_trailing_22, trailing_zstd_ratio_22=0,0
                comp_gzip_trailing, trailing_gzip_ratio=0,0


            # Store compressed sizes and dictionaries
            leading_RlE.append(RLE_size_L)
            tailing_RlE.append(RLE_size_t)
            content_RlE.append(RLE_size_c)
            leading_zstd.append(comp_zstd_leading)
            content_zstd.append(comp_zstd_content)
            tailing_zstd.append(comp_zstd_trailing)
            leading_zstd_22.append(comp_zstd_leading_22)
            content_zstd_22.append(comp_zstd_content_22)
            tailing_zstd_22.append(comp_zstd_trailing_22)
            leading_zstd_R.append(leading_zstd_ratio)
            content_zstd_R.append(content_zstd_ratio)
            tailing_zstd_R.append(trailing_zstd_ratio)
            leading_zstd_22_R.append(leading_zstd_ratio_22)
            content_zstd_22_R.append(content_zstd_ratio_22)
            tailing_zstd_22_R.append(trailing_zstd_ratio_22)
            lead_comp_size_d.append(dict_leading)
            tail_comp_size_d.append(dict_trailing)
            content_comp_size_d.append(dict_content)
            lead_comp_size.append(encoded_text_leading)
            tail_comp_size.append(encoded_text_trailing)
            content_comp_size.append(encoded_text_content)
            lead_entropy.append(leading_entropy)
            tail_entropy.append(contents_entropy)
            content_entropy.append(trailing_entropy)
            lead_shape_m.append(tl_m)
            tail_shap_m.append(tt_m)
            content_shap_m.append(tc_m)
            lead_shape_n.append(tl_n)
            tail_shap_n.append(tt_n)
            content_shap_n.append(tc_n)
            leading_zero_array_orig1.append(leading_zero_array_orig)
            content_array_orig1.append(content_array_orig)
            trailing_mixed_array_orig1.append(trailing_mixed_array_orig)
            leading_gzip_R.append(leading_gzip_ratio)
            content_gzip_R.append(content_gzip_ratio)
            tailing_gzip_R.append(trailing_gzip_ratio)
            leading_gzip.append(comp_gzip_leading)
            content_gzip.append(comp_gzip_content)
            tailing_gzip.append(comp_gzip_trailing)

    return (lead_comp_size, tail_comp_size, content_comp_size, lead_comp_size_d, tail_comp_size_d, content_comp_size_d,lead_entropy, tail_entropy, content_entropy,lead_shape_m,
            tail_shap_m, content_shap_m,lead_shape_n, tail_shap_n, content_shap_n,leading_zero_array_orig1,
            content_array_orig1,trailing_mixed_array_orig1,leading_RlE,content_RlE,tailing_RlE,leading_zstd,  content_zstd,tailing_zstd,
            leading_zstd_22,content_zstd_22, tailing_zstd_22,leading_zstd_R, content_zstd_R,tailing_zstd_R,
            leading_zstd_22_R,content_zstd_22_R, tailing_zstd_22_R,leading_gzip_R, content_gzip_R,tailing_gzip_R,leading_gzip, content_gzip,tailing_gzip)


def bits_to_float32(bit_array):
    """Convert a 1D array of bits to an array of float32 values.
    If the bit array length is not divisible by 32, pad with 0s at the end."""

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
    return compressed, comp_ratio
#########################################################################
def compress_with_gzip(array):
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


####################################################
def huffman_code_array(array):
    # compute the frequency of the values
    frq_dict = compute_repetition(array)
    pattern_count = np.fromiter(frq_dict.values(), dtype=int)
    binary_code = np.array(range(len(pattern_count)), dtype=int)
    dict_code = {}
    for i in range(len(pattern_count)):
        dict_code[str(binary_code[i])] = pattern_count[i]
    root = create_huffman_tree(dict_code)
    # Create Huffman codes dictionary
    codes = {}
    create_huffman_codes(root, "", codes)
   # plot_historgam(codes, axs[1, 0], True, "Patterns")
    # print(codes)
    estimated_size = 0
    for key in codes:
        estimated_size += len(codes[key]) * pattern_count[int(key)]
    dict_size = 0
    # for key and value in codes
    for key, value in codes.items():
        dict_size += len(value)
    return estimated_size, estimated_size + dict_size
def compute_repetition(array):
    unique, counts = np.unique(array, return_counts=True)
    # get the top 10 values in counts
    max_top_10 = np.argsort(counts)[-50:]
    #print("Top 10 values: ", unique[max_top_10], counts[max_top_10])
    return dict(zip(unique, counts))
#############################################################3
PLOTING_DISABLE = False
# make a plot with 4x2 subplots
#fig, axs = plt.subplots(5, 2, figsize=(20, 20))
# increase the distance of two row subplots
#plt.subplots_adjust(hspace=1)


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


def are_equal(val1, val2):
    # Check if two values are equal, treating NaN as equal to NaN
    if np.isnan(val1) and np.isnan(val2):
        return True
    return val1 == val2



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


def ieee754_to_float32(binary_array):
    """Convert a binary array (32 bits) to its corresponding float32 value."""
    binary_str = ''.join(str(int(bit)) for bit in binary_array)
    as_int = int(binary_str, 2)
    return np.float32(np.uint32(as_int).view(np.float32))


def decompress_final(final_decoded_data, metadata1):
    # Both final_decoded_data and metadata1 are in IEEE 754 binary format (0s and 1s)

    # Step 1: Calculate the final size after decompression
    final_size = len(final_decoded_data)

    for i in range(0, len(metadata1), 3):
        # Extract the count value from the binary metadata
        count = int(ieee754_to_float32(metadata1[i + 2]))  # Convert IEEE 754 binary to int32 for count
        final_size += count  # Add the count to the final size

    # Step 2: Create a new array with the final size, initialized to zeros (in binary format)
    reconstructed_data = np.zeros((int(final_size), 32))  # 32 bits per float, so shape is (final_size, 32)

    # Keep track of the current position in the reconstructed_data
    current_position = 0

    # Process metadata1 to perform insertions
    original_position = 0  # Track the position in the original final_decoded_data
    prvious_count=0

    for i in range(0, len(metadata1), 3):
        # Convert start_index and count to integers from their binary representations
        start_index = int(ieee754_to_float32(metadata1[i]))  # Convert IEEE 754 binary to int32 for start_index
        value = metadata1[i + 1]  # Value remains in IEEE 754 binary format (32-bit binary array)
        count = int(ieee754_to_float32(metadata1[i + 2]))  # Convert IEEE 754 binary to int32 for count

        # Copy data from final_decoded_data up to the start_index
        reconstructed_data[current_position:current_position + (start_index - original_position - prvious_count)] = final_decoded_data[original_position:start_index - prvious_count]

        current_position += (start_index - original_position - prvious_count)

        # Insert the value 'count' times at the correct position
        for _ in range(count):
            reconstructed_data[current_position] = value
            current_position += 1

        # Update the original position
        original_position = start_index
        prvious_count=count

    # Copy any remaining data from final_decoded_data after the last insertion
    reconstructed_data[current_position:] = final_decoded_data[original_position - prvious_count:]

    return reconstructed_data
def find_mismatch(reconstructed_data, final_decoded_data):
    # Ensure both arrays are the same length before comparing
    if len(reconstructed_data) != len(final_decoded_data):
        print("Arrays are of different lengths!")
        return

    # Find mismatches
    mismatches = np.where(reconstructed_data != final_decoded_data)[0]

    if len(mismatches) == 0:
        print("No mismatches found. The arrays are identical.")
    else:
        print(f"Found {len(mismatches)} mismatches at indices: {mismatches}")
        # Print the mismatched values with their indices
        for idx in mismatches:
            print(f"Index {idx}: Reconstructed = {reconstructed_data[idx]}, Original = {final_decoded_data[idx]}")
def calculate_exact_metadata_size(metadata):
    total_bits = 0

    for entry in metadata:
        start_index = entry['start_index']
        value = entry['value']
        count = entry['count']

        # Calculate the number of bits required for each field
        start_index_bits = math.ceil(math.log2(start_index +1))
        value_bits =  math.ceil(math.log2(value +1))
        count_bits = math.ceil(math.log2(count + 1))  # Number of bits for the count

        # Total bits for this metadata entry
        total_bits += start_index_bits + value_bits + count_bits

    return total_bits
def run_and_collect_data(dataset_path):
    dataset_path = "/home/jamalids/Documents/2D/data1/test_low/"
    #dataset_path ="/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"
    #datasets = [dataset_path]
    datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if
                f.endswith('.tsv')]
    results = []
    for dataset_path in datasets:
        result_row = {}

        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print("datasetname##################################",dataset_name)
        group = ts_data1.drop(ts_data1.columns[0], axis=1)
        group=group.iloc[0:4000000,:]
        group = group.T
        #group = group.iloc[:, 0:3000000]
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

        # Huffman compression
        est_size, Non_uniform_1x4_1 = huffman_code_array(non_consecutive_array)
        inverse_cw_dict, root, tree_size, encoded_text = pattern_based_compressor(bool_array3, m1, n1, ts_m1, ts_n)
        compressed_size_w, encoded_size_w, dic_size_bits_w = measure_total_compressed_size(encoded_text,
                                                                                           inverse_cw_dict)
        Non_uniform_1x4 = bool_array_size_bits / (compressed_size_w + size_metadata)

        pattern_size_list = [2]
        n_list = [8]
        for m in pattern_size_list:
            for n in n_list:
                print("m", m, "n", n)
                #group1 = group2
               # new_array_size = group1.shape[0] - group1.shape[0] % m
               # group1 = group1[:new_array_size]
               # bool_array2 = float_to_ieee754(group1)
                group = non_consecutive_array


                # Reshape group based on `m`
                new_array_size = group.shape[0] - group.shape[0] % m
                group = group[:new_array_size]
                ts_m = group.shape[0]

                bool_array = float_to_ieee754(group)
                entropy_all = calculate_entropy(bool_array)
                print("entropy_all", entropy_all)

                # Compress the data
                # inverse_cw_dict, root, tree_size, encoded_text = pattern_based_compressor(bool_array, m, n, ts_m, ts_n)

                # Decompress the data
                # Decodedata = pattern_based_decompressor(inverse_cw_dict, encoded_text, m, n, ts_m, ts_n)
                # tot_compressed_size, tot_encoded_size, tot_dic_size_bits = measure_total_compressed_size(
                #    encoded_text, inverse_cw_dict)
                # verify_flag_data = np.array_equal(bool_array, Decodedata)

                # Decomposition-based compression
                l_z_array, t_z_array = compute_leading_tailing_zeros(bool_array)

                (encoded_text_leading, encoded_text_trailing, encoded_text_content, dict_leading, dict_trailing,
                 dict_content, lead_entropy, tail_entropy, content_entropy, lead_shape_m, tail_shap_m, content_shap_m,
                 lead_shape_n, tail_shap_n, content_shap_n, leading_zero_array_orig, content_array_orig,
                 trailing_mixed_array_orig, leading_RlE, content_RlE, tailing_RlE, leading_zstd, content_zstd,
                 tailing_zstd,
                 leading_zstd_22, content_zstd_22, tailing_zstd_22, leading_zstd_R, content_zstd_R, tailing_zstd_R,
                 leading_zstd_22_R, content_zstd_22_R, tailing_zstd_22_R,leading_gzip_R, content_gzip_R,tailing_gzip_R,
                 leading_gzip, content_gzip,tailing_gzip)  = decomposition_based_compression(bool_array,l_z_array,t_z_array,m, n)

                # Store results dynamically
                result_row = {"M": m, "N": n, "Original Size (bits)": bool_array_size_bits}
                total_encoded_b = {}
                encoded_b = {}
                entropy_b = {}
                R_encoded_b = {}
                zstd_encoded_b = {}
                zstd_encoded_b_22 = {}
                zstd_encoded_b_R = {}
                zstd_encoded_b_R_22 = {}
                gzip_encoded_b_R = {}
                gzip_encoded_b = {}

                # Process leading part
                for idx, (encoded_array, dictionary, lead_entropy1, lead_shap_m1,lead_shape_n1) in enumerate(
                        zip(encoded_text_leading, dict_leading, lead_entropy, lead_shape_m,lead_shape_n), start=1):
                    leading_compressed_size, l_encoded_size, l_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{idx}_leading_compressed_size"] = leading_compressed_size
                    result_row[f"b{idx}_leading_encoded_size"] = l_encoded_size
                    result_row[f"b{idx}_leading_dic_size_bits"] = l_dic_size_bits
                    result_row[f"b{idx}_leading_entropy"] = lead_entropy1
                    result_row[f"b{idx}_leading_Wieght_size"] = (lead_shap_m1* lead_shape_n1)/bool_array_size_bits
                    encoded_b[idx] = encoded_b.get(idx, 0) + l_encoded_size
                    total_encoded_b[idx] = total_encoded_b.get(idx, 0) + leading_compressed_size
                    entropy_b[idx] = entropy_b.get(idx, 0) + lead_entropy1

                # Process content part
                for idx, (encoded_array, dictionary, content_entropy1,content_shap_m1,content_shap_n1) in enumerate(
                        zip(encoded_text_content, dict_content, content_entropy,content_shap_m,content_shap_n), start=1):
                    content_compressed_size, c_encoded_size, c_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{idx}_content_compressed_size"] = content_compressed_size
                    result_row[f"b{idx}_content_encoded_size"] = c_encoded_size
                    result_row[f"b{idx}_content_dic_size_bits"] = c_dic_size_bits
                    result_row[f"b{idx}_content_entropy"] = content_entropy1
                    result_row[f"b{idx}_content_Wieght_size"] = (content_shap_m1 * content_shap_n1) / bool_array_size_bits
                    encoded_b[idx] = encoded_b.get(idx, 0) + c_encoded_size
                    total_encoded_b[idx] = total_encoded_b.get(idx, 0) + content_compressed_size
                    # R_encoded_b[idx] = R_encoded_b.get(idx, 0) + content_compressed_size
                    entropy_b[idx] = entropy_b.get(idx, 0) + content_entropy1

                # Process trailing part
                for idx, (encoded_array, dictionary, tail_entropy1,tail_shap_m1,tail_shap_n1) in enumerate(
                        zip(encoded_text_trailing, dict_trailing, tail_entropy,tail_shap_m,tail_shap_n), start=1):
                    trailing_compressed_size, t_encoded_size, t_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{idx}_trailing_compressed_size"] = trailing_compressed_size
                    result_row[f"b{idx}_trailing_encoded_size"] = t_encoded_size
                    result_row[f"b{idx}_trailing_dic_size_bits"] = t_dic_size_bits
                    result_row[f"b{idx}_tailing_entropy"] = tail_entropy1
                    result_row[f"b{idx}_tailingt_Wieght_size"] = (tail_shap_m1 *tail_shap_n1) / bool_array_size_bits
                    encoded_b[idx] = encoded_b.get(idx, 0) + t_encoded_size
                    total_encoded_b[idx] = total_encoded_b.get(idx, 0) + trailing_compressed_size
                    # R_encoded_b[idx] = R_encoded_b.get(idx, 0) + trailing_compressed_size
                    entropy_b[idx] = entropy_b.get(idx, 0) + tail_entropy1

                for idx, leading_RlE1 in enumerate(leading_RlE, start=1):
                    result_row[f"b{idx}_RLe_leading_size"] = leading_RlE1
                    R_encoded_b[idx] = R_encoded_b.get(idx, 0) + leading_RlE1

                for idx, content_RlE1 in enumerate(content_RlE, start=1):
                    result_row[f"b{idx}_RLe_content_size"] = content_RlE1
                    R_encoded_b[idx] = R_encoded_b.get(idx, 0) + content_RlE1

                for idx, tailing_RlE1 in enumerate(tailing_RlE, start=1):
                    result_row[f"b{idx}_RLe_tailing_size"] = tailing_RlE1
                    R_encoded_b[idx] = R_encoded_b.get(idx, 0) + tailing_RlE1
                #################zstd##########################################
                for idx, leading_zstd1 in enumerate(leading_zstd, start=1):
                    if leading_zstd1==0:
                        result_row[f"b{idx}_zstd_leading_size"]=0
                    else:
                        result_row[f"b{idx}_zstd_leading_size"] = len(leading_zstd1) * 8
                        zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + len(leading_zstd1) * 8

                for idx, content_zstd1 in enumerate(content_zstd, start=1):
                    if content_zstd1==0:
                        result_row[f"b{idx}_zstd_content_size"]=0
                    else:
                        result_row[f"b{idx}_zstd_content_size"] = len(content_zstd1) * 8
                        zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + len(content_zstd1) * 8


                for idx, tailing_zstd1 in enumerate(tailing_zstd, start=1):
                    if tailing_zstd1==0:
                        result_row[f"b{idx}_zstd_tailing_size"] =0
                    else:
                        result_row[f"b{idx}_zstd_tailing_size"] = len(tailing_zstd1) * 8
                        zstd_encoded_b[idx] = zstd_encoded_b.get(idx, 0) + len(tailing_zstd1) * 8



                for idx, leading_zstd1 in enumerate(leading_zstd_22, start=1):
                    if leading_zstd1==0:
                        result_row[f"b{idx}_zstd22_leading_size"] =0
                    else:
                        result_row[f"b{idx}_zstd22_leading_size"] = len(leading_zstd1) * 8
                        zstd_encoded_b_22[idx] = zstd_encoded_b_22.get(idx, 0) + len(leading_zstd1) * 8

                for idx, content_zstd1 in enumerate(content_zstd_22, start=1):
                    if content_zstd1==0:
                        result_row[f"b{idx}_zstd22_content_size_22"] =0
                    else:
                        result_row[f"b{idx}_zstd22_content_size_22"] = len(content_zstd1) * 8
                        zstd_encoded_b_22[idx] = zstd_encoded_b_22.get(idx, 0) + len(content_zstd1) * 8


                for idx, tailing_zstd1 in enumerate(tailing_zstd_22, start=1):
                    if tailing_zstd1==0:
                        result_row[f"b{idx}_zstd22_tailing_size"] =0
                    else:
                        result_row[f"b{idx}_zstd22_tailing_size"] = len(tailing_zstd1) * 8
                        zstd_encoded_b_22[idx] = zstd_encoded_b_22.get(idx, 0) + len(tailing_zstd1) * 8

                ############################################################
                for idx, leading_zstd1 in enumerate(leading_zstd_R, start=1):
                    result_row[f"b{idx}_zstd_leading_comp_ratio"] = leading_zstd1
                    zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + leading_zstd1

                for idx, content_zstd1 in enumerate(content_zstd_R, start=1):
                    result_row[f"b{idx}_zstd_content_comp_ratio"] = content_zstd1
                    zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + content_zstd1

                for idx, tailing_zstd1 in enumerate(tailing_zstd_R, start=1):
                    result_row[f"b{idx}_zstd_tailing_comp_ratio"] = tailing_zstd1
                    zstd_encoded_b_R[idx] = zstd_encoded_b_R.get(idx, 0) + tailing_zstd1

                for idx, leading_zstd1 in enumerate(leading_zstd_22_R, start=1):
                    result_row[f"b{idx}_zstd22_leading_comp_ratio"] = leading_zstd1
                    zstd_encoded_b_R_22[idx] = zstd_encoded_b_R_22.get(idx, 0) + leading_zstd1

                for idx, content_zstd1 in enumerate(content_zstd_22_R, start=1):
                    result_row[f"b{idx}_zstd22_content_comp_ratio_22"] = content_zstd1
                    zstd_encoded_b_R_22[idx] = zstd_encoded_b_R_22.get(idx, 0) + content_zstd1

                for idx, tailing_zstd1 in enumerate(tailing_zstd_22_R, start=1):
                    result_row[f"b{idx}_zstd22_tailing_comp_ratio"] = tailing_zstd1
                    zstd_encoded_b_R_22[idx] = zstd_encoded_b_R_22.get(idx, 0) + tailing_zstd1
                ########################gzip############################
                for idx, leading_gzip1 in enumerate(leading_gzip, start=1):
                    if leading_gzip1==0:
                        result_row[f"b{idx}_gzip_leading_size"]=0
                    else:
                        result_row[f"b{idx}_gzip_leading_size"] = (leading_gzip1) * 8
                        gzip_encoded_b[idx] = gzip_encoded_b.get(idx, 0) + (leading_gzip1) * 8

                for idx, content_gzip1 in enumerate(content_gzip, start=1):
                    if content_gzip1==0:
                        result_row[f"b{idx}_gzip_content_size"]=0
                    else:
                        result_row[f"b{idx}_gzip_content_size"] = (content_gzip1) * 8
                        gzip_encoded_b[idx] = gzip_encoded_b.get(idx, 0) + (content_gzip1) * 8


                for idx, tailing_gzip1 in enumerate(tailing_gzip, start=1):
                    if tailing_gzip1==0:
                        result_row[f"b{idx}_gzip_tailing_size"] =0
                    else:
                        result_row[f"b{idx}_gzip_tailing_size"] = (tailing_gzip1) * 8
                        gzip_encoded_b[idx] = gzip_encoded_b.get(idx, 0) + (tailing_gzip1) * 8

                for idx, leading_gzip1 in enumerate(leading_gzip_R, start=1):
                    result_row[f"b{idx}_gzip_leading_comp_ratio"] = leading_gzip1
                    gzip_encoded_b_R[idx] = gzip_encoded_b_R.get(idx, 0) + leading_gzip1

                for idx, content_gzip1 in enumerate(content_gzip_R, start=1):
                    result_row[f"b{idx}_gzip_content_comp_ratio"] = content_gzip1
                    gzip_encoded_b_R[idx] = gzip_encoded_b_R.get(idx, 0) + content_gzip1

                for idx, tailing_gzip1 in enumerate(tailing_gzip_R, start=1):
                    result_row[f"b{idx}_gzip_tailing_comp_ratio"] = tailing_gzip1
                    gzip_encoded_b_R[idx] = gzip_encoded_b_R.get(idx, 0) + tailing_gzip1
################################################################################################3


                # Calculate compression ratios dynamically for all available `b` components
                for idx in encoded_b:
                    result_row[f"com_ratio_b{idx}"] = bool_array_size_bits / (encoded_b[idx] + size_metadata) if \
                        encoded_b[idx] > 0 else None
                    result_row[f"t_com_ratio_b{idx}"] = bool_array_size_bits / (total_encoded_b[idx] + size_metadata) if \
                        total_encoded_b[idx] > 0 else None
                for idx in R_encoded_b:
                    result_row[f"R_com_ratio_b{idx}"] = bool_array_size_bits / (R_encoded_b[idx] + size_metadata) if \
                        R_encoded_b[idx] > 0 else None

                for idx in zstd_encoded_b_22:
                    result_row[f"zstd_22_com_ratio_b{idx}"] = bool_array_size_bits / (
                                zstd_encoded_b_22[idx] + size_metadata) if \
                        zstd_encoded_b_22[idx] > 0 else None
                for idx in zstd_encoded_b:
                    result_row[f"zstd_com_ratio_b{idx}"] = bool_array_size_bits / (
                                zstd_encoded_b[idx] + size_metadata) if \
                        zstd_encoded_b[idx] > 0 else None
                for idx in gzip_encoded_b:
                    result_row[f"gzip_com_ratio_b{idx}"] = bool_array_size_bits / (
                                gzip_encoded_b[idx] + size_metadata) if \
                        gzip_encoded_b[idx] > 0 else None
                # Store Zstd and Huffman results
                result_row["comp_ratio_zstd_default"] = comp_ratio_zstd_default
                result_row["comp_ratio_l22"] = comp_ratio_l22
                result_row["comp_ratio_gzip"] = comp_ratio_gzip
                result_row["Non_uniform_1x4"] = Non_uniform_1x4
                result_row["Non_uniform_1x4_1"] = bool_array_size_bits / Non_uniform_1x4_1
                result_row["bool_array_size_bits"] = bool_array_size_bits
                result_row["entropy_remainig"] = entropy_all
                result_row["entropy_float"] = entropy_float
                result_row["dataset_name"] = dataset_name
                result_row["verify_flag_final"] = verify_flag_final
                result_row["len(metadata)"] = len(metadata)
                result_row["len(non_consecutive_array)"] = len(non_consecutive_array)
                result_row["dataset_name"] = dataset_name


                results.append(result_row)

        save_results(pd.DataFrame(results), dataset_name)

    return pd.DataFrame(results)


def save_results(df_results, name_dataset):
    # Check which com_ratio columns exist dynamically
    com_ratio_cols = [col for col in df_results.columns if col.startswith("com_ratio_b")]
    if com_ratio_cols:  # Ensure the list is not empty
        df_results["max_com_ratio"] = df_results[com_ratio_cols].max(axis=1)
        Decomposion_pattern = df_results["max_com_ratio"].max()
    else:
        Decomposion_pattern = 0  # Fallback value if no columns exist

    # Similarly handle the entropy columns dynamically
    entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]
    if entropy_cols:  # Ensure the list is not empty
        entropy_full_data = df_results[entropy_cols].max().max()
    else:
        entropy_full_data = 0  # Fallback value if no columns exist

    # Handle t_com_ratio columns
    t_com_ratio_cols = [col for col in df_results.columns if col.startswith("t_com_ratio_b")]
    if t_com_ratio_cols:  # Ensure the list is not empty
        df_results["t-max_com_ratio"] = df_results[t_com_ratio_cols].max(axis=1)
        Decomposion_pattern_with_dict = df_results["t-max_com_ratio"].max()
    else:
        Decomposion_pattern_with_dict = 0  # Fallback value if no columns exist

    # Handle zstd compression ratio columns
    zstd_22_com_ratio_cols = [col for col in df_results.columns if col.startswith("zstd_22_com_ratio_b")]
    if zstd_22_com_ratio_cols:  # Ensure the list is not empty
        df_results["max_Decom+zstd_22_com_ratio"] = df_results[zstd_22_com_ratio_cols].max(axis=1)

    zstd_com_ratio_cols = [col for col in df_results.columns if col.startswith("zstd_com_ratio_b")]
    if zstd_com_ratio_cols:  # Ensure the list is not empty
        df_results["max_Decom+zstd_com_ratio"] = df_results[zstd_com_ratio_cols].max(axis=1)

    # Handle gzip compression ratio columns (New Part)
    gzip_com_ratio_cols = [col for col in df_results.columns if col.startswith("gzip_com_ratio_b")]
    if gzip_com_ratio_cols:  # Ensure the list is not empty
        df_results["max_Decom+gzip_com_ratio"] = df_results[gzip_com_ratio_cols].max(axis=1)

    # Loop through each row and calculate the entropy for the component with the maximum zstd compression ratio
    for idx, row in df_results.iterrows():
        # Find the column that gave the maximum zstd compression ratio for the current row
        max_decom_ratio_col = row[zstd_com_ratio_cols].idxmax()
        max_decom_ratio_value = row[max_decom_ratio_col]
        df_results.at[idx, "max_Decom+zstd_com_ratio"] = max_decom_ratio_value

        # Extract the index of the `b` component with the maximum compression ratio
        b_idx = max_decom_ratio_col.split('_')[-1].replace('b', '')

        # Calculate the sum of entropies for the component that gave the max zstd compression ratio
        leading_entropy = row.get(f"b{b_idx}_leading_entropy", 0) * row.get(f"b{b_idx}_leading_Wieght_size", 0)
        content_entropy = row.get(f"b{b_idx}_content_entropy", 0) * row.get(f"b{b_idx}_content_Wieght_size", 0)
        tailing_entropy = row.get(f"b{b_idx}_tailing_entropy", 0) * row.get(f"b{b_idx}_tailingt_Wieght_size", 0)
        ###########################################################################################################
        # Calculate the sum of entropies for the component that gave the max zstd compression ratio
        leading_entropy_sh = row.get(f"b{b_idx}_leading_entropy", 0)
        content_entropy_sh = row.get(f"b{b_idx}_content_entropy", 0)
        tailing_entropy_sh = row.get(f"b{b_idx}_tailing_entropy", 0)

        # Sum up the entropies for the current row
        sum_entropy = leading_entropy + content_entropy + tailing_entropy
        sum_entropy_sh = leading_entropy_sh + content_entropy_sh + tailing_entropy_sh
        df_results.at[idx, f"sum_entropy_b{b_idx}"] = sum_entropy
        df_results.at[idx, f"sum_entropy_sh_b{b_idx}"] = sum_entropy_sh

        # Find the column that gave the maximum gzip compression ratio for the current row
        max_gzip_ratio_col = row[gzip_com_ratio_cols].idxmax()
        max_gzip_ratio_value = row[max_gzip_ratio_col]
        df_results.at[idx, "max_Decom+gzip_com_ratio"] = max_gzip_ratio_value

        # Extract the index of the `b` component with the maximum gzip compression ratio
        b_idx_gzip = max_gzip_ratio_col.split('_')[-1].replace('b', '')

        # Calculate the sum of entropies for the component that gave the max gzip compression ratio
        leading_entropy_gzip = row.get(f"b{b_idx_gzip}_leading_entropy", 0) * row.get(f"b{b_idx_gzip}_leading_Wieght_size", 0)
        content_entropy_gzip = row.get(f"b{b_idx_gzip}_content_entropy", 0) * row.get(f"b{b_idx_gzip}_content_Wieght_size", 0)
        tailing_entropy_gzip = row.get(f"b{b_idx_gzip}_tailing_entropy", 0) * row.get(f"b{b_idx_gzip}_tailingt_Wieght_size", 0)

        leading_entropy_sh_gzip = row.get(f"b{b_idx_gzip}_leading_entropy", 0)
        content_entropy_sh_gzip = row.get(f"b{b_idx_gzip}_content_entropy", 0)
        tailing_entropy_sh_gzip = row.get(f"b{b_idx_gzip}_tailing_entropy", 0)

        # Sum up the entropies for the current row
        sum_entropy_gzip = leading_entropy_gzip + content_entropy_gzip + tailing_entropy_gzip
        sum_entropy_sh_gzip = leading_entropy_sh_gzip + content_entropy_sh_gzip + tailing_entropy_sh_gzip
        df_results.at[idx, f"sum_entropy_b{b_idx_gzip}_gzip"] = sum_entropy_gzip
        df_results.at[idx, f"sum_entropy_b{b_idx_gzip}_sh_gzip"] = sum_entropy_sh_gzip

        # Optionally print or log the result for the current row
        print(f"Row {idx}: Max Zstd Compression Ratio Component: b{b_idx}, Sum Entropy: {sum_entropy}")
        print(f"Row {idx}: Max Gzip Compression Ratio Component: b{b_idx_gzip}, Sum Entropy (gzip): {sum_entropy_gzip}")

    df_results.to_csv("Decom+zstd+gzip.csv")
    return df_results

def save_results1(df_results, name_dataset):
    # Check which com_ratio columns exist dynamically
    com_ratio_cols = [col for col in df_results.columns if col.startswith("com_ratio_b")]
    if com_ratio_cols:  # Ensure the list is not empty
        df_results["max_com_ratio"] = df_results[com_ratio_cols].max(axis=1)
        Decomposion_pattern = df_results["max_com_ratio"].max()
    else:
        Decomposion_pattern = 0  # Fallback value if no columns exist

    # Similarly handle the entropy columns dynamically
    entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]
    if entropy_cols:  # Ensure the list is not empty
        entropy_full_data = df_results[entropy_cols].max().max()
    else:
        entropy_full_data = 0  # Fallback value if no columns exist

    # Handle t_com_ratio columns
    t_com_ratio_cols = [col for col in df_results.columns if col.startswith("t_com_ratio_b")]
    if t_com_ratio_cols:  # Ensure the list is not empty
        df_results["t-max_com_ratio"] = df_results[t_com_ratio_cols].max(axis=1)
        Decomposion_pattern_with_dict = df_results["t-max_com_ratio"].max()
    else:
        Decomposion_pattern_with_dict = 0  # Fallback value if no columns exist

    # Handle zstd compression ratio columns
    com_ratio_cols = [col for col in df_results.columns if col.startswith("zstd_22_com_ratio_b")]
    if com_ratio_cols:  # Ensure the list is not empty
        df_results["max_Decom+zstd_22_com_ratio"] = df_results[com_ratio_cols].max(axis=1)
    else:
        Decomposion_pattern = 0  # Fallback value if no columns exist

    com_ratio_cols = [col for col in df_results.columns if col.startswith("zstd_com_ratio_b")]
    if com_ratio_cols:  # Ensure the list is not empty
        df_results["max_Decom+zstd_com_ratio"] = df_results[com_ratio_cols].max(axis=1)
    else:
        Decomposion_pattern = 0  # Fallback value if no columns exist



    # Loop through each row and calculate the entropy for the component with the maximum compression ratio
    for idx, row in df_results.iterrows():
        # Find the column that gave the maximum zstd compression ratio for the current row
        max_decom_ratio_col = row[com_ratio_cols].idxmax()
        max_decom_ratio_value = row[max_decom_ratio_col]
        df_results.at[idx, "max_Decom+zstd_com_ratio"] = max_decom_ratio_value

        # Extract the index of the `b` component with the maximum compression ratio
        b_idx = max_decom_ratio_col.split('_')[-1].replace('b', '')

        # Calculate the sum of entropies for the component that gave the max compression ratio

        leading_entropy = row.get(f"b{b_idx}_leading_entropy", 0)* row.get(f"b{b_idx}_leading_Wieght_size",0)
        content_entropy = row.get(f"b{b_idx}_content_entropy", 0) * row.get(f"b{b_idx}_content_Wieght_size",0)
        tailing_entropy = row.get(f"b{b_idx}_tailing_entropy", 0)* row.get(f"b{b_idx}_tailingt_Wieght_size",0)

        # Sum up the entropies for the current row
        sum_entropy = leading_entropy + content_entropy + tailing_entropy
        df_results.at[idx, f"sum_entropy_b{b_idx}"] = sum_entropy

        # Optionally print or log the result for the current row
        print(f"Row {idx}: Max Compression Ratio Component: b{b_idx}, Sum Entropy: {sum_entropy}")
    df_results.to_csv("Decom+zstd.csv")
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