import sys
import pandas as pd
import numpy as np
import pickle
import zstandard as zstd
import snappy
from utils import floats_to_bool_arrays,bool_array_to_float321 ,int_to_bool1,float32_to_bool_array1,bool_to_int1, generate_boolean_array, bool_to_int,binary_to_int, char_to_bool,char_to_binary, int_to_bool, bool_array_to_float32
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode
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


def compute_huffman_code(dict_code,original_data_bool):
    # Extract pattern counts from dict_code
    pattern_count = list(dict_code.values())

    # Create Huffman tree from dict_code
    root = create_huffman_tree(dict_code)

    # Create Huffman codes dictionary
    codes = {}
    create_huffman_codes(root, "", codes)
    encode_text=encode(original_data_bool,codes)

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

    return estimated_size, estimated_size + dict_size,codes, root,tree_size

def replace_patterns_with_cw(bool_array, rectangles, m, n):
    ts_m, ts_n = bool_array.shape
    # get the bit length of the code words
    bit_length = int(np.ceil(np.log2(len(rectangles))))
    # make a numpy char array of zeros
    compressed_char_array_loc = np.zeros((int(ts_m/m), int(ts_n/n)*bit_length), dtype='S1')
    for i in range(0, ts_n, n):
        for j in range(0, ts_m, m):
            rect = bool_array[j:j + m, i:i + n]
            rect_int = binary_to_int(rect)
            cw = rectangles[rect_int]
            # if i == 0 and j == 0:
            #     print("comp:\n pattern: ", rect, "dict: ", rect_int, "cw: ", cw)
            #if rect_int in rectangles:
            j_comp = j // m
            i_comp = (i // n) * bit_length
            # convert cw to binary strings of 0 and 1
            cw = np.array(list(cw), dtype='S1')
            compressed_char_array_loc[j_comp:j_comp+1, i_comp:i_comp + bit_length] = cw
    return compressed_char_array_loc

def pattern_based_compressor(original_data_bool, m, n,ts_m, ts_n):
    # for each rectangle 4x8, convert it to an integer and udpate the dictionary
    rectangles = get_dict(original_data_bool, m, n,ts_m, ts_n)
    # create a dictionary of code words
    estimated_size, estimated_all,cw_dict,root,tree_size=compute_huffman_code(rectangles,original_data_bool)
    #cw_dict, inverse_cw_dict, cw_bit_len = create_cw_dicts(rectangles)
    # replace the rectangles with code words
    compressed_char_array = replace_patterns_with_cw(original_data_bool, cw_dict, m, n)
    # convert to bytes
    compressed_bool_array = char_to_binary(compressed_char_array)
    compressed_byte_array = bool_to_int(compressed_bool_array)


    return compressed_byte_array, compressed_char_array,cw_dict,root,tree_size

def run_and_collect_data(dataset_path):
    results = []
    m, n = 1, 32
    ts_n = 32

    dataset_path="/home/jamalids/Documents/2D/UCRArchive_2018/AllGestureWiimoteX/AllGestureWiimoteX_TEST.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    ts_data1 = ts_data1.iloc[0:1, 0:15]

    # Get the shape of the data
    row, col = ts_data1.shape
    # Calculate the remainder
    remainder = row % m
    # Adjust h to be the largest number divisible by m if it's not already divisible
    if remainder != 0:
        row = row - remainder
    # Select only the rows that are divisible by m
    ts_data1 = ts_data1.iloc[:row, :]



    groups = ts_data1.groupby(0)
    print(groups)
    #dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    for group_id, group in groups:
        group.drop(group.columns[0], axis=1, inplace=True)
        group.fillna(0, inplace=True)
        group = group.astype(np.float32)
        group = group.to_numpy().reshape(-1)
        ts_m = group.shape[0]
        bool_array = float_to_ieee754(group)


        #bool_array = floats_to_bool_arrays(group)



        # Compress the data
        compressed_byte_array, compressed_char_array, inverse_cw_dict,root,tree_size = pattern_based_compressor(
            bool_array, m, n, ts_m, ts_n)

        # Decompress the data
        original_sorted_values = [value for key, value in inverse_cw_dict.items()]

       # uncompressed_data, reconstructed_dic = pattern_based_decompressor(compressed_char_array, comp_int_dict,
                                                                         # original_sorted_values, cw_bit_len, m, n)
        root1=create_huffman_tree_from_dict(inverse_cw_dict)
        Decodedata=decode(compressed_char_array,root1)

