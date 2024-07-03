import sys
import pandas as pd
import numpy as np
import pickle
import zstandard as zstd
import struct
from utils import bool_array_to_float321 ,int_to_bool1,float32_to_bool_array1,bool_to_int1, generate_boolean_array, bool_to_int, char_to_bool, int_to_bool, bool_array_to_float32
import matplotlib.pyplot as plt

def get_dict(bool_array, m, n,ts_m, ts_n):
    rectangles = {}
    for i in range(0, ts_n, n):
        for j in range(0, ts_m, m):
            rect = bool_array[j:j + m, i:i + n]
            rect_int = bool_to_int(rect)
            # increment the number of times we have seen this rectangle
            rectangles[rect_int] = rectangles.get(rect_int, 0) + 1
            # if i == 0 and j == 0:
            #     print("dict building:\n pattern: ", rect, "dict: ", rect_int)
    return rectangles


def create_cw_dicts(rectangle_dict):
    cw_dict, inverse_cw_dict = {}, {}
    bit_length = np.ceil(np.log2(len(rectangle_dict)))
    for i, (key, value) in enumerate(rectangle_dict.items()):
        cw_bits = bin(i)[2:].zfill(int(bit_length))
        cw_dict[key] = cw_bits
        inverse_cw_dict[cw_bits] = key
    return cw_dict, inverse_cw_dict, bit_length


def replace_patterns_with_cw(bool_array, rectangles, m, n):
    ts_m, ts_n = bool_array.shape
    # get the bit length of the code words
    bit_length = int(np.ceil(np.log2(len(rectangles))))
    # make a numpy char array of zeros
    compressed_char_array_loc = np.zeros((int(ts_m/m), int(ts_n/n)*bit_length), dtype='S1')
    for i in range(0, ts_n, n):
        for j in range(0, ts_m, m):
            rect = bool_array[j:j + m, i:i + n]
            rect_int = bool_to_int(rect)
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


def replace_cw_with_pattern(compressed_bool_array, dict_in, m, n, cw_bit_len):
    ts_m, ts_n = compressed_bool_array.shape
    # create compressed bool array
    bool_array = np.zeros((ts_m * m, (ts_n//cw_bit_len) * n), dtype='bool')
    for i in range(0, ts_n, cw_bit_len):
        for j in range(0, ts_m, 1):
            cw = compressed_bool_array[j:j + 1, i:i + cw_bit_len]
            # convert cw to binary strings of 0 and 1
            cw_bit_string = cw.flatten().tobytes().decode('utf-8')
            rect = dict_in[cw_bit_string]
            rect_bool = int_to_bool(rect, m, n)
            #if i == 0 and j == 0:
            #    print("decomp:\n pattern: ", rect_bool, "dict: ", rect, "cw: ", cw_bit_string)
            j_uncomp = j * m
            i_uncomp = (i//cw_bit_len) * n
            bool_array[j_uncomp:j_uncomp + m, i_uncomp:i_uncomp + n] = rect_bool
    return bool_array

def compute_comp_size(comp_data, dict, actual_dict_size, m, n):
    comp_data_bytes = len(comp_data.flatten()) // 8
    # convert the dictionary to bytes
    actual_dict_byte = pickle.dumps(dict)
    #dict_bytes = len(actual_dict_byte)
    # # compress rectangles using zstd
    cctx = zstd.ZstdCompressor(level=3)
    compressed_dict = cctx.compress(actual_dict_byte)
    total_ztd = len(compressed_dict) + comp_data_bytes
    #total = dict_bytes + comp_data_bytes
    total = actual_dict_size + comp_data_bytes
    #total_estimated = estimated_dict_size + comp_data_bytes

    int_array_dict, comp_int_dict = convert_dict_to_array(dict, m, n)
    assert len(int_array_dict.tobytes()) == (len(dict)*m*n)//8
    total_zstd_int = len(comp_int_dict) + comp_data_bytes
    return total, total_ztd,  total_zstd_int


def get_dict_size_in_bytes(dict_in, m, n, bit_length):
    # number of entries in the dictionary
    len_dict = len(dict_in)
    # size of each code word
    code_word_size = (m * n) + bit_length
    # size of the dictionary in bytes
    estimated_dict_size = (len_dict * code_word_size) // 8
    return estimated_dict_size

def get_bit_size(bit_key):
    """Calculate the size of a bit key in bits."""
    if isinstance(bit_key, str):
        # Each character in the string is a bit ('0' or '1')
        return len(bit_key)
    elif isinstance(bit_key, int):
        # Measure the size of an integer in bits
        return bit_key.bit_length()
    else:
        raise TypeError("Bit key must be a string or integer.")

def get_integer_size(value):
    """Calculate the size of an integer in bits."""
    if isinstance(value, int):
        return value.bit_length()
    else:
        raise TypeError("Value must be an integer.")

def get_total_size(obj, seen=None):
    """Recursively finds the total size of an object in bits."""
    size = 0  # Start with zero size
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # If we've already seen this object, return 0 to avoid double counting

    # Mark the object as seen
    seen.add(obj_id)

    if isinstance(obj, dict):
        # Add size of dictionary keys and values
        for k, v in obj.items():
            # Get size of key in bits and value in bits
            key_size_bits = get_bit_size(k)
            value_size_bits = get_integer_size(v)
            size += key_size_bits + value_size_bits
    elif hasattr(obj, '__dict__'):
        # If the object has a __dict__ attribute, add its size
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        # If the object is an iterable but not a string/bytes/bytearray, add sizes of its items
        size += sum(get_total_size(i, seen) for i in obj)

    return size
def convert_dict_to_array(dict_in, m, n):

    dict_array = np.zeros(len(dict_in) * m, dtype='float32')
    for i, (key, value) in enumerate(dict_in.items()):
        bit_array = int_to_bool1(value, m, n)
        float_array = bool_array_to_float321(bit_array)
        #print(f"Key1: {key}, Value1: {value}")
        dict_array[i * m:(i + 1) * m] = float_array[:m]
        #print(f"Float1: {dict_array}")
    cctx = zstd.ZstdCompressor(level=0)
    compressed_dict = cctx.compress(dict_array.tobytes())
    print(f"dictionary sizes:{(get_total_size(dict_in))}, compressed dictionary size:{len(compressed_dict)} ")
    return dict_array, compressed_dict



def pattern_based_compressor(original_data_bool, m, n,ts_m, ts_n):
    # for each rectangle 4x8, convert it to an integer and udpate the dictionary
    rectangles = get_dict(original_data_bool, m, n,ts_m, ts_n)
    # create a dictionary of code words
    cw_dict, inverse_cw_dict, cw_bit_len = create_cw_dicts(rectangles)
    # replace the rectangles with code words
    compressed_char_array = replace_patterns_with_cw(original_data_bool, cw_dict, m, n)
    # convert to bytes
    compressed_bool_array = char_to_bool(compressed_char_array)
    compressed_byte_array = bool_to_int(compressed_bool_array)
    return compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len


def pattern_based_decompressor(compressed_byte_array, inverse_cw_dict, cw_bit_len, m, n):
    # byte array to char array
    # uncompress the data
    uncompressed_data = replace_cw_with_pattern(compressed_byte_array, inverse_cw_dict, m, n, int(cw_bit_len))
    return uncompressed_data



def decompress_dict_array(compressed_dict):

    #Decompresses the compressed dictionary data back into an array usig Zstd.

    # Create a decompression context
    dctx = zstd.ZstdDecompressor()
    # Decompress the data
    decompressed_bytes = dctx.decompress(compressed_dict)

    dict_array = np.frombuffer(decompressed_bytes, dtype='float32')
    return dict_array

def reconstruct_dict_from_array(dict_array, m, n):
    num_entries = len(dict_array) // m
    bit_length = int(np.ceil(np.log2(num_entries)))
    reconstructed_dict = {}

    for i in range(num_entries):
        float_segment = dict_array[i*m:(i+1)*m]
        bit_array = float32_to_bool_array1(float_segment)
        int_key =bool_to_int1(bit_array)
        cw_bits = bin(i)[2:].zfill(bit_length)
        reconstructed_dict[cw_bits] = int_key

    # print(f"Key: {cw_bits}, Value: {int_key}")

    return reconstructed_dict

def are_dicts_equal(dict1, dict2):
    return dict1 == dict2


def run_and_collect_data():
    results = []
    sizes = [1,100,200,1000,10000]
    ts_m = 10
    m, n = 10, 32

    for in_size in sizes:
        ts_n = in_size * 64
        bool_array ,f= generate_boolean_array(ts_m, ts_n)
        print(len(bool_array.tobytes()))

        # Compress the data
        compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len = pattern_based_compressor(bool_array, m, n,ts_m, ts_n)

        # Decompress the data
        uncompressed_data = pattern_based_decompressor(compressed_char_array, inverse_cw_dict, cw_bit_len, m, n)

        # Decompress the dictionary
        int_array_dict, comp_int_dict = convert_dict_to_array(inverse_cw_dict, m, n)
        dict_array = decompress_dict_array(comp_int_dict)
        reconstructed_dic = reconstruct_dict_from_array(dict_array, m, n)

        # Verify flags
        verify_flag_data = np.array_equal(bool_array, uncompressed_data)
        verify_flag_dict = are_dicts_equal(reconstructed_dic, inverse_cw_dict)

        #  dictionary size if stored at bit level
        estimated_dict_size = get_dict_size_in_bytes(inverse_cw_dict, m, n, cw_bit_len)
        dictionary_size = get_total_size(inverse_cw_dict)

        # Compute the size of the compressed data
        compressed_bool_array = char_to_bool(compressed_char_array)
        pattern_comp, pattern_comp_dict_zstd,  pattern_comp_dict_int_zstd = compute_comp_size(compressed_bool_array, inverse_cw_dict, dictionary_size, m, n)

        # Zstd compression of the original float array
        cctx = zstd.ZstdCompressor(level=3)
        compressed_float_array_zstd = cctx.compress(bool_array.tobytes())
        original_size = len(bool_array.tobytes())
        zstd_comp_size = len(compressed_float_array_zstd)

        
        results.append({

            "Original Size (bytes)": original_size,
            "Dictionary Size (bytes)": dictionary_size,
            "estimated_dict_size ":estimated_dict_size,
            "Compressed Dictionary Size (bytes)": len(comp_int_dict),
            "Verify Flag Data": verify_flag_data,
            "Verify Flag Dictionary": verify_flag_dict,
            "pattern_comp":pattern_comp,
            "pattern_comp_dict_zstd":pattern_comp_dict_zstd,
            "zstd_comp_size":zstd_comp_size,
            "pattern_comp_dict_int_zstd":pattern_comp_dict_int_zstd
        })

    return pd.DataFrame(results)
df_results = run_and_collect_data()
df_results.to_csv('results.csv')
