import sys

import numpy as np
import pickle
import zstandard as zstd
from utils import generate_boolean_array, bool_to_int, char_to_bool, int_to_bool, bool_array_to_float32

def get_dict(bool_array, m, n):
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


def replace_cw_with_pattern_array_dict(compressed_bool_array, dict_array, m, n, cw_bit_len):
    # TODO : needs to be fixed
    ts_m, ts_n = compressed_bool_array.shape
    # create compressed bool array
    bool_array = np.zeros((ts_m * m, (ts_n//cw_bit_len) * n), dtype='bool')
    for i in range(0, ts_n, cw_bit_len):
        for j in range(0, ts_m, 1):
            cw = compressed_bool_array[j:j + 1, i:i + cw_bit_len]
            # convert cw to binary strings of 0 and 1
            cw_bit_string = cw.flatten().tobytes().decode('utf-8')
            #convert the string to an integer
            cw_int = int(cw_bit_string, 2)
            # TODO : it should
            rect = dict_array[cw_int*(m*n):(cw_int+1)*(m*n)]
            rect_bool = int_to_bool(rect, m, n)
            j_uncomp = j * m
            i_uncomp = (i//cw_bit_len) * n
            bool_array[j_uncomp:j_uncomp + m, i_uncomp:i_uncomp + n] = rect_bool
    return bool_array


def compute_comp_size(comp_data, dict, estimated_dict_size, m, n):
    comp_data_bytes = len(comp_data.flatten()) // 8
    # convert the dictionary to bytes
    actual_dict_byte = pickle.dumps(dict)
    dict_bytes = len(actual_dict_byte)
    # # compress rectangles using zstd
    cctx = zstd.ZstdCompressor(level=3)
    compressed_dict = cctx.compress(actual_dict_byte)
    total_ztd = len(compressed_dict) + comp_data_bytes
    total = dict_bytes + comp_data_bytes
    total_estimated = estimated_dict_size + comp_data_bytes

    int_array_dict, comp_int_dict = convert_dict_to_array(dict, m, n)
    assert len(int_array_dict.tobytes()) == (len(dict)*m*n)//8
    total_zstd_int = len(comp_int_dict) + comp_data_bytes
    return total, total_ztd, total_estimated, total_zstd_int


def get_dict_size_in_bytes(dict_in, m, n, bit_length):
    # number of entries in the dictionary
    len_dict = len(dict_in)
    # size of each code word
    code_word_size = (m * n) + bit_length
    # size of the dictionary in bytes
    estimated_dict_size = (len_dict * code_word_size) // 8
    return estimated_dict_size


def convert_dict_to_array(dict_in, m, n):
    """
    assumes codewords are integers and consecutive
    :param dict_in:
    :param m:
    :param n:
    :return:
    """
    # put values in consecutive memory locations
    dict_array = np.zeros(len(dict_in)*m, dtype='float32')
    for i, (key, value) in enumerate(dict_in.items()):
        bit_array = int_to_bool(value, m, n)
        float_array = bool_array_to_float32(bit_array)
        dict_array[i:i+m] = float_array
    cctx = zstd.ZstdCompressor(level=3)
    compressed_dict = cctx.compress(dict_array.tobytes())
    #print("dict sizes: ", len(dict_array.tobytes()), len(compressed_dict))
    return dict_array, compressed_dict


def pattern_based_compressor(original_data_bool, m, n):
    # for each rectangle 4x8, convert it to an integer and udpate the dictionary
    rectangles = get_dict(original_data_bool, m, n)
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


def pattern_based_decompressor_array_dic(compressed_byte_array, inverse_cw_dict_array, cw_bit_len, m, n):
    #TODO: needs to be fixed
    # uncompress the data
    uncompressed_data = replace_cw_with_pattern_array_dict(compressed_byte_array, inverse_cw_dict_array, m, n, int(cw_bit_len))
    return uncompressed_data


# generate a 2D numpy array of boolean values
in_size = int(sys.argv[1])
ts_m, ts_n = 8, in_size*1024
bool_array, float_array = generate_boolean_array(ts_m, ts_n)
m, n = 8, 32

# compress the data
compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len = pattern_based_compressor(bool_array, m, n)


# decompress the data
uncompressed_data = pattern_based_decompressor(compressed_char_array, inverse_cw_dict, cw_bit_len, m, n)
# second decompression
# TODO: impelment this second compression to make sure the data is correct
#int_array_dict, comp_int_dict = convert_dict_to_array(inverse_cw_dict, m, n)
#uncompressed_data_array_dict = pattern_based_decompressor_array_dic(compressed_char_array, int_array_dict, cw_bit_len, m, n)


# verification
verify_flag = np.array_equal(bool_array, uncompressed_data)
#verify_flag_array_dict = np.array_equal(bool_array, uncompressed_data_array_dict)

# statistics
# estimate dict size if stored at bit level
estimated_dict_size = get_dict_size_in_bytes(inverse_cw_dict, m, n, cw_bit_len)
# compute the size of the compressed data
compressed_bool_array = char_to_bool(compressed_char_array)
pattern_comp, pattern_comp_dict_zstd, pattern_comp_dict_estimated, pattern_comp_dict_int_zstd = compute_comp_size(compressed_bool_array, inverse_cw_dict, estimated_dict_size, m, n)

# zstd compression
cctx = zstd.ZstdCompressor(level=3)
compressed_float_array_zstd = cctx.compress(float_array.tobytes())
origi_float_Size = len(float_array.tobytes())
zstd_comp_size = len(compressed_float_array_zstd)

#print("sizes: ", len(float_array.tobytes()), len(compressed_float_array_zstd))

print(ts_m, ts_n, m, n, verify_flag, origi_float_Size, len(bool_array.flatten()), pattern_comp, pattern_comp_dict_zstd, pattern_comp_dict_estimated, pattern_comp_dict_int_zstd, zstd_comp_size, sep=",")





