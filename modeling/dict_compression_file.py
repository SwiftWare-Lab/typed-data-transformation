import sys
import pandas as pd
import numpy as np
import pickle
import zstandard as zstd
import snappy
from utils import floats_to_bool_arrays,bool_array_to_float321 ,int_to_bool1,float32_to_bool_array1,bool_to_int1, generate_boolean_array, bool_to_int, char_to_bool, int_to_bool, bool_array_to_float32
import argparse
def get_dict(bool_array, m, n,ts_m, ts_n):
    rectangles = {}
    for i in range(0, ts_n, n):
        for j in range(0, ts_m, m):
            rect = bool_array[j:j + m, i:i + n]
            rect_int = bool_to_int(rect)
            # increment the number of times we have seen this rectangle
            rectangles[rect_int] = rectangles.get(rect_int, 0) + 1

    return rectangles

def create_cw_dicts(rectangle_dict):
    cw_dict, inverse_cw_dict = {}, {}
    sorted_items = sorted(rectangle_dict.items())  # Sort the items in ascending order
    bit_length = np.ceil(np.log2(len(rectangle_dict)))
    for i, (key, value) in enumerate(sorted_items):
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
    cctx = zstd.ZstdCompressor(level=0)
    compressed_dict = cctx.compress(actual_dict_byte)
    total_ztd = len(compressed_dict) + comp_data_bytes
    #total = dict_bytes + comp_data_bytes
    total = actual_dict_size + comp_data_bytes
    #total_estimated = estimated_dict_size + comp_data_bytes

    int_array_dict, comp_int_dict,compressed_dict_snappy = convert_dict_to_array(dict, m, n)
    int_array_dict_d, comp_int_dict_d,compressed_dict_snappy_d = convert_values_to_array_delta(dict, m, n)
    # assert len(int_array_dict.tobytes()) == (len(dict)*m*n)//8
    total_zstd_int = len(comp_int_dict) + comp_data_bytes
    total_snappy_int=len(compressed_dict_snappy) + comp_data_bytes
    total_zstd_int_d = len(comp_int_dict_d) + comp_data_bytes
    total_snappy_int_d=len(compressed_dict_snappy_d) + comp_data_bytes
    return total, total_ztd,  total_zstd_int,total_snappy_int ,total_zstd_int_d,total_snappy_int_d


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
   ###Recursively finds the total size of an object in bits
    size = 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

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
    dict_array = np.zeros(len(dict_in) * m * 2, dtype='float32')
    for i, (key, value) in enumerate(dict_in.items()):
        value_bit_array = int_to_bool1(value, m, n)
        value_float_array = bool_array_to_float321(value_bit_array, n)

        if len(value_float_array) < m:
            value_float_array = np.pad(value_float_array, (0, m - len(value_float_array)), 'constant')
        elif len(value_float_array) > m:
            value_float_array = value_float_array[:m]

       # dict_array[(i * m * 2) + m:(i * m * 2) + (2 * m)] = value_float_array
        dict_array[i * m:(i * m) + m] = value_float_array

    cctx = zstd.ZstdCompressor(level=0)
    compressed_dict_zstd = cctx.compress(dict_array.tobytes())
    #print(f"dictionary sizes: {get_total_size(dict_in)}, compressed dictionary size-zstd: {len(compressed_dict_zstd)}")
    compressed_dict_snappy = snappy.compress(dict_array.tobytes())
    return dict_array, compressed_dict_zstd,compressed_dict_snappy

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
    # convert dictionary to array-dict
    int_array_dict_d, comp_int_dict_d, compressed_dict_snappy_d = convert_values_to_array_delta(inverse_cw_dict, m, n)
    return compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len,int_array_dict_d, comp_int_dict_d, compressed_dict_snappy_d


def pattern_based_decompressor(compressed_char_array, comp_int_dict,original_sorted_values,cw_bit_len, m, n):

    dict_array = decompress_dict_array(comp_int_dict)
    reconstructed_dic = reconstruct_dict_from_array(dict_array, m, n, int(cw_bit_len), original_sorted_values)
    uncompressed_data = replace_cw_with_pattern(compressed_char_array, reconstructed_dic, m, n, int(cw_bit_len))
    return uncompressed_data,reconstructed_dic

def reconstruct_dict_from_array(dict_array, m, n, bit_length, original_values_check):
    # Calculate the number of entries based on the size of dict_array and parameters m
    num_entries = len(dict_array) // (m * 2)
    reconstructed_dict = {}
    original_values = [0] * num_entries  # Pre-allocate the list with zeros

    # First loop: convert all segments to original integer values
    for i in range(num_entries):
        # Extract the float32 segment corresponding to the current entry
        value_float_segment = dict_array[i * m:(i * m) + m]

        # Convert the float32 array back to a boolean array
        value_bit_array = float32_to_bool_array1(value_float_segment, m, n)

        # Convert the boolean array back to the original integer value
        original_value = bool_to_int1(value_bit_array)

        # Assign the original integer value to the list at the current index
        original_values[i] = original_value

    # Decode all original values at once to get the full sequence of decoded values
    decoded_values = delta_decode(original_values)

    # Check if the lengths are the same before comparing
    if len(decoded_values) == len(original_values_check):
        verification_result = (decoded_values == original_values_check)
    else:
        verification_result = False
        print(
            f"Error: Mismatched lengths. Decoded values length: {len(decoded_values)}, Original values length: {len(original_values_check)}")

    print("Verification of Decoded Values Match the Original Values:", verification_result)

    # Second loop: map each decoded value to its corresponding binary-coded key in the dictionary
    for i, value in enumerate(decoded_values):
        # Generate binary code with zero-padding according to bit_length
        cw_bits = bin(i)[2:].zfill(bit_length)

        # Store the decoded value in the dictionary with cw_bits as the key
        reconstructed_dict[cw_bits] = value

    return reconstructed_dict

def decompress_dict_array(compressed_dict):
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_dict)
    dict_array = np.frombuffer(decompressed_bytes, dtype='float32')
    return dict_array


def are_dicts_equal(dict1, dict2):
    return dict1 == dict2

def delta_encode(data):
    if not data:
        return []
    # List comprehension to create deltas directly
    deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
    return [data[0]] + deltas  # Prepend the first element to maintain the original structure


def delta_decode(deltas):
    if not deltas:
        return []

    # Initialize the first element
    data = [deltas[0]]

    # Use a list comprehension with cumulative sum approach
    data.extend(data[i] + deltas[i + 1] for i in range(len(deltas) - 1))
    return data

def convert_values_to_array_delta(dict_in, m, n):
    sorted_values = [value for key, value in dict_in.items()]
    deltas = delta_encode(sorted_values)

    dict_array = np.zeros(len(dict_in) * m * 2, dtype='float32')
    for i,  value in enumerate(deltas):
        value_bit_array = int_to_bool1(value, m, n)
        value_float_array = bool_array_to_float321(value_bit_array, n)

        if len(value_float_array) < m:
            value_float_array = np.pad(value_float_array, (0, m - len(value_float_array)), 'constant')
        elif len(value_float_array) > m:
            value_float_array = value_float_array[:m]

        dict_array[i * m:(i * m) + m] = value_float_array

    cctx = zstd.ZstdCompressor(level=0)
    compressed_dict_zstd = cctx.compress(dict_array.tobytes())
    # print(f"dictionary sizes: {get_total_size(dict_in)}, compressed dictionary size-zstd: {len(compressed_dict_zstd)}")
    compressed_dict_snappy = snappy.compress(dict_array.tobytes())
    return dict_array, compressed_dict_zstd, compressed_dict_snappy

def decompress_dict_snappy(compressed_data):
    decompressed_data = snappy.uncompress(compressed_data)
    return np.frombuffer(decompressed_data, dtype='float32')

def run_and_collect_data(dataset_path):
    results = []
    m, n = 10, 32
    ts_n = 32

    #dataset_path="/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

    ts_data1 = ts_data1.iloc[:, 1:]
    # Get the shape of the data
    row, col = ts_data1.shape
    # Calculate the remainder
    remainder = row % m
    # Adjust h to be the largest number divisible by m if it's not already divisible
    if remainder != 0:
        row = row - remainder
    # Select only the rows that are divisible by m
    ts_data1 = ts_data1.iloc[:row, :]
    ts_m = row
    ts_data = ts_data1.T
    ts_data.insert(0, "feature_index", 1)
    # ts_data = ts_data.iloc[:, 0:101]
    groups = ts_data.groupby("feature_index")
    #dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    for group_id, group in groups:
        group = group.drop(columns="feature_index")
        group.fillna(0, inplace=True)
        group=group

        group = group.astype(np.float32)

        bool_array = floats_to_bool_arrays(group)

        print(len(bool_array.tobytes()))

        # Compress the data
        compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len, int_array_dict, comp_int_dict, compressed_dict_snappy = pattern_based_compressor(
            bool_array, m, n, ts_m, ts_n)

        # Decompress the data
        original_sorted_values = [value for key, value in inverse_cw_dict.items()]
        uncompressed_data, reconstructed_dic = pattern_based_decompressor(compressed_char_array, comp_int_dict,
                                                                          original_sorted_values, cw_bit_len, m, n)

        # Verify flags
        verify_flag_data = np.array_equal(bool_array, uncompressed_data)
        verify_flag_dict = are_dicts_equal(reconstructed_dic, inverse_cw_dict)
        print(f"verify_flag_data: {verify_flag_data}")
        print(f"verify_flag_dict: {verify_flag_dict}")

        #  dictionary size if stored at bit level
        estimated_dict_size = get_dict_size_in_bytes(inverse_cw_dict, m, n, cw_bit_len)
        dictionary_size = get_total_size(inverse_cw_dict)

        # Compute the size of the compressed data
        compressed_bool_array = char_to_bool(compressed_char_array)
        pattern_comp, pattern_comp_dict_zstd,  pattern_comp_dict_int_zstd ,pattern_comp_dict_int_snappy, pattern_comp_dict_int_zstd_delta ,pattern_comp_dict_int_snappy_delta= compute_comp_size(compressed_bool_array, inverse_cw_dict, dictionary_size, m, n)

        # Zstd compression of the original float array
        cctx = zstd.ZstdCompressor(level=3)
        compressed_float_array_zstd = cctx.compress(bool_array.tobytes())
        original_size = len(bool_array.tobytes())
        zstd_comp_size = len(compressed_float_array_zstd)
        # snappy compression of the original float array
        compressed_dict_snappy = snappy.compress(bool_array.tobytes())
        snappy_comp_size = len(compressed_dict_snappy)
        #save dictionary
        with open('../num_brain_f64.pkl', 'wb') as pickle_file:
            pickle.dump(inverse_cw_dict, pickle_file)

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
            "snappy_comp_size":snappy_comp_size,
            "pattern_comp_dict_int_zstd":pattern_comp_dict_int_zstd,
            "pattern_comp_dict_int_snappy":pattern_comp_dict_int_snappy,
            "pattern_comp_dict_int_zstd_delta":pattern_comp_dict_int_zstd_delta ,
            "pattern_comp_dict_int_snappy_delta":pattern_comp_dict_int_snappy_delta
        })

    return pd.DataFrame(results)

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
    df_results.to_csv('results.csv')
   # df_results.to_csv(log_file, index=False, header=True)

