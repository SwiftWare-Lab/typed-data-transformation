import sys
import pandas as pd
import numpy as np
import pickle
import zstandard as zstd
import snappy
from utils import floats_to_bool_arrays,bool_array_to_float321 ,int_to_bool1,float32_to_bool_array1,bool_to_int1, generate_boolean_array, bool_to_int, char_to_bool, int_to_bool, bool_array_to_float32
import argparse
from collections import Counter

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

def get_total_size(obj):
    # get the total size of the dictionary
    import sys
    return sys.getsizeof(obj) + sum(sys.getsizeof(v) for v in obj.values())


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
#############################################################################



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
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_dict)
    dict_array = np.frombuffer(decompressed_bytes, dtype='float32')
    return dict_array
def decompress_dict_snappy(compressed_dict):
    decompressed_bytes = snappy.uncompress(compressed_dict)
    dict_array_snappy = np.frombuffer(decompressed_bytes, dtype='float32')
    return dict_array_snappy

def reconstruct_dict_from_array(dict_array, m, n, bit_length):
    num_entries = len(dict_array) // (m * 2)
    reconstructed_dict = {}

    for i in range(num_entries):
        #value_float_segment = dict_array[(i * m * 2) + m:(i * m * 2) + (2 * m)]
        value_float_segment =dict_array[i * m:(i * m) + m]

        value_bit_array = float32_to_bool_array1(value_float_segment, m, n)

        original_value = bool_to_int1(value_bit_array)

        cw_bits = bin(i)[2:].zfill(bit_length)
        reconstructed_dict[cw_bits] = original_value

    return reconstructed_dict


def are_dicts_equal(dict1, dict2):
    return dict1 == dict2
####################
def calculate_entropy(data):
    # Count the frequency of each value in the dataset
    data_count = Counter(data)

    # Calculate the probability of each value
    probabilities = [count / len(data) for count in data_count.values()]

    # Calculate the entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    return entropy
####################################################
def delta_encode(sorted_values):

    if not sorted_values:
        return []
    deltas = [sorted_values[0]]
    for i in range(1, len(sorted_values)):
        deltas.append(sorted_values[i] - sorted_values[i - 1])
    return deltas

def convert_values_to_array_delta(dict_in, m, n):
    # Extract and sort values from the dictionary
    sorted_values = sorted(dict_in.values())

    # Apply delta encoding to the sorted values
    delta_encoded_values = delta_encode(sorted_values)

    # Initialize array to store the converted data
    #dict_array = np.zeros(len(dict_in) * m * 2, dtype='float32')
    dict_array = np.zeros(len(dict_in) * m , dtype='float32')

    # Convert delta-encoded values to float arrays and store in dict_array
    for i, value in enumerate(delta_encoded_values):
        value_bit_array = int_to_bool1(value, m, n)
        value_float_array = bool_array_to_float321(value_bit_array, n)
        if len(value_float_array) < m:
            value_float_array = np.pad(value_float_array, (0, m - len(value_float_array)), 'constant')
        elif len(value_float_array) > m:
            value_float_array = value_float_array[:m]
        # dict_array[(i * m * 2) + m:(i * m * 2) + (2 * m)] = value_float_array
        dict_array[i * m:(i * m) + m]=value_float_array

    cctx = zstd.ZstdCompressor(level=0)
    compressed_dict_zstd = cctx.compress(dict_array.tobytes())
    #print(f"dictionary sizes: {get_total_size(dict_in)}, compressed dictionary size-zstd: {len(compressed_dict_zstd)}")
    compressed_dict_snappy = snappy.compress(dict_array.tobytes())
    return dict_array, compressed_dict_zstd,compressed_dict_snappy

def delta_decode(deltas):
    if not deltas:
        return []

    data = [deltas[0]]
    for i in range(1, len(deltas)):
        data.append(data[-1] + deltas[i])
    return data
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
        entropy = calculate_entropy(group)
        print(f'Entropy: {entropy}')
        group = group.astype(np.float32)

        bool_array = floats_to_bool_arrays(group)

        print(len(bool_array.tobytes()))

        # Compress the data
        compressed_byte_array, compressed_char_array, inverse_cw_dict, cw_bit_len = pattern_based_compressor(bool_array, m, n,ts_m,ts_n)

        # Decompress the data
        uncompressed_data = pattern_based_decompressor(compressed_char_array, inverse_cw_dict, cw_bit_len, m, n)

        # Decompress the dictionary
        int_array_dict, comp_int_dict,compressed_dict_snappy = convert_dict_to_array(inverse_cw_dict, m, n)

        dict_array = decompress_dict_array(comp_int_dict)
        dict_array_snappy=decompress_dict_snappy(compressed_dict_snappy)

        reconstructed_dic = reconstruct_dict_from_array(dict_array_snappy, m, n, int(cw_bit_len))


        # Verify flags
        verify_flag_zstd = np.array_equal(int_array_dict, dict_array)
        verify_flag_snappy = np.array_equal(int_array_dict, dict_array_snappy)
        #print(f"verify_flag_zstd:{verify_flag_zstd}")
        # print(f"verify_flag_snappy:{verify_flag_snappy}")

        verify_flag_data = np.array_equal(bool_array, uncompressed_data)
        verify_flag_dict = are_dicts_equal(reconstructed_dic, inverse_cw_dict)
        # print(f"reconstructed dict {reconstructed_dic}")
        # print(f"inverse_cw_dict {inverse_cw_dict}")

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
        with open('num_brain_f64.pkl', 'wb') as pickle_file:
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
    #df_results.to_csv('results.csv')
    df_results.to_csv(log_file, index=False, header=True)

