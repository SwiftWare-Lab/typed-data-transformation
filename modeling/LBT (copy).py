
import pandas as pd
import numpy as np
from utils import binary_to_int
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data


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
        trailing_zeros_array[i] = 32 - trailing_zeros
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
def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos,m,n):
    min_lead, max_lead, avg_lead, min_tail, max_tail, avg_tail = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), \
        int(np.mean(leading_zero_pos)), int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))
    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead, "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)
    bnd1 = max_lead if max_lead < 28 else avg_lead  # 28 and 4 are ad hoc number to avoid weird case all zeros
    bnd2 = min_tail if min_tail >= 4 else avg_tail
    print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
    leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2, image_ts)
    ts_m_l, ts_n_l = leading_zero_array_orig.shape
    dict_leading, root_leading, tree_size_leading, encoded_text_leading = pattern_based_compressor(leading_zero_array_orig, m, n, ts_m_l, ts_n_l)
    row, col = content_array_orig.shape
    if row != 0 or col != 0:
        dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(
            content_array_orig, m, n, row, col)
    else:
        dict_content, root_content, tree_size_content, encoded_text_conten = 0, 0, 0, 0

    row, col = trailing_mixed_array_orig.shape
    dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(trailing_mixed_array_orig,m, n, row, col)
    ##########################
    original_sorted_values1 = [value for key, value in dict_leading.items()]
    Decodedata = pattern_based_decompressor(dict_leading, encoded_text_leading, m, n, ts_m_l, ts_n_l)
    verify_flag_data1 = np.array_equal(leading_zero_array_orig, Decodedata)
    print(verify_flag_data1)

    return dict_leading,encoded_text_leading,dict_content,encoded_text_content,dict_trailing,encoded_text_trailing,ts_n_l


def run_and_collect_data(dataset_path):
    results = []
    m, n = 4, 1
    ts_n = 32

    dataset_path="/home/jamalids/Documents/2D/UCRArchive_2018/AllGestureWiimoteX/AllGestureWiimoteX_TEST.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    #ts_data1 = ts_data1.iloc[0:5, 0:15]

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
    for group_id, group in groups:
        group.drop(group.columns[0], axis=1, inplace=True)
        group.fillna(0, inplace=True)
        group = group.astype(np.float32)
        group = group.to_numpy().reshape(-1)
        ts_m = group.shape[0]
        bool_array = float_to_ieee754(group)
        # Compress the data
        inverse_cw_dict,root,tree_size,encoded_text = pattern_based_compressor(bool_array, m, n, ts_m, ts_n)

        # Decompress the data
        original_sorted_values = [value for key, value in inverse_cw_dict.items()]
        Decodedata=pattern_based_decompressor(inverse_cw_dict, encoded_text, m, n, ts_m, ts_n)
        verify_flag_data = np.array_equal(bool_array, Decodedata)
        print(verify_flag_data)
        # pattern based decomposition
        l_z_array, t_z_array = compute_leading_tailing_zeros(bool_array)
        dict_leading,encoded_text_leading,dict_content,encoded_text_content,dict_trailing,encoded_text_trailing,ts_n_l=decomposition_based_compression(bool_array, l_z_array, t_z_array,m,n)





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
   # df_results.to_csv(log_file, index=False, header=True)

