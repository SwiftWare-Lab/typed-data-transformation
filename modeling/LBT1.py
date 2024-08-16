import sys

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
def decomposition_based_compression1(image_ts, leading_zero_pos, tail_zero_pos,m,n):
    min_lead, max_lead, avg_lead, min_tail, max_tail, avg_tail = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), \
        int(np.mean(leading_zero_pos)), int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))
    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead, "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)
    bnd1 = max_lead if max_lead < 28 else avg_lead  # 28 and 4 are ad hoc number to avoid weird case all zeros

    bnd2 = min_tail if min_tail >= 4 else avg_tail
    tune_decomp = [0, 1, 2]
    lead_comp_size, tail_comp_size, content_comp_size = [], [], []
    lead_comp_size_d, tail_comp_size_d, content_comp_size_d= [], [], []
    for i in tune_decomp:
        print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
        print("Tune Decomp: ", i)
        bnd1 = bnd1 + i
        bnd2 = bnd2 - i
        leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2,
                                                                                                       image_ts)
        ts_m_l, ts_n_l = leading_zero_array_orig.shape
        if ts_n_l != 0:
            dict_leading, root_leading, tree_size_leading, encoded_text_leading = pattern_based_compressor(
                leading_zero_array_orig, m, n, ts_m_l, ts_n_l)
        else:
            dict_leading, root_leading, tree_size_leading, encoded_text_leading = {}, None, 0, ''

        # dict_leading, root_leading, tree_size_leading, encoded_text_leading = pattern_based_compressor(leading_zero_array_orig, m, n, ts_m_l, ts_n_l)
        ts_m_c, ts_n_c = content_array_orig.shape
        if ts_n_c != 0:
            dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(
                content_array_orig, m, n, ts_m_c, ts_n_c)
        else:
            dict_content, root_content, tree_size_content, encoded_text_content = {}, None, 0, ''

        # dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(content_array_orig, m, n, ts_m_c, ts_n_c)

        ts_m_t, ts_n_t = trailing_mixed_array_orig.shape
        if ts_n_t != 0:
            dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(
                trailing_mixed_array_orig, m, n, ts_m_t, ts_n_t)
        else:
            dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = {}, None, 0, ''

        lead_comp_size_d.append(dict_leading)
        tail_comp_size_d.append(dict_trailing)
        content_comp_size_d.append(dict_content)
        lead_comp_size.append(encoded_text_leading)
        tail_comp_size.append(encoded_text_trailing)
        content_comp_size.append(encoded_text_content)
        # dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(trailing_mixed_array_orig,m, n, ts_m_t, ts_n_t)
        ##########################
        original_sorted_values1 = [value for key, value in dict_leading.items()]
    #  Decodedata = pattern_based_decompressor(dict_leading, encoded_text_leading, m, n, ts_m_l, ts_n_l)
    #  verify_flag_data1 = np.array_equal(leading_zero_array_orig, Decodedata)
    #  print(verify_flag_data1)

    return lead_comp_size, tail_comp_size, content_comp_size,lead_comp_size_d, tail_comp_size_d, content_comp_size_d


def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos, m, n):
    # Calculate min, max, and avg for leading and tail zeros
    min_lead, max_lead, avg_lead = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), int(
        np.mean(leading_zero_pos))
    min_tail, max_tail, avg_tail = int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))

    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead,
          "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)

    # Set bounds based on ad-hoc conditions
    bnd1 = max_lead if max_lead < 28 else avg_lead
    bnd2 = min_tail if min_tail >= 4 else avg_tail

    # Tune decomposition steps
    tune_decomp = [0, 1, 2]

    # Initialize lists to store compressed sizes and dictionaries
    lead_comp_size, tail_comp_size, content_comp_size = [], [], []
    lead_comp_size_d, tail_comp_size_d, content_comp_size_d = [], [], []

    for i in tune_decomp:
        print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
        print("Tune Decomp: ", i)

        # Adjust bounds based on tuning step
        bnd1 = bnd1 + i
        bnd2 = bnd2 - i

        # Decompose the array into three parts
        leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2,
                                                                                                       image_ts)

        # Compress leading zero array
        ts_m_l, ts_n_l = leading_zero_array_orig.shape
        if ts_n_l != 0:
            dict_leading, root_leading, tree_size_leading, encoded_text_leading = pattern_based_compressor(
                leading_zero_array_orig, m, n, ts_m_l, ts_n_l)
        else:
            dict_leading, root_leading, tree_size_leading, encoded_text_leading = {}, None, 0, ''

        # Compress content array
        ts_m_c, ts_n_c = content_array_orig.shape
        if ts_n_c != 0:
            dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(
                content_array_orig, m, n, ts_m_c, ts_n_c)
        else:
            dict_content, root_content, tree_size_content, encoded_text_content = {}, None, 0, ''

        # Compress trailing mixed array
        ts_m_t, ts_n_t = trailing_mixed_array_orig.shape
        if ts_n_t != 0:
            dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(
                trailing_mixed_array_orig, m, n, ts_m_t, ts_n_t)
        else:
            dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = {}, None, 0, ''

        # Store compressed sizes and dictionaries
        lead_comp_size_d.append(dict_leading)
        tail_comp_size_d.append(dict_trailing)
        content_comp_size_d.append(dict_content)
        lead_comp_size.append(encoded_text_leading)
        tail_comp_size.append(encoded_text_trailing)
        content_comp_size.append(encoded_text_content)

        ##########################
        # You can add verification or any other process here
        # Example: original_sorted_values1 = [value for key, value in dict_leading.items()]

    # Return compressed sizes and dictionaries
    return lead_comp_size, tail_comp_size, content_comp_size, lead_comp_size_d, tail_comp_size_d, content_comp_size_d


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
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    # comp ratio
    print(len(data.tobytes()))
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio
#########################################################################
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


def run_and_collect_data(dataset_path):
    results = []
    m, n = 8, 2
    ts_n = 32
    dataset_path = "/home/jamalids/Documents/2D/UCRArchive_2018/ACSF1/ACSF1_TEST.tsv"

    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

    # Adjust the dataset to be divisible by m
    row, col = ts_data1.shape
    remainder = row % m
    if remainder != 0:
        row = row - remainder
    ts_data1 = ts_data1.iloc[:row, :]

    group = ts_data1.drop(ts_data1.columns[0], axis=1)
    group = group.astype(np.float32).to_numpy().reshape(-1)
    ts_m = group.shape[0]

    # Zstd and Huffman
    zstd_compressed_ts, comp_ratio_zstd_default = compress_with_zstd(group)
    zstd_compressed_ts_l22, comp_ratio_l22 = compress_with_zstd(group, 22)
    est_size, Non_uniform_1x4 = huffman_code_array(group)

    pattern_size_list = [4, 6, 8, 10, 12]
    n_list = [1, 2]

    for m in pattern_size_list:
        for n in n_list:
            bool_array = float_to_ieee754(group)
            bool_array_size_bits = bool_array.nbytes  # Size in bits

            # Compress the data
            inverse_cw_dict, root, tree_size, encoded_text = pattern_based_compressor(bool_array, m, n, ts_m, ts_n)

            # Decompress the data
            Decodedata = pattern_based_decompressor(inverse_cw_dict, encoded_text, m, n, ts_m, ts_n)
            tot_compressed_size, tot_encoded_size, tot_dic_size_bits = measure_total_compressed_size(encoded_text,
                                                                                                     inverse_cw_dict)
            verify_flag_data = np.array_equal(bool_array, Decodedata)

            # Pattern-based decomposition
            l_z_array, t_z_array = compute_leading_tailing_zeros(bool_array)
            encoded_text_leading, encoded_text_trailing, encoded_text_content, dict_leading, dict_trailing, dict_content = decomposition_based_compression(
                bool_array, l_z_array, t_z_array, m, n)

            # Initialize a dictionary to hold the b1, b2, b3, ..., bn values
            result_row = {"M": m, "N": n, "Original Size (bits)": bool_array_size_bits}

            # Initialize variables to calculate total encoded size for b1, b2, etc.
            total_encoded_b1 = total_encoded_b2 = total_encoded_b3 = 0
            encoded_b1 = encoded_b2 = encoded_b3 = 0

            # Process leading part
            leading_idx = 1
            if len(encoded_text_leading) == len(dict_leading):
                for encoded_array, dictionary in zip(encoded_text_leading, dict_leading):
                    leading_compressed_size, l_encoded_size, l_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{leading_idx}_leading_compressed_size"] = leading_compressed_size
                    result_row[f"b{leading_idx}_leading_encoded_size"] = l_encoded_size
                    result_row[f"b{leading_idx}_leading_dic_size_bits"] = l_dic_size_bits

                    # Accumulate encoded size for b1, b2, b3
                    if leading_idx == 1:
                        encoded_b1 += l_encoded_size
                    elif leading_idx == 2:
                        encoded_b2 += l_encoded_size
                    elif leading_idx == 3:
                        encoded_b3 += l_encoded_size

                    if leading_idx == 1:
                        total_encoded_b1 += leading_compressed_size
                    elif leading_idx == 2:
                        total_encoded_b2 += leading_compressed_size
                    elif leading_idx == 3:
                        total_encoded_b3 += leading_compressed_size

                    leading_idx += 1

            # Process content part (Restart idx to b1 for content)
            content_idx = 1
            if len(encoded_text_content) == len(dict_content):
                for encoded_array, dictionary in zip(encoded_text_content, dict_content):
                    content_compressed_size, c_encoded_size, c_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{content_idx}_content_compressed_size"] = content_compressed_size
                    result_row[f"b{content_idx}_content_encoded_size"] = c_encoded_size
                    result_row[f"b{content_idx}_content_dic_size_bits"] = c_dic_size_bits

                    # Accumulate encoded size for b1, b2, b3
                    if content_idx == 1:
                        encoded_b1 += c_encoded_size
                    elif content_idx == 2:
                        encoded_b2 += c_encoded_size
                    elif content_idx == 3:
                        encoded_b3 += c_encoded_size
                    if content_idx == 1:
                        total_encoded_b1 += content_compressed_size
                    elif content_idx == 2:
                        total_encoded_b2 += content_compressed_size
                    elif content_idx == 3:
                        total_encoded_b3 += content_compressed_size

                    content_idx += 1

            # Process trailing part (Restart idx to b1 for trailing)
            trailing_idx = 1
            if len(encoded_text_trailing) == len(dict_trailing):
                for encoded_array, dictionary in zip(encoded_text_trailing, dict_trailing):
                    trailing_compressed_size, t_encoded_size, t_dic_size_bits = measure_total_compressed_size(
                        encoded_array, dictionary)
                    result_row[f"b{trailing_idx}_trailing_compressed_size"] = trailing_compressed_size
                    result_row[f"b{trailing_idx}_trailing_encoded_size"] = t_encoded_size
                    result_row[f"b{trailing_idx}_trailing_dic_size_bits"] = t_dic_size_bits

                    # Accumulate encoded size for b1, b2, b3
                    if trailing_idx == 1:
                        encoded_b1 += t_encoded_size
                    elif trailing_idx == 2:
                        encoded_b2 += t_encoded_size
                    elif trailing_idx == 3:
                        encoded_b3 += t_encoded_size
                    if trailing_idx == 1:
                        total_encoded_b1 += trailing_compressed_size
                    elif trailing_idx == 2:
                        total_encoded_b2 += trailing_compressed_size
                    elif trailing_idx == 3:
                        total_encoded_b3 += trailing_compressed_size


                    trailing_idx += 1

            # Calculate com_ratio_b1, com_ratio_b2, com_ratio_b3
            result_row["com_ratio_b1"] = bool_array_size_bits / encoded_b1 if encoded_b1 > 0 else None
            result_row["com_ratio_b2"] = bool_array_size_bits / encoded_b2 if encoded_b2 > 0 else None
            result_row["com_ratio_b3"] = bool_array_size_bits / encoded_b3 if encoded_b3 > 0 else None
            result_row["t_com_ratio_b1"] = bool_array_size_bits / total_encoded_b1 if total_encoded_b1 > 0 else None
            result_row["t_com_ratio_b2"] = bool_array_size_bits / total_encoded_b2 if total_encoded_b2 > 0 else None
            result_row["t_com_ratio_b3"] = bool_array_size_bits / total_encoded_b3 if total_encoded_b3 > 0 else None

            # Store Zstd and Huffman results
            result_row["comp_ratio_zstd_default"] = comp_ratio_zstd_default
            result_row["comp_ratio_l22"] = comp_ratio_l22

            results.append(result_row)

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
    df_results.to_csv('results11.csv')
   # df_results.to_csv(log_file, index=False, header=True)