
import math
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import binary_to_int
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data,decode_decompose,concat_decompose
import plotly.graph_objects as go


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
        trailing_zeros_array[i] = 22 - trailing_zeros
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

def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos, m, n,fig, axs ):
    # Calculate min, max, and avg for leading and tail zeros
    min_lead, max_lead, avg_lead = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), int(
        np.mean(leading_zero_pos))
    min_tail, max_tail, avg_tail = int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))

    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead,
          "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)

    # Set bounds based on ad-hoc conditions
    bnd1 = max_lead if max_lead < 28 else avg_lead
    bnd2 = min_tail if min_tail >= 4 else avg_tail
    plot_ts(leading_zero_pos, axs[1, 0], "Leading Zeros")
    plot_ts(tail_zero_pos, axs[1, 1], "Trailing Zeros")

    # Tune decomposition steps
    tune_decomp = [0, 1, 2]

    # Initialize lists to store compressed sizes and dictionaries
    lead_comp_size, tail_comp_size, content_comp_size = [], [], []
    lead_comp_size_d, tail_comp_size_d, content_comp_size_d = [], [], []
    lead_entropy, tail_entropy, content_entropy = [], [], []
    lead_shape_m, tail_shap_m, content_shap_m = [], [], []
    lead_shape_n, tail_shap_n, content_shap_n = [], [], []
    leading_zero_array_orig1, content_array_orig1, trailing_mixed_array_orig1=[],[],[]

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
                    leading_zero_array_orig, m, n, ts_m_l, ts_n_l)
                leading_entropy = calculate_entropy(leading_zero_array_orig)
            else:
                dict_leading, root_leading, tree_size_leading, encoded_text_leading = {}, None, 0, ''
                leading_entropy = 0

            # Compress content array
            ts_m_c, ts_n_c = content_array_orig.shape
            if ts_n_c != 0:
                dict_content, root_content, tree_size_content, encoded_text_content = pattern_based_compressor(
                    content_array_orig, m, n, ts_m_c, ts_n_c)
                contents_entropy = calculate_entropy(content_array_orig)
            else:
                dict_content, root_content, tree_size_content, encoded_text_content = {}, None, 0, ''
                contents_entropy = 0

            # Compress trailing mixed array
            ts_m_t, ts_n_t = trailing_mixed_array_orig.shape
            if ts_n_t != 0:
                dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = pattern_based_compressor(
                    trailing_mixed_array_orig, m, n, ts_m_t, ts_n_t)
                trailing_entropy = calculate_entropy(trailing_mixed_array_orig)

            else:
                dict_trailing, root_trailing, tree_size_trailing, encoded_text_trailing = {}, None, 0, ''
                trailing_entropy = 0

            # Store compressed sizes and dictionaries
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

    return (lead_comp_size, tail_comp_size, content_comp_size, lead_comp_size_d, tail_comp_size_d, content_comp_size_d,lead_entropy, tail_entropy, content_entropy,lead_shape_m,
            tail_shap_m, content_shap_m,lead_shape_n, tail_shap_n, content_shap_n,leading_zero_array_orig1,content_array_orig1,trailing_mixed_array_orig1)


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

def plot_historgam(freq_dict, ax=None, log_scale=False, y_label=""):
    if PLOTING_DISABLE:
        return
    ax.bar(freq_dict.keys(), freq_dict.values(), color='b')
    # log scale
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel("Frequency of "+ y_label)
    # set y label




def plot_bar(values, x_labels, y_label, ax=None):
    if PLOTING_DISABLE:
        return
    ax.bar(np.arange(len(values)), values, color='b')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_ylabel(y_label)
    # Set the x-axis label explicitly to the bottom-left corner
    label = ax.set_xlabel('X-axis Label')


def plot_multiple_lines(series, configs, labels, ax=None, y_label="Entropy", xlabel="Configurations", rotation=40):
    if PLOTING_DISABLE:
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series with a label
    for values, label in zip(series, labels):
        ax.plot(configs, values, marker='o', linestyle='-', label=label)

    # Set labels and rotation
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=rotation, ha='right')

    # Add grid and legend for better readability
    ax.grid(True)
    ax.legend()

def plot_ts(ts_array, ax=None, plot_y_axis=""):
    if PLOTING_DISABLE:
        return
    ax.plot(ts_array)
    #ax.set_title(plot_title)
    # set y axis
    ax.set_ylabel(plot_y_axis)
#name_dataset="Beef_test.tsv"

def split_array_on_multiple_consecutive_values(data, threshold_percentage=9):
    total_length = len(data)
    threshold = total_length * (threshold_percentage / 100.0)
    threshold=10
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

import numpy as np
###################################################################
def split_ieee754_binary(binary_array):
    """Split the binary representation into two parts: the first 10 bits and the remaining 22 bits."""
    first_10_bits = binary_array[:, :10]  # Take the first 10 bits
    remaining_22_bits = binary_array[:, 10:]  # Take the remaining 22 bits
    return first_10_bits, remaining_22_bits

import plotly.graph_objects as go




def rle_encode(data):
    """
    Run-Length Encoding (RLE) for an array of integers.

    :param data: numpy array of integers
    :return: a list of tuples representing the value and its count
    """
    encoded = []
    count = 1
    previous = data[0]

    # Loop through the array starting from the second element
    for value in data[1:]:
        if value == previous:
            count += 1
        else:
            encoded.append((previous, count))
            count = 1
            previous = value

    # Append the last run
    encoded.append((previous, count))
    return encoded


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


def convert_to_int_array(binary_arrays):
    """
    Convert a 2D numpy array of binary arrays (each row represents binary digits) to an array of integers.

    :param binary_arrays: numpy array with shape (n, m), where each row represents binary digits
    :return: numpy array of integers
    """
    # Initialize an empty list to store the integer results
    int_array = []

    # Loop through each row of the binary array and convert it to an integer
    for binary_array in binary_arrays:
        int_array.append(binary_to_int(binary_array))

    # Convert the list to a numpy array
    return np.array(int_array)
def calculate_size_in_bits(array, bit_length_func):
    """
    Calculate the size of an array in bits.

    :param array: The array to calculate the size for.
    :param bit_length_func: A function that calculates the number of bits for a given element.
    :return: Total size in bits.
    """
    return sum(bit_length_func(int(value)) for value in array)  # Convert to Python int


def bit_length_for_int(value):
    """
    Calculate the number of bits required to represent an integer.
    """
    return value.bit_length()


def bit_length_for_rle(encoded_data):
    """
    Calculate the total number of bits for RLE encoded data.
    Each tuple in RLE contains a value and a count.
    """
    total_bits = 0
    for value, count in encoded_data:
        total_bits += int(value).bit_length()  # Convert to Python int
        total_bits += int(count).bit_length()  # Convert to Python int
    return total_bits

###################plot
def plot_table(binary_data, float_values, name):
    """
    Plots a table with 32-bit binary data and corresponding float values, expanding it to cover the full page and removing extra white space.

    Args:
        binary_data: List of lists containing binary data (e.g., 0s and 1s).
        float_values: List of float values corresponding to the binary data rows.
        name: The filename to save the image.
    """
    # Column headers for the 32 bits and the float value
    columns = [f"Bit {i + 1}" for i in range(32)] + ["Float Value"]

    # Prepare data for the table: Transpose binary data and then append float values
    binary_transposed = list(zip(*binary_data))  # Transpose binary data
    binary_transposed.append(float_values)  # Add the float values as the last column

    # Create the table with expanded column widths
    fig = go.Figure(data=[go.Table(
        columnwidth=[40] * 32 + [400],  # Adjust the column widths, increasing float column width
        header=dict(values=columns,  # Header for 32 bits + float value
                    fill_color='white',
                    align='center'),
        cells=dict(values=binary_transposed,  # Binary columns + float value column
                   fill_color='white',
                   align='center')
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=1600,  # Adjust width based on table size
        height=250,  # Adjust height based on table size
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Make the background transparent (no white space)
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Increase scale for higher resolution

def plot_table_float(float_values, name):
    """
    Plots a simple table with float values, removing extra white space and color.

    Args:
        float_values: List of float values.
        name: The filename to save the image.
    """
    # Column headers for the float values
    columns = ["Float Value"]

    # Prepare the data for the table
    table_data = [float_values]

    # Create the table without any color formatting
    fig = go.Figure(data=[go.Table(
        columnwidth=[400],  # Adjust the column width for float values
        header=dict(values=columns,  # Only "Float Value" header
                    fill_color='white',  # No color
                    align='center'),
        cells=dict(values=table_data,  # Only float values
                   fill_color='white',  # No color
                   align='center')
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=400,  # Adjust width based on table size
        height=200,  # Adjust height based on table size
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Make the background transparent (no white space)
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Increase scale for higher resolution


def run_and_collect_data(dataset_path):
    results = []
    verify_flag_final=False
    m, n = 1, 16
    ts_n = 32
    #dataset_path = "/home/jamalids/Documents/2D/UCRArchive_2018 (copy)/AllGestureWiimoteX/AllGestureWiimoteX_TEST.tsv"
    #dataset_path ="/home/jamalids/Documents/2D/UCRArchive_2018 (copy)/InsectEPGSmallTrain/InsectEPGSmallTrain_TEST.tsv"
    dataset_path ="/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"
    #dataset_path ="/home/jamalids/Documents/2D/data1/hst_wfc3_ir_f32.tsv"
    datasets = [dataset_path]
    #datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]

    for dataset_path in datasets:

        fig, axs = plt.subplots(5, 2, figsize=(20, 20))  # Adjust the subplot grid and figure size as needed
        plt.subplots_adjust(hspace=1)  # Adjust the space between rows

        results = []
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        ts_data1=ts_data1.drop(ts_data1.columns[0], axis=1)
        ts_data1=ts_data1.T
        ts_data1 = ts_data1.iloc[0:1,75:83]
        zstd_compressed_ts_all=0
        zstd_compressed_ts_22_all=0
        for index, row in ts_data1.iterrows():
            group=row
        #for column in ts_data1.columns:

            # group = ts_data1[column]
            #group = ts_data1.drop(ts_data1.columns[0], axis=1)
            group = group.astype(np.float32).to_numpy().reshape(-1)
            bool_array5= float_to_ieee754(group)
            plot_table(bool_array5, group, "table_with_binary_and_float1.png")
            plot_table_float(group, "table_with_float1.png")


            entropy_float = calculate_entropy_float(group)
            print("entropy_float=", entropy_float)

            # Calculate the total number of elements
            total_elements = len(group)
            positive_values = np.sum(group > 0)
            negative_values = np.sum(group < 0)

            # Calculate the percentages
            positive_percentage = (positive_values / total_elements) * 100
            negative_percentage = (negative_values / total_elements) * 100
            # codes = {"positive_percentage": positive_percentage, "negative_percentage": negative_percentage}
            codes = {"positive_values": positive_values, "negative_values": negative_values}
            plot_historgam(codes, axs[3, 1], True, "_positive_negative")
            codes = {"positive_values": positive_values, "negative_values": negative_values}

            # Plot the time series
            plot_ts(group, axs[0, 0], "Original Values")

            # Zstd and Huffman
            zstd_compressed_ts, comp_ratio_zstd_default = compress_with_zstd(group)
            zstd_compressed_ts_all += len( zstd_compressed_ts)
            zstd_compressed_ts_l22, comp_ratio_l22 = compress_with_zstd(group, 22)
            zstd_compressed_ts_22_all += len(zstd_compressed_ts_l22)

            bool_array = float_to_ieee754(group)

            bool_array_size_bits = bool_array.nbytes  # Size in bits
            ############RLE############################
            first_10_bits, remaining_22_bits = split_ieee754_binary(bool_array)
            array10 = convert_to_int_array(first_10_bits)
            array22 = convert_to_int_array(remaining_22_bits)
            result_array10 = rle_encode(array10)
            ##########################################
            array22 = convert_to_int_array(remaining_22_bits)
            results22 = rle_encode(array22)
            result_all = rle_encode(group)
            # array10_size_in_bits = calculate_size_in_bits(result_array10, bit_length_for_int)
            rle_encoded_array10_size_in_bits = bit_length_for_rle(result_array10)
            rle_encoded_array22_size_in_bits = bit_length_for_rle(results22)
           # rle_encoded_all_size_in_bits = bit_length_for_rle(result_all)
            entropy_10bits = calculate_entropy(first_10_bits)
            entropy_22bits = calculate_entropy(remaining_22_bits)

            #####################################

            # Split array and apply RLE
            non_consecutive_array, metadata = split_array_on_multiple_consecutive_values(array22,
                                                                                         threshold_percentage=1)
            metadata1 = convert_RLE(metadata)
            metadata_array = float_to_ieee754(metadata1)

            # Huffman compression
            est_size, Non_uniform_1x4 = huffman_code_array(group)
            ts_m = bool_array.shape[0]
            inverse_cw_dict_h, root_h, tree_size_h, encoded_text_h = pattern_based_compressor(bool_array, m, n, ts_m, ts_n)
            Non_uniform_1x4, encoded_size_h, dic_size_bits_h = measure_total_compressed_size(
                encoded_text_h, inverse_cw_dict_h)
            #############################################
            bool_array1 = float_to_ieee754(non_consecutive_array)
            ts_m = bool_array1.shape[0]
            inverse_cw_dict1, root, tree_size, encoded_text = pattern_based_compressor(bool_array1, m, n, ts_m, ts_n)
            compressed_size, encoded_size, dic_size_bits = measure_total_compressed_size(
                encoded_text, inverse_cw_dict1)
            max_code_length_32 = max(len(code) for code in inverse_cw_dict1.values())

            ts_n = 22
            ##########################################3
            frq_dict = compute_repetition(group)
            plot_historgam(frq_dict, axs[0, 1], False, "Pattern 1x4")
            size_metadata1 = len(metadata) * 96  # Example size calculation
            size_metadata =size_metadata1+ rle_encoded_array10_size_in_bits
            pattern_size_list = [8]
            n_list = [1]
            for m in pattern_size_list:
                for n in n_list:
                    print("m", m, "n", n)
                    group = remaining_22_bits
                    new_array_size = group.shape[0] - group.shape[0] % m
                    group = group[:new_array_size]
                    bool_array = group
                    # group=non_consecutive_array

                    # Reshape group based on `m`
                    # new_array_size = group.shape[0] - group.shape[0] % m
                    # group = group[:new_array_size]
                    ts_m = group.shape[0]

                    # bool_array = float_to_ieee754(group)
                    # bool_array =remaining_22_bits
                    entropy_all = calculate_entropy(bool_array)
                    print("entropy_all", entropy_all)

                    # Compress the data
                    inverse_cw_dict, root, tree_size, encoded_text = pattern_based_compressor(bool_array, m, n, ts_m,
                                                                                              ts_n)
                    compressed_size22, encoded_size22, dic_size_bits22 = measure_total_compressed_size(
                        encoded_text, inverse_cw_dict)
                    max_code_length_16 = max(len(code) for code in inverse_cw_dict.values())

                    # Decomposition-based compression
                    l_z_array, t_z_array = compute_leading_tailing_zeros(bool_array)

                    (encoded_text_leading, encoded_text_trailing, encoded_text_content, dict_leading, dict_trailing,
                     dict_content, lead_entropy, tail_entropy, content_entropy, lead_shape_m, tail_shap_m,
                     content_shap_m,
                     lead_shape_n, tail_shap_n, content_shap_n, leading_zero_array_orig, content_array_orig,
                     trailing_mixed_array_orig) = decomposition_based_compression(bool_array, l_z_array, t_z_array, m,
                                                                                  n,
                                                                                  fig, axs)

                    # Store results dynamically
                    result_row = {"M": m, "N": n, "Original Size (bits)": bool_array_size_bits}
                    total_encoded_b = {}
                    encoded_b = {}
                    entropy_b = {}

                    # Process leading part
                    for idx, (encoded_array, dictionary, lead_entropy1) in enumerate(
                            zip(encoded_text_leading, dict_leading, lead_entropy), start=1):
                        leading_compressed_size, l_encoded_size, l_dic_size_bits = measure_total_compressed_size(
                            encoded_array, dictionary)
                        result_row[f"b{idx}_leading_compressed_size"] = leading_compressed_size
                        result_row[f"b{idx}_leading_encoded_size"] = l_encoded_size
                        result_row[f"b{idx}_leading_dic_size_bits"] = l_dic_size_bits
                        result_row[f"b{idx}_leading_entropy"] = lead_entropy1
                        encoded_b[idx] = encoded_b.get(idx, 0) + l_encoded_size
                        total_encoded_b[idx] = total_encoded_b.get(idx, 0) + leading_compressed_size
                        entropy_b[idx] = entropy_b.get(idx, 0) + lead_entropy1

                    # Process content part
                    for idx, (encoded_array, dictionary, content_entropy1) in enumerate(
                            zip(encoded_text_content, dict_content, content_entropy), start=1):
                        content_compressed_size, c_encoded_size, c_dic_size_bits = measure_total_compressed_size(
                            encoded_array, dictionary)
                        result_row[f"b{idx}_content_compressed_size"] = content_compressed_size
                        result_row[f"b{idx}_content_encoded_size"] = c_encoded_size
                        result_row[f"b{idx}_content_dic_size_bits"] = c_dic_size_bits
                        result_row[f"b{idx}_content_entropy"] = content_entropy1
                        encoded_b[idx] = encoded_b.get(idx, 0) + c_encoded_size
                        total_encoded_b[idx] = total_encoded_b.get(idx, 0) + content_compressed_size
                        entropy_b[idx] = entropy_b.get(idx, 0) + content_entropy1

                    # Process trailing part
                    for idx, (encoded_array, dictionary, tail_entropy1) in enumerate(
                            zip(encoded_text_trailing, dict_trailing, tail_entropy), start=1):
                        trailing_compressed_size, t_encoded_size, t_dic_size_bits = measure_total_compressed_size(
                            encoded_array, dictionary)
                        result_row[f"b{idx}_trailing_compressed_size"] = trailing_compressed_size
                        result_row[f"b{idx}_trailing_encoded_size"] = t_encoded_size
                        result_row[f"b{idx}_trailing_dic_size_bits"] = t_dic_size_bits
                        result_row[f"b{idx}_tailing_entropy"] = tail_entropy1
                        encoded_b[idx] = encoded_b.get(idx, 0) + t_encoded_size
                        total_encoded_b[idx] = total_encoded_b.get(idx, 0) + trailing_compressed_size
                        entropy_b[idx] = entropy_b.get(idx, 0) + tail_entropy1

                    # Calculate compression ratios dynamically for all available `b` components
                    for idx in encoded_b:
                        result_row[f"com_ratio_b{idx}"] = bool_array_size_bits / (encoded_b[idx] + size_metadata) if \
                            encoded_b[idx] > 0 else None
                        result_row[f"t_com_ratio_b{idx}"] = bool_array_size_bits / (
                                    total_encoded_b[idx] + size_metadata) if \
                            total_encoded_b[idx] > 0 else None

                    # Store Zstd and Huffman results
                    result_row["comp_ratio_zstd_default"] = comp_ratio_zstd_default
                    result_row["zstd_compressed_ts"] = zstd_compressed_ts_all
                    result_row["zstd_compressed_ts_22_all"] =zstd_compressed_ts_22_all
                    result_row["comp_ratio_l22"] = comp_ratio_l22
                    result_row["Non_uniform_1x4"] = Non_uniform_1x4
                    result_row["Non_uniform_1_1x4"] = compressed_size+size_metadata1
                    result_row["Huffman_22_dict_1x4"] = bool_array_size_bits / (compressed_size22 + size_metadata)
                    result_row["Huffman_22_1x4"] = bool_array_size_bits / (encoded_size22 + size_metadata)
                    result_row["bool_array_size_bits"] = bool_array_size_bits
                    result_row["entropy_all"] = entropy_all
                    result_row["dataset_name"] = dataset_name
                    result_row["verify_flag_final"] = verify_flag_final
                    result_row["compressed_size22"] = compressed_size22
                    result_row["encoded_size22"] = encoded_size22
                    result_row["dic_size_bits22"] = dic_size_bits22
                    result_row["rle_encoded_array10_size_in_bits"] = rle_encoded_array10_size_in_bits
                    result_row["rle_encoded_array22_size_in_bits"] = rle_encoded_array22_size_in_bits
                    result_row["comp_ratio_RLE_10_22"] = bool_array_size_bits / (
                                rle_encoded_array10_size_in_bits + rle_encoded_array22_size_in_bits)

                    results.append(result_row)

            save_results(pd.DataFrame(results), dataset_name, fig, axs)

        return pd.DataFrame(results)


def save_results(df_results, name_dataset, fig, axs):
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

    comp_ratio_zstd_default = df_results.get("comp_ratio_zstd_default", pd.Series([0])).max()
    comp_ratio_l22 = df_results.get("comp_ratio_l22", pd.Series([0])).max()
    Non_uniform_1x4 = df_results.get("Non_uniform_1x4", pd.Series([0])).max()
    Non_uniform_1_1x4 = df_results.get("Non_uniform_1_1x4", pd.Series([0])).max()
    bool_array_size_bits = df_results.get("bool_array_size_bits", pd.Series([0])).max()
    Huffman_22_1x4=df_results.get("Huffman_22_1x4", pd.Series([0])).max()
    Huffman_22_1x4_dict=df_results.get("Huffman_22_dict_1x4", pd.Series([0])).max()
    comp_ratio_RLE_10_22=df_results.get("comp_ratio_RLE_10_22", pd.Series([0])).max()

    comp_ratio_array = np.array([
        comp_ratio_zstd_default,
        comp_ratio_l22,
        bool_array_size_bits / Non_uniform_1x4 if Non_uniform_1x4 else 0,
        bool_array_size_bits / Non_uniform_1_1x4 if Non_uniform_1_1x4 else 0,
        Decomposion_pattern,
        Decomposion_pattern_with_dict,
        Huffman_22_1x4,
        Huffman_22_1x4_dict,
        comp_ratio_RLE_10_22
    ])
    plot_bar(comp_ratio_array, ["Zstd Default-3", "Zstd Ultimate-22", "Huffman 1x4","Huffman_RLE_1x4" ,"Decomposion pattern",
                                "Decomposion pattern with dict","Huffman_22_1x4","Huffman_22_1x4_dict","comp_ratio_RLE_10_22"], "Compression Ratio", axs[2, 0])

    # Dynamically find all entropy columns
    entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]

    # Collect the max values of these entropy columns
    entropy_array = np.array([df_results[col].max() for col in entropy_cols])

    # Add the full entropy data if needed
    entropy_array = np.append(entropy_array, entropy_full_data)

    # Create labels dynamically for these entropy columns
    entropy_labels = entropy_cols + ["entropy_full_data"]

    # Plot the entropy array with the corresponding labels
    plot_bar(entropy_array, entropy_labels, "Entropy", axs[2, 1])
    configs = [f"{m} x {n}" for m, n in zip(df_results['M'], df_results['N'])]
    series3 = [df_results[col].tolist() for col in entropy_cols if col in df_results]
    plot_multiple_lines(series3, configs, entropy_cols, ax=axs[3, 0], y_label="Entropy", xlabel="Configurations")
    series1 = [df_results[col].tolist() for col in com_ratio_cols if col in df_results]
    plot_multiple_lines(series1, configs, com_ratio_cols, ax=axs[4, 0], y_label="Com_Ratio", xlabel="Configurations")

    series2 = [df_results[col].tolist() for col in t_com_ratio_cols if col in df_results]
    plot_multiple_lines(series2, configs, t_com_ratio_cols, ax=axs[4, 1], y_label="Com_Ratio_Dic", xlabel="Configurations")

    df_results.to_csv(f"results/{name_dataset}.csv")

    if not PLOTING_DISABLE:
       plt.savefig(f"results/{name_dataset}.png")
       plt.close(fig)  # This is crucial to reset the plot state for the next dataset


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
