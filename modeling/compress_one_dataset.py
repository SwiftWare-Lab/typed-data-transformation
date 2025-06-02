import struct
import sys
import os
import pandas as pd
import numpy as np
import math
import argparse
import zlib
import fpzip
import snappy
import zstandard as zstd
import lz4.frame


def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array


def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')


def float_to_bin1(value, m, n):

    # Convert float value to its IEEE 754 binary representation
    binary_representation = format(struct.unpack('!I', struct.pack('!f', value))[0], '032b')
    # Remove the leading '0b' and zero fill it to the desired length
    return binary_representation.zfill(m * n)


def bin_to_image(b):
    img = []
    #print(b)
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    return img


def generate_patterns_from_data(data,m,n):
    pattern_set = set()
    #print(data,m,n)
    unique_patterns = []
    for i in range(0, len(data), m):
        for j in range(0,len(data[0]) ,n):
            pattern = [data[i + r][j:j + n] for r in range(m)]
            pattern_tuple = tuple(map(tuple, pattern))
            if pattern_tuple not in pattern_set:
                unique_patterns.append(pattern)
                pattern_set.add(pattern_tuple)
    return unique_patterns


def generate_patterns_from_data_all(data, m, n):
    pattern_set = set()
    unique_patterns = []
    for i in range(len(data) - m + 1):
        for j in range(len(data[0]) - n + 1):
            # Extract a sub-pattern of size m x n
            pattern = [data[i + r][j:j + n] for r in range(m)]
            # Ensure the pattern is of size m x n
            if all(len(row) == n for row in pattern):
                # Convert the sub-pattern to a tuple to use in the set
                pattern_tuple = tuple(map(tuple, pattern))
                if pattern_tuple not in pattern_set:
                    unique_patterns.append(pattern)
                    pattern_set.add(pattern_tuple)
    return unique_patterns


def hash_pattern(pattern):
    # Convert the pattern array to a string representation
    return hash(pattern.tostring())


def get_pattern_occurance_non_overlapping(mat, pattern_list,lookup_table):
    #print("mat",mat)
    pattern_occurance = {}
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
    #print(pm,pn)
    matched_binary_representations = []
    # Calculate hash values for patterns
    pattern_hashes = [hash_pattern(pattern) for pattern in pattern_list]
    # Go over each row in the matrix
    for i in range(0, mat.shape[0], pm):
        j = 0
        for j in range(0, mat.shape[1], pn):
            for k, pattern in enumerate(pattern_list):
                pattern_hash = pattern_hashes[k]
                pm, pn = pattern.shape[0], pattern.shape[1]
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if hash_pattern(mat[i:i + pm, j:j + pn]) == pattern_hash:
                        pattern_occurance[pattern_hash] = pattern_occurance.get(pattern_hash, 0) + 1
                        compressed_value = lookup_table[tuple(map(tuple, mat[i:i + pm, j:j + pn]))]
                        matched_binary_representations.append(compressed_value)
                        #j += pn
                        break
                else:
                    # Pad mat with zeros
                    slice = mat[i:i + pm, j:j + pn]
                    slice_mat = np.pad(slice, ((0, pm - slice.shape[0]), (0, pn - slice.shape[1])), 'constant')
                    if hash_pattern(slice_mat) == pattern_hash:
                        pattern_occurance[pattern_hash] = pattern_occurance.get(pattern_hash, 0) + 1
                        compressed_value = lookup_table[tuple(map(tuple, mat[i:i + pm, j:j + pn]))]
                        matched_binary_representations.append(compressed_value)
                        #j += pn
                        break
            j += 1

    # Convert hash codes to pattern occurrences
    pattern_occurance_list = [pattern_occurance.get(pattern_hash, 0) for pattern_hash in pattern_hashes]
    return pattern_occurance_list,matched_binary_representations


def calculate_min_bits(patterns_list):
    # Determine the maximum index of the patterns
    max_index = len(patterns_list) - 1
    min_bits = math.ceil(math.log2(max_index + 1))
    return min_bits


def to_n_bit_binary(index, n):
    # Convert an index to an n-bit binary string
    return f"{index:0{n}b}"


def compress_block_based(mat, m, n,pattern_list,lookup_table):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    pattern_occurance,matched_binary_representations=get_pattern_occurance_non_overlapping(mat, pattern_list,lookup_table )
    #print("matched_binary_representations",matched_binary_representations)
    stats['num_patterns'] = len(pattern_list)
    stats['m'] = m
    stats['n'] = n
    # total occurence
    sum_all_occurrences = sum(pattern_occurance)
    stats['total_occurrences'] = sum_all_occurrences
    # get the size of each pattern in bit
    num_nz_pattern_occured = sum(np.array(pattern_occurance) > 0)
    stats['num_nz_patterns'] = num_nz_pattern_occured
    #print(num_nz_pattern_occured)
    size_per_pattern = np.log2(num_nz_pattern_occured)  # in bits
    stats['size_per_pattern'] = size_per_pattern
    # round up to bit
    size_per_pattern_bit_roundup = np.ceil(np.log2(num_nz_pattern_occured))  # in bits
    stats['size_per_pattern_bit_roundup'] = size_per_pattern_bit_roundup
    # round up to the nearest integer of multiple of 8 (byte)
    size_per_pattern_byte_roundup = np.ceil(size_per_pattern / 8) * 8
    stats['size_per_pattern_byte_roundup'] = size_per_pattern_byte_roundup

    # uniform coding size in bit #
    size_uniform_code = size_per_pattern * sum_all_occurrences
    size_uniform_bit_roundup = size_per_pattern_bit_roundup * sum_all_occurrences
    size_unifrom_byte_roundup = size_per_pattern_byte_roundup * sum_all_occurrences
    stats['size_uniform_code'] = size_uniform_code
    stats['size_uniform_bit_roundup'] = size_uniform_bit_roundup
    stats['size_unifrom_byte_roundup'] = size_unifrom_byte_roundup
    return stats, pattern_occurance,matched_binary_representations,pattern_list,lookup_table


def reverse_lookup_table(lookup_table):
    # Create a reverse lookup table from binary codes back to matrix patterns
    return {v: k for k, v in lookup_table.items()}


def decompress_data(binary_representations, reverse_table, img_shape, m, n, num_cols):
    reconstructed_img = np.zeros(img_shape, dtype=int)
    block_index = 0
    # Place each block in its correct position
    for i in range(0, img_shape[0], m):
        for j in range(0, img_shape[1], n):
            if block_index < len(binary_representations):
                # Get the binary representation of the current block
                binary = binary_representations[block_index]
                # Fetch the pattern using reverse lookup
                pattern = np.array(reverse_table[binary])

                actual_m = min(m, img_shape[0] - i)
                actual_n = min(n, img_shape[1] - j)

                # Place the pattern in the reconstructed
                reconstructed_img[i:i+actual_m, j:j+actual_n] = pattern[:actual_m, :actual_n]

                # Increment block index
                block_index += 1

    return reconstructed_img


def process_and_compress_by_chunks(data, chunk_size, m, n, pattern_list):
    total_rows = data.shape[0]
    compressed_data_all = []
    total_original_size = 0
    total_compressed_size = 0
    min_bits=calculate_min_bits(pattern_list)
    lookup_table = {
    tuple(map(tuple, pattern)):to_n_bit_binary(index, min_bits)
    for index, pattern in enumerate(pattern_list)}

    for start_row in range(0, total_rows, chunk_size):
        #print("start_row",start_row)
        end_row = min(start_row + chunk_size, total_rows)
        # Select the chunk of data
        data_chunk = data[start_row:end_row]
        #print("data_chunk",data_chunk)

        # Convert float data to binary image
        if isinstance(data_chunk, pd.DataFrame):
            data_chunk = data_chunk.to_numpy()
        str_array = float_to_bin_array(data_chunk.flatten())
        img_orig = bin_to_image(str_array)
        img_array = np.array(img_orig)
        total_original_size += img_array.size
        #print("img_array",img_array)
        # Compress the chunk
        stats, pattern_occurance, matched_binary_representations, _, lookup_table = compress_block_based(img_array, m, n,pattern_list,lookup_table)
        #print("matched_binary_representations",matched_binary_representations)
        # Compute compressed size for the chunk
        min_bits = calculate_min_bits(np.array(pattern_list))
        compressed_size_chunk = len(matched_binary_representations) * min_bits
        total_compressed_size += compressed_size_chunk
        compressed_data_all.extend(matched_binary_representations)

    # Calculate compression ratio
    if total_compressed_size > 0:
        compression_ratio = total_original_size / total_compressed_size
    else:
        compression_ratio = 0

    #print("compressed_data_all",compressed_data_all)
    return {
        'compressed_data_all': compressed_data_all,
        'total_original_size': total_original_size,
        'total_compressed_size': total_compressed_size,
        'matched_binary_representations': matched_binary_representations,
        'lookup_table':lookup_table
    }


def calculate_compression_ratios(result, pattern_list, min_bits):
    ideal_ratio = result['total_original_size'] / result['total_compressed_size']
    base_ratio = result['total_original_size'] / (result['total_compressed_size'] + (pattern_list.shape[0] * pattern_list[0].shape[1] + 16))

    lookup_ratio = result['total_original_size'] / (result['total_compressed_size'] + (pattern_list.shape[0] * pattern_list[0].shape[1] + len(pattern_list) * min_bits + 16))

    return ideal_ratio, base_ratio, lookup_ratio
# Function to process a single feature

def process_feature(dataset_name, feature_idx, feature_data3):
    chunk_size = 10
    data= feature_data3

    feature_data1= apply_padding(data, chunk_size)
    #print((feature_data1))
    ##########################
    try:
        # Attempt to reshape using NumPy if the data is already an array
        feature_data1 = apply_padding(data, chunk_size)

        # If feature_data1 is a DataFrame, this will raise AttributeError
        feature_data = feature_data1.reshape(-1)

    except AttributeError:

        if isinstance(feature_data1, pd.DataFrame):

            feature_data = feature_data1.to_numpy().reshape(-1)
        else:
            print(" another type that doesn't support reshape")
            raise
    ###########################
    #feature_data = feature_data1.reshape(-1)

    str_array = float_to_bin_array(feature_data)

    img_orig = bin_to_image(str_array)
    img_array = np.array(img_orig)

    # Generate patterns
    m, n = 10, 16
    patterns_from_data = generate_patterns_from_data(img_array, m, n)
    pattern_list = np.array(patterns_from_data)
    #print("len(pattern_list)",len(pattern_list))
    # Compress the data

    result = process_and_compress_by_chunks(feature_data1, chunk_size, m, n, pattern_list)

    # Calculate compression ratios
    min_bits = calculate_min_bits(pattern_list)
    ratios = calculate_compression_ratios(result, pattern_list, min_bits)

    return dataset_name, feature_idx, ratios,result
########################


def apply_padding(data, chunk_size):

    # Calculate the padding needed to make rows divisible by chunk_size
    rows_needed = chunk_size - (data.shape[0] % chunk_size) if data.shape[0] % chunk_size != 0 else 0
    # Pad along the first dimension only if necessary
    if rows_needed > 0:
        data = np.pad(data, ((0, rows_needed), (0, 0)), mode='constant', constant_values=0)
    return data


##################################################################################################
def read_and_compress_dataset(dataset_path):
    results = []

    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    try:
        ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
        # Split the DataFrame based on the first column
        groups = ts_data.groupby(0)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        for group_id, group in groups:
            group = group.drop(columns=0)
            group.fillna(0, inplace=True)
            group = group.astype(float)

            n_samples, n_timesteps = group.shape

            print(f"Group {group_id}: {n_samples} rows, {n_timesteps} columns")

            result = process_feature(os.path.basename(dataset_path).replace('.tsv', ''), group_id, group)
            #print(result)
            results.append( result)
       # print("results")
       # print(results)
    except Exception as e:
        print(f"Error processing dataset: {e}")

    return results


def read_and_describe_dataset_all(dataset_path):
    results = []
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return
    try:
        ts_data = pd.read_csv(dataset_path, delimiter='\t')

        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        ts_data = ts_data.iloc[:, 1:]  #  drops the first column
        ts_data.fillna(0, inplace=True)
        ts_data = ts_data.astype(float)
        print(f"Number of instances: {len(ts_data)}")
        print(f"Number of features: {len(ts_data.columns)}")
        print(dataset_name)
        print(ts_data)
        n_samples, n_timesteps = ts_data.shape

        feature_data1 = ts_data
        result = process_feature(dataset_name, 1,  feature_data1)

        results.append(result)
        print("\n\n")

    except Exception as e:
        print(f"Failed to read the dataset. Error: {e}")
    return results


#######################################################OTHER TOOLS################################
class GorillaCompressor:
    def __init__(self):
        self.compressed_data = []
        self.prev_value = None
        self.compressed_size_bits = 0

    def compress(self, value):
        if not isinstance(value, float):
            raise ValueError(f"Expected a float, got {type(value)} with value {value}")
        if self.prev_value is None:
            self.compressed_data.append(float_to_bits(value))
            self.compressed_size_bits += 32
        else:
            xor_result = float_to_bits(value) ^ float_to_bits(self.prev_value)
            if xor_result == 0:
                self.compressed_data.append(0)
                self.compressed_size_bits += 1
            else:
                self.compressed_data.append(1)
                self.compressed_data.append(xor_result)
                self.compressed_size_bits += 1 + 32
        self.prev_value = value

    def get_compressed_data(self):
        return self.compressed_data

    def get_compression_ratio(self):
        original_size_bits = len(self.compressed_data) * 32
        return original_size_bits / self.compressed_size_bits


def float_to_bits(f):
    return struct.unpack('>Q', struct.pack('>d', f))[0]


def bits_to_float(b):
    return struct.unpack('>d', struct.pack('>Q', b))[0]


def compress_float_fpzip(input_data, precision=0):
    if not isinstance(input_data, np.ndarray) or not np.issubdtype(input_data.dtype, np.floating):
        raise TypeError("FPZIP compression requires input data to be a numpy floating-point array.")
    compressed_data = fpzip.compress(input_data, precision=precision)
    compressed_size = len(compressed_data)
    return compressed_data, compressed_size


def arg_parser():
    parser = argparse.ArgumentParser(description='Compress one dataset and store the log.')
    parser.add_argument('--dataset', dest='dataset_path', help='Path to the UCR dataset tsv/csv.')
    parser.add_argument('--variant', dest='variant', default="dictionary", help='Variant of the algorithm.')
    parser.add_argument('--pattern', dest='pattern', default="10*16", help='Pattern to match the files.')
    parser.add_argument('--outcsv', dest='log_file', default="./log_out.csv", help='Output directory for the sbatch scripts.')
    parser.add_argument('--nthreads', dest='num_threads', default=1, type=int, help='Number of threads to use.')
    parser.add_argument('--mode', dest='mode',default="signal", help='run mode.')
    return parser


def read_and_describe_dataset_othertools(dataset_path):
    results = []
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file not found

    try:
        ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
        groups = ts_data.groupby(0)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

        for group_id, group in groups:
            group = group.drop(columns=0)
            group.fillna(0, inplace=True)
            group = group.astype(float)
            n_samples, n_timesteps = group.shape
            print(f"Group {group_id}: {n_samples} rows, {n_timesteps} columns")

            data_bytes = group.to_numpy().tobytes()
            original_size = len(data_bytes)

            # Compression methods application
            compressed_data_snappy = snappy.compress(data_bytes)
            compressed_data_fpzip, compressed_size_fpzip = compress_float_fpzip(group.to_numpy())
            compressor = GorillaCompressor()
            for value in np.nditer(group):
                compressor.compress(float(value))
            compressed_data_gorilla = compressor.get_compressed_data()
            compressed_data_zstd = zstd.compress(data_bytes)
            compressed_data_lz4 = lz4.frame.compress(data_bytes)

            # Calculate compression ratios
            compression_ratios = [
                original_size / len(compressed_data_snappy),
                original_size / compressed_size_fpzip,
                original_size / len(compressed_data_gorilla),
                original_size / len(compressed_data_zstd),
                original_size / len(compressed_data_lz4)
            ]

            # Create DataFrame for this group
            df_group = pd.DataFrame([compression_ratios], columns=["Snappy", "FPZIP", "Gorilla", "zstd", "LZ4"])
            df_group['Feature Index'] = group_id
            df_group['Dataset Name'] = dataset_name
            results.append(df_group)

    except Exception as e:
        print(f"Error processing dataset: {e}")

    if not results:
        print("No valid data to process")
        return pd.DataFrame()  # Return an empty DataFrame if no groups were processed

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    dataset_path = args.dataset_path
    comp_variant = args.variant
    pattern = args.pattern
    log_file = args.log_file
    num_threads = args.num_threads
    mode = args.mode

    print(f"Compressing dataset: {dataset_path} with variant: {comp_variant} and pattern: {pattern}..., log file: {log_file}..., num_threads: {num_threads}...,mode:{mode}")

    if mode == 'signal':
        results = read_and_compress_dataset(dataset_path)
        results_other =read_and_describe_dataset_othertools(dataset_path)

    elif mode == 'all':
        results =read_and_describe_dataset_all(dataset_path)
    else:
        print("Invalid mode. Use 'save' or 'read'.")
        sys.exit(1)



    results_df = pd.DataFrame({
      'Dataset Name': [result[0] for result in results],
      'Feature Index': [result[1] for result in results],
      'Ideal Ratio': [result[2][0] for result in results],
      'Base Ratio': [result[2][1] for result in results],
      'Lookup Ratio': [result[2][2] for result in results],
     })


    Combined_df = pd.merge(results_df, results_other, on=['Dataset Name', 'Feature Index'], how='inner')
    print(Combined_df)

    Combined_df.to_csv(log_file, index=False, header=True)


    print(f"Results have been saved to {log_file}")
