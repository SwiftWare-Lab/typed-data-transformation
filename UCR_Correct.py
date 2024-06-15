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

    # Create a DataFrame to show each pattern and its occurrences
    df_pattern_occurrences = pd.DataFrame({
        'Pattern': [str(pattern) for pattern in pattern_list],
        'Occurrences': pattern_occurance_list
    })
    return pattern_occurance_list,matched_binary_representations,df_pattern_occurrences


def calculate_min_bits(patterns_list):
    # Determine the maximum index of the patterns
    #max_index = len(patterns_list) - 1
    max_index = len(patterns_list)
    min_bits = math.ceil(math.log2(max_index + 1))
    return min_bits


def to_n_bit_binary(index, n):
    # Convert an index to an n-bit binary string
    return f"{index:0{n}b}"



def calculate_entropy(occurrences):
    
    if not all(isinstance(x, (int, float)) for x in occurrences):
        raise ValueError("All elements in 'occurrences' must be integers or floats.")

    total_occurrences = sum(occurrences)
    if total_occurrences == 0:
        return 0

    probabilities = [o / total_occurrences for o in occurrences if o > 0]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

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
    min_bits = calculate_min_bits(pattern_list)
    lookup_table = {tuple(map(tuple, pattern)): to_n_bit_binary(index, min_bits) for index, pattern in enumerate(pattern_list)}
    combined_pattern_occurrences = pd.DataFrame()
    combined_pattern_occurrences = pd.DataFrame(columns=['Pattern', 'Occurrences'])

    # Initialize global statistics
    overall_stats = {
        'total_occurrences': 0,
        'num_patterns': len(pattern_list),
        #'num_nz_patterns': 0,
        'entropy': 0,
        'size_per_pattern': 0,
        'size_per_pattern_bit_roundup': 0,
        'size_per_pattern_byte_roundup': 0,
        'size_uniform_code': 0,
        'size_uniform_bit_roundup': 0,
        'size_unifrom_byte_roundup': 0,
        'm':m,
        'n':n,
        'size_dictionary': m * n * len(pattern_list)
    }

    for start_row in range(0, total_rows, chunk_size):
        print("Processing start_row:", start_row)
        end_row = min(start_row + chunk_size, total_rows)
        data_chunk = data[start_row:end_row]

        if isinstance(data_chunk, pd.DataFrame):
            data_chunk = data_chunk.to_numpy()
        str_array = float_to_bin_array(data_chunk.flatten())
        img_orig = bin_to_image(str_array)
        img_array = np.array(img_orig)
        total_original_size += img_array.size


        pattern_occurance, matched_binary_representations ,df_pattern_occurrences = get_pattern_occurance_non_overlapping(img_array, pattern_list, lookup_table)
        overall_stats['total_occurrences'] += sum(pattern_occurance)

        compressed_size_chunk = len(matched_binary_representations) * min_bits
        total_compressed_size += compressed_size_chunk
        compressed_data_all.extend(matched_binary_representations)
        # Append the current chunk's pattern occurrences to the combined DataFrame
        #combined_pattern_occurrences = combined_pattern_occurrences.append(df_pattern_occurrences, ignore_index=True)
        #combined_pattern_occurrences = pd.concat([combined_pattern_occurrences, df_pattern_occurrences], ignore_index=True)
        # Update combined_pattern_occurrences
        for index, row in df_pattern_occurrences.iterrows():

            Pattern = row['Pattern']

            Occurrences = row['Occurrences']

            if Pattern in combined_pattern_occurrences['Pattern'].values:
                combined_pattern_occurrences.loc[
                    combined_pattern_occurrences['Pattern'] == Pattern, 'Occurrences'] += Occurrences

            else:
                combined_pattern_occurrences = pd.concat([combined_pattern_occurrences,
                                                          pd.DataFrame([[Pattern, Occurrences]],
                                                                       columns=['Pattern', 'Occurrences'])],
                                                         ignore_index=True)

    # Calculate final statistics based on global accumulations
    overall_stats['entropy'] = calculate_entropy(list(pattern_occurance))
    overall_stats['size_per_pattern'] = np.log2(overall_stats['num_patterns']) if overall_stats['num_patterns'] else 0
    overall_stats['size_per_pattern_bit_roundup'] = np.ceil(overall_stats['size_per_pattern'])
    overall_stats['size_per_pattern_byte_roundup'] = np.ceil(overall_stats['size_per_pattern'] / 8) * 8

    overall_stats['size_uniform_code'] = overall_stats['size_per_pattern'] * overall_stats['total_occurrences']
    overall_stats['size_uniform_bit_roundup'] = overall_stats['size_per_pattern_bit_roundup'] * overall_stats['total_occurrences']
    overall_stats['size_unifrom_byte_roundup'] = overall_stats['size_per_pattern_byte_roundup'] * overall_stats['total_occurrences']


    # Calculate compression ratio
    compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0

    return {
        'compressed_data_all': compressed_data_all,
        'total_original_size': total_original_size,
        'total_compressed_size': total_compressed_size,
        'compression_ratio': compression_ratio,
        'matched_binary_representations': matched_binary_representations,
        'lookup_table': lookup_table,
        'stats': overall_stats,
        'combined_pattern_occurrences': combined_pattern_occurrences

    }

def calculate_compression_ratios(result, pattern_list, min_bits):
    ideal_ratio = result['total_original_size'] / result['total_compressed_size']
    #base_ratio = result['total_original_size'] / (result['total_compressed_size'] + (pattern_list.shape[0] * pattern_list[0].shape[1] + 16))
    base_ratio = result['total_original_size'] / (
                result['total_compressed_size'] + (pattern_list.size + 16))


    lookup_ratio = result['total_original_size'] / (result['total_compressed_size'] + pattern_list.size + (len(pattern_list) * min_bits + 16))

    return ideal_ratio, base_ratio, lookup_ratio

def process_feature(dataset_name, feature_idx, feature_data3,m,n):
    chunk_size = m
    data= feature_data3

    try:
       
        feature_data1 = apply_padding(data, chunk_size)
        feature_data = feature_data1.reshape(-1)

    except AttributeError:

        if isinstance(feature_data1, pd.DataFrame):

            feature_data = feature_data1.to_numpy().reshape(-1)
        else:
            print(" another type that doesn't support reshape")
            raise

    str_array = float_to_bin_array(feature_data)

    img_orig = bin_to_image(str_array)
    img_array1 = np.array(img_orig)
    img_array =img_array1

    # Generate patterns
    patterns_from_data = generate_patterns_from_data(img_array, m, n)
    pattern_list = np.array(patterns_from_data)
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



def read_and_compress_dataset(dataset_path, block_sizes):
    results_df = pd.DataFrame()
    results_pattern_df=pd.DataFrame()

    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    try:
        ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
        groups = ts_data.groupby(0)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

        for group_id, group in groups:
            group = group.drop(columns=0)
            group.fillna(0, inplace=True)
            group = group.astype(np.float32)

            n_samples, n_timesteps = group.shape

            print(f"Group {group_id}: {n_samples} rows, {n_timesteps} columns")

            for m, n in block_sizes:
                print(m, n)
                result = process_feature(dataset_name, group_id, group, m, n)
                combined_pattern_occurrences = result[3]['combined_pattern_occurrences']


                # Create a DataFrame for the current iteration's results
                current_results_df = pd.DataFrame({
                    'Dataset Name': [result[0]],
                    'Feature Index': [result[1]],
                    'Ideal Ratio': [result[2][0]],
                    'Base Ratio': [result[2][1]],
                    'Lookup Ratio': [result[2][2]],
                    'Original Size': [result[3]['total_original_size']],
                    'm': [result[3]['stats']['m'] if 'stats' in result[3] and 'm' in result[3]['stats'] else result[5]],  # Get m from stats if available
                    'n': [result[3]['stats']['n'] if 'stats' in result[3] and 'n' in result[3]['stats'] else result[6]],  # Get n from stats if available
                    'Total Occurrences': [result[3]['stats']['total_occurrences'] if 'stats' in result[3] and 'total_occurrences' in result[3]['stats'] else None],
                    'num_patterns': [result[3]['stats']['num_patterns'] if 'stats' in result[3] and 'num_patterns' in result[3]['stats'] else None],
                    'Size Per Pattern Bit Roundup': [result[3]['stats']['size_per_pattern_bit_roundup'] if 'stats' in result[3] and 'size_per_pattern_bit_roundup' in result[3]['stats'] else None],
                    'Size Per Pattern Byte Roundup': [result[3]['stats']['size_per_pattern_byte_roundup'] if 'stats' in result[3] and 'size_per_pattern_byte_roundup' in result[3]['stats'] else None],
                    'Size Uniform Code': [result[3]['stats']['size_uniform_code'] if 'stats' in result[3] and 'size_uniform_code' in result[3]['stats'] else None],
                    'entropy': [result[3]['stats']['entropy'] if 'stats' in result[3] and 'entropy' in result[3]['stats'] else None],
                    'Size Uniform Byte Roundup': [result[3]['stats']['size_unifrom_byte_roundup'] if 'stats' in result[3] and 'size_unifrom_byte_roundup' in result[3]['stats'] else None],
                    'Size Dictionary':[result[3]['stats']['size_dictionary'] if 'stats' in result[3] and 'size_dictionary' in result[3]['stats'] else None]

                })

                results_pattern_df = pd.concat([results_pattern_df, combined_pattern_occurrences], ignore_index=True)
                results_df = pd.concat([results_df, current_results_df], ignore_index=True)
                results_df.to_csv(log_file, index=False, header=True)
                results_pattern_df.to_csv(log_file2, index=False, header=True)

    except Exception as e:
        print(f"Error processing dataset: {e}")

    return results_df

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
    parser.add_argument('--outcsv2', dest='log_file2', default="./log_out2.csv", help='Second output CSV file.')
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

            group = group.astype(np.float32)
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
                compressor.get_compression_ratio(),
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
    log_file2 = args.log_file2
    num_threads = args.num_threads
    mode = args.mode
    block_sizes = [(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8),(9,8),(10, 8),(1,16),(2,16),
                 (3,16),(4,16),(5,16),(6,16),(7,16),(8, 16),(9, 16),  (10, 16),(1,32),(2,32),(3,32),(4,32),
                (5,32),(6,32),(7,32),(8,32),(9,32),(10, 32)]



    print(f"Compressing dataset: {dataset_path} with variant: {comp_variant} and pattern: {pattern}..., log file: {log_file}..., num_threads: {num_threads}...,mode:{mode}")

    if mode == 'signal':
        results_df = read_and_compress_dataset(dataset_path, block_sizes)
        results_other =read_and_describe_dataset_othertools(dataset_path)

    elif mode == 'all':
        results =read_and_describe_dataset_all(dataset_path)
    else:
        print("Invalid mode. Use 'save' or 'read'.")
        sys.exit(1)

    Combined_df = pd.merge(results_df, results_other, on=['Dataset Name', 'Feature Index'], how='inner')
    Combined_df.to_csv(log_file, index=False, header=True)
