import matplotlib

matplotlib.use('TkAgg')  # Switch backend if needed

import os
import math
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------- Entropy Calculation --------------------------- #

def compute_entropy(byte_data):
    """
    Compute the entropy (in bits) of the entire byte_data.
    `byte_data` is expected to be a bytes object.
    """
    freq = Counter(byte_data)
    total = len(byte_data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_total_entropy(byte_data):
    """
    Compute the total entropy for the given byte data (without windowing).
    """
    return compute_entropy(byte_data)


# --------------------------- Data Splitting --------------------------- #

def split_bytes_into_components(byte_array, total_channels, bytes_per_value):
    """
    Split the raw byte array into individual channels (components).

    The raw data is assumed to be interleaved such that each record contains
    `total_channels` values, and each value occupies `bytes_per_value` bytes.

    For example, if there are 4 channels and each value is 4 bytes (32-bit float),
    then the total stride per record is 4 * 4 = 16 bytes. Channel 1 is extracted by taking
    the bytes starting at offset 0 and then every 16 bytes, channel 2 from offset 4, etc.
    """
    # Convert the raw bytes into a numpy array of uint8.
    raw = np.frombuffer(byte_array, dtype=np.uint8)
    stride = total_channels * bytes_per_value
    components = []
    for i in range(total_channels):
        offset = i * bytes_per_value
        component = raw[offset::stride]
        components.append(component)
    return components


# --------------------------- Config String Parsing --------------------------- #

def parse_config_string_to_groups(config_str):
    """
    Parse a configuration string to extract the grouping.

    For example:
        "{ [1 2]- [3]- [4] }"  returns [[1, 2], [3], [4]]
        "{ [1]- [2]- [3]- [4] }" returns [[1], [2], [3], [4]]

    Thus, if the config string is "{ [1]- [2]- [3]- [4] }" it means we have 4 groups,
    with each byte group (channel) treated as its own cluster.

    If config_str is "0", returns None (which signals that no grouping was applied).
    (The numbers are assumed to be 1-indexed.)
    """
    config_str = config_str.strip()
    if config_str == "0":
        return None
    groups = re.findall(r'\[(.*?)\]', config_str)
    parsed = []
    for group in groups:
        # Convert each number (as string) to an integer.
        indices = [int(x) for x in group.split() if x.strip()]
        parsed.append(indices)
    return parsed


def compute_conditional_entropy(byte_groups, config_str):
    """
    Compute the conditional entropy based on the grouping defined in config_str.

    If config_str is "0", no grouping was applied so the full file entropy is computed.
    Otherwise, merge the original channels (from byte_groups) according to the parsed grouping.
    For each merged cluster, compute its entropy (on the complete merged byte stream),
    and then return the weighted average entropy (weighted by the proportion of bytes).

    For example, if config_str is "{ [1]- [2]- [3]- [4] }" then each channel is its own cluster.
    """
    if config_str.strip() == "0":
        merged = np.concatenate(byte_groups)
        return compute_total_entropy(merged.tobytes())

    groups = parse_config_string_to_groups(config_str)
    total_length = sum(len(g) for g in byte_groups)
    cond_entropy = 0.0
    for subgroup in groups:
        # The numbers in subgroup are 1-indexed; adjust index when accessing byte_groups.
        merged = np.concatenate([byte_groups[i - 1] for i in subgroup])
        subgroup_length = len(merged)
        group_entropy = compute_total_entropy(merged.tobytes())
        cond_entropy += (subgroup_length / total_length) * group_entropy
    return cond_entropy


# --------------------------- Main Analysis Function --------------------------- #

def run_analysis(csv_file, raw_data_folder, output_csv="entropy_and_compression_ratiosfastlz64.csv",
                 default_full_channels=8, data_dtype=np.float64):
    """
    For each dataset listed in the CSV file:
      - Load the corresponding raw TSV file.
      - Convert the numeric data into bytes using the specified data_dtype
        (e.g. np.float32 for 32-bit floats, np.float64 for 64-bit floats).
      - Flatten the data in Fortran (column-major) order so that the columns (channels)
        are interleaved.
      - Split the resulting byte stream into the original channels using the known number
        of channels and the number of bytes per value.
      - Compute the full entropy (on the entire byte stream) and the conditional entropy by merging
        channels as defined by the ConfigString.
      - Compute two ratios:
            * EntropyRatio = FullEntropy / ConditionalEntropy
            * CompRatioRatio = CompressionRatio_Full / CompressionRatio_Chunk
      - Save the results (including FullEntropy, ConditionalEntropy, EntropyRatio,
        CompressionRatio_Full, CompressionRatio_Chunk, and CompRatioRatio) to a CSV file.
      - Plot a scatter plot with:
            x-axis: EntropyRatio (with dataset names as annotations)
            y-axis: CompRatioRatio

    The CSV file passed to this function must contain at least the following columns:
        - DatasetName
        - RunType (with values "Full" and "Chunked_Decompose_Parallel")
        - ConfigString (for the Chunked run; e.g. "0" or "{ [1]- [2]- [3]- [4] }")
        - CompressionRatio (for both Full and Chunked runs, which become available as
                              CompressionRatio_Full and CompressionRatio_Chunk after merging)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Separate Full and Chunked runs and merge them by DatasetName.
    df_full = df[df['RunType'] == 'Full'].copy()
    df_chunk = df[df['RunType'] == 'Decompose_Block_Parallel'].copy()
    df_merged = pd.merge(df_full, df_chunk, on='DatasetName', suffixes=('_Full', '_Chunk'))

    results = []
    for idx, row in df_merged.iterrows():
        dataset_name = row['DatasetName']
        raw_file = os.path.join(raw_data_folder, dataset_name + '.tsv')
        if not os.path.exists(raw_file):
            print(f"Raw file {raw_file} not found, skipping {dataset_name}.")
            continue
        try:
            ts_data = pd.read_csv(raw_file, delimiter='\t', header=None)
            # Assume the first column is an identifier; the remaining columns contain numeric data.
            byte_columns = ts_data.columns[1:]
            # Convert the numeric data into the desired type and flatten in Fortran order
            # so that the columns (channels) are interleaved.
            arr = ts_data[byte_columns].to_numpy().astype(data_dtype)
            byte_data = arr.flatten(order='F').tobytes()
        except Exception as e:
            print(f"Error loading {raw_file}: {e}")
            continue

        # Use the Chunked run's ConfigString. If it is "0", assume no grouping.
        config_str = str(row['ConfigString_Chunk']).strip()
        if config_str == "0":
            total_channels = default_full_channels
        else:
            groups = parse_config_string_to_groups(config_str)
            # The total number of original channels is the sum of the counts in the config.
            # For example, if config_str is "{ [1]- [2]- [3]- [4] }", then groups = [[1],[2],[3],[4]]
            # and total_channels = 1+1+1+1 = 4.
            total_channels = sum(len(g) for g in groups)

        # Determine the number of bytes per value.
        bytes_per_value = np.dtype(data_dtype).itemsize
        # Split the raw byte stream into channels.
        byte_groups = split_bytes_into_components(byte_data, total_channels, bytes_per_value)

        # Compute the full entropy over the entire byte stream.
        full_entropy = compute_total_entropy(byte_data)
        # Compute the conditional entropy by merging channels according to the config string.
        cond_entropy = compute_conditional_entropy(byte_groups, config_str)

        # Extract compression ratios from the CSV for the Full and Chunked runs.
        comp_ratio_full = row['CompressionRatio_Full']
        comp_ratio_chunk = row['CompressionRatio_Chunk']

        # Compute the desired ratios.
        entropy_ratio = full_entropy / cond_entropy if cond_entropy != 0 else np.nan
        comp_ratio_ratio = comp_ratio_full / comp_ratio_chunk if comp_ratio_chunk != 0 else np.nan

        results.append({
            'DatasetName': dataset_name,
            'FullEntropy': full_entropy,
            'ConditionalEntropy': cond_entropy,
            'EntropyRatio': entropy_ratio,
            'CompRatioFull': comp_ratio_full,
            'CompRatioChunk': comp_ratio_chunk,
            'CompRatioRatio': comp_ratio_ratio
        })
        print(f"Processed {dataset_name}: FullEntropy={full_entropy:.4f}, "
              f"ConditionalEntropy={cond_entropy:.4f}, EntropyRatio={entropy_ratio:.4f}, "
              f"CompRatioFull={comp_ratio_full}, CompRatioChunk={comp_ratio_chunk}, "
              f"CompRatioRatio={comp_ratio_ratio:.4f}")

    df_results = pd.DataFrame(results)

    # Save the results to a CSV file.
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # --------------------------- Plotting --------------------------- #
    # Remove points with ConditionalEntropy equal to 0 to avoid division issues.
    df_plot = df_results[(df_results['ConditionalEntropy'] != 0) & (df_results['EntropyRatio'] < 4)]


    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['EntropyRatio'], df_plot['CompRatioRatio'], color='blue', s=60)

    # Annotate each point with its DatasetName.
    for _, r in df_plot.iterrows():
        plt.text(r['EntropyRatio'] * 1.01, r['CompRatioRatio'] * 1.01, r['DatasetName'], fontsize=9)

    plt.xlabel('Entropy Ratio (FullEntropy / ConditionalEntropy)')
    plt.ylabel('Compression Ratio Ratio (Full / Chunked)')
    plt.title('Entropy Ratio vs Compression Ratio Ratio')
    plt.grid(True)
    plt.tight_layout()
    output_plot = "entropy_ratio_vs_compression_ratio_ratio64-fastlz64.png"
    plt.savefig(output_plot)
    print(f"Plot saved as {output_plot}")
    plt.show()


# --------------------------- Main Execution --------------------------- #

if __name__ == "__main__":
    # Path to the CSV file that contains grouping and compression ratio information.
    csv_file = "/home/jamalids/Documents/fastlz1/results/max_decompression_throughput_pairs.csv"  # Update this path as needed.
    # Folder containing the raw data TSV files (one per dataset).
    raw_data_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64"  # Update as needed.
    # Specify the desired data type: use np.float32 for 32-bit (4 bytes per value) or np.float64 for 64-bit.
    run_analysis(csv_file, raw_data_folder, default_full_channels=4, data_dtype=np.float32)
