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
        # "No grouping" => interpret it as the entire array is one chunk
        # The existing code merges all channels and returns single-entropy
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


# --------------------------- Byte-Group Entropy --------------------------- #

def compute_bytegroup_entropy(byte_data, bytes_per_value=4):
    """
    For a float32 array (4 bytes each), we create 4 clusters:
      - Cluster #1: all first bytes of every float
      - Cluster #2: all second bytes
      - Cluster #3: all third bytes
      - Cluster #4: all fourth bytes

    Then we compute the weighted average of entropies of these 4 clusters.
    If you use float64, you'd have 8 bytes per value => 8 clusters, etc.
    """

    # raw is the entire dataset's bytes. We'll separate it by offset 0..(bytes_per_value-1)
    raw = np.frombuffer(byte_data, dtype=np.uint8)
    # total floats = len(raw) // bytes_per_value
    total_len = len(raw)
    # We'll build the 4 clusters
    cluster_entropies = []
    stride = bytes_per_value
    for offset in range(bytes_per_value):
        # Extract the offset-th byte from each float
        # i.e. raw[offset], raw[offset+stride], raw[offset+2*stride], ...
        cluster = raw[offset::stride]
        # compute entropy of this cluster
        e = compute_total_entropy(cluster.tobytes())
        # Weighted by cluster size
        w = len(cluster) / total_len
        cluster_entropies.append(w * e)

    # sum them up for final "conditional" measure
    return sum(cluster_entropies)


# --------------------------- Main Analysis Function --------------------------- #

def run_analysis(csv_file, raw_data_folder, output_csv="entropy_and_compression_ratioszstd32.csv",
                 default_full_channels=4, data_dtype=np.float32):
    """
    For each dataset listed in the CSV file:
      - Load the corresponding raw TSV file.
      - Convert the numeric data into bytes using the specified data_dtype
        (e.g. np.float32 for 32-bit floats, np.float64 for 64-bit floats).
      - Flatten the data in Fortran (column-major) order so that the columns (channels)
        are interleaved.
      - Split the resulting byte stream into the original channels using the known number
        of channels and the number of bytes per value.
      - Compute:
          1) full_entropy (the entire byte stream)
          2) conditional_entropy (based on config_str from CSV)
          3) bytegroup_entropy (4 clusters for float32, or 8 for float64, etc.)
      - Then compute two ratios:
            * EntropyRatio = FullEntropy / ConditionalEntropy
            * CompRatioRatio = CompressionRatio_Full / CompressionRatio_Chunk
      - Save the results (including FullEntropy, ConditionalEntropy, ByteGroupEntropy, EntropyRatio,
        CompressionRatio_Full, CompressionRatio_Chunk, and CompRatioRatio) to a CSV file.
      - Plot:
          (A) A scatter: x=EntropyRatio, y=CompRatioRatio
          (B) A grouped bar chart of [FullEntropy, ConditionalEntropy, ByteGroupEntropy]
              for each dataset on the x-axis.

    The CSV file must contain at least:
        - DatasetName
        - RunType (with values "Full" and "Decompose_Block_Parallel")
        - ConfigString (for the Chunked run; e.g. "0" or "{ [1]- [2]- [3]- [4] }")
        - CompressionRatio (for both Full and Chunked runs)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Separate Full and Chunked runs and merge them by DatasetName.
    df_full = df[df['RunType'] == 'Full'].copy()
    df_chunk = df[df['RunType'] == 'Chunked_Decompose_Parallel'].copy()
    df_merged = pd.merge(df_full, df_chunk, on='DatasetName', suffixes=('_Full', '_Chunk'))

    results = []

    ### Identify how many bytes each data type has
    bytes_per_value = np.dtype(data_dtype).itemsize  # 4 if float32, 8 if float64

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

        # 1) Full entropy
        full_entropy = compute_total_entropy(byte_data)

        # 2) Conditional entropy (based on config_str from CSV)
        config_str = str(row['ConfigString_Chunk']).strip()
        if config_str == "0":
            total_channels = default_full_channels
        else:
            groups = parse_config_string_to_groups(config_str)
            total_channels = sum(len(g) for g in groups)

        # Split the raw byte stream into channels as per original code
        byte_groups = split_bytes_into_components(byte_data, total_channels, bytes_per_value)
        cond_entropy = compute_conditional_entropy(byte_groups, config_str)

        # 3) Byte-group entropy
        #    E.g. float32 => 4 clusters: each offset in [0..3] => weighted sum
        #    float64 => 8 clusters, etc.
        bytegroup_entropy = compute_bytegroup_entropy(byte_data, bytes_per_value=bytes_per_value)

        # Extract compression ratios
        comp_ratio_full = row['CompressionRatio_Full']
        comp_ratio_chunk = row['CompressionRatio_Chunk']

        # Compute the desired ratios
        entropy_ratio = full_entropy / cond_entropy if cond_entropy != 0 else np.nan
        comp_ratio_ratio = (comp_ratio_full / comp_ratio_chunk) if comp_ratio_chunk != 0 else np.nan

        results.append({
            'DatasetName': dataset_name,
            'FullEntropy': full_entropy,
            'ConditionalEntropy': cond_entropy,
            'ByteGroupEntropy': bytegroup_entropy,
            'EntropyRatio': entropy_ratio,
            'CompRatioFull': comp_ratio_full,
            'CompRatioChunk': comp_ratio_chunk,
            'CompRatioRatio': comp_ratio_ratio
        })
        print(f"Processed {dataset_name}: FullEntropy={full_entropy:.4f}, "
              f"ConditionalEntropy={cond_entropy:.4f}, ByteGroupEntropy={bytegroup_entropy:.4f}, "
              f"EntropyRatio={entropy_ratio:.4f}, CompRatioFull={comp_ratio_full}, "
              f"CompRatioChunk={comp_ratio_chunk}, CompRatioRatio={comp_ratio_ratio:.4f}")

    df_results = pd.DataFrame(results)

    # Save the results to a CSV file.
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # --------------------------- (A) Scatter Plot --------------------------- #
    # Remove points with ConditionalEntropy=0 to avoid /0 in EntropyRatio
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
    output_plot = "entropy_ratio_vs_compression_ratio_ratio64-zstd32.png"
    plt.savefig(output_plot)
    print(f"Plot saved as {output_plot}")
    plt.show()

    # --------------------------- (B) Bar Chart: Full vs Conditional vs ByteGroup --------------------------- #
    # We create a grouped bar chart with 3 bars for each dataset
    df_plot2 = df_results[['DatasetName', 'FullEntropy', 'ConditionalEntropy', 'ByteGroupEntropy']].copy()
    # Sort by dataset name for consistent x-axis
    df_plot2.sort_values(by='DatasetName', inplace=True)

    x_labels = df_plot2['DatasetName'].tolist()
    x = np.arange(len(x_labels))  # index for each dataset
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df_plot2['FullEntropy'], width, label='FullEntropy')
    ax.bar(x, df_plot2['ConditionalEntropy'], width, label='ConditionalEntropy')
    ax.bar(x + width, df_plot2['ByteGroupEntropy'], width, label='ByteGroupEntropy')

    ax.set_xlabel("DatasetName")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Comparison of Entropy: Full vs. Conditional vs. ByteGroup")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig("entropy_comparison_barplot.png")
    print("Bar chart saved as entropy_comparison_barplot.png")
    plt.show()


# --------------------------- Main Execution --------------------------- #

if __name__ == "__main__":
    # Path to the CSV file that contains grouping and compression ratio information.
    csv_file = "/home/jamalids/Documents/zstd/results/max_decompression_throughput_pairs.csv"  # Update this path as needed.
    # Folder containing the raw data TSV files (one per dataset).
    raw_data_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"  # Update as needed.

    # data_dtype => np.float32 => 4 bytes per value => 4 byte clusters
    # or np.float64 => 8 bytes => 8 byte clusters
    run_analysis(csv_file, raw_data_folder,
                 output_csv="entropy_and_compression_ratiosfastzstd_extended.csv",
                 default_full_channels=4,
                 data_dtype=np.float32)
