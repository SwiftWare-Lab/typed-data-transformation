import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import zlib
import subprocess

################## Your Compression Tools (from second code) ##################

from modeling.utils import tuple_to_string, compute_entropy, list_to_string, find_max_consecutive_similar_values
from modeling.compression_tools import (
    zstd_comp, zlib_comp, bz2_comp, fastlz_compress,
    rle_compress, huffman_compress, blosc_comp, blosc_comp_bit
)


################## 1) Basic Compression Helpers ##################
def zstd_comp_cmd(data_bytes, level=3):
    """
    If you don't want to rely on an external 'zstd' command, you can remove this.
    Otherwise, it writes data to a temp file and calls the zstd binary.
    """
    with open("tmp_zstd_input.bin", "wb") as f:
        f.write(data_bytes)
    out_file = "tmp_zstd_output.zst"
    cmd = ["zstd", f"-{level}", "tmp_zstd_input.bin", "-o", out_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out_file, "rb") as f:
        compressed = f.read()
    os.remove("tmp_zstd_input.bin")
    os.remove("tmp_zstd_output.zst")
    return compressed


def ratio_or_inf(orig_size, comp_size):
    if comp_size == 0:
        return float('inf')
    return float(orig_size) / comp_size


################## 2) Entropy & Feature Extraction (from first code) ##################
def compute_entropy(data_window):
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_entropy_over_data(data, window_size=65536):
    entropies = []
    for start_idx in range(0, len(data), window_size):
        window = data[start_idx:start_idx + window_size]
        if window.size == 0:
            break
        ent = compute_entropy(window)
        entropies.append(ent)
    return entropies


def extract_5_features(byte_group, window_size=65536):
    """
    [avg_ent, std_ent, max_ent, min_ent, freq_std]
    """
    entropies = calculate_entropy_over_data(byte_group, window_size)
    if len(entropies) > 0:
        avg_ent = np.mean(entropies)
        std_ent = np.std(entropies)
        max_ent = np.max(entropies)
        min_ent = np.min(entropies)
    else:
        avg_ent = std_ent = max_ent = min_ent = 0

    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    freq_std = np.std(byte_freqs)

    return np.array([avg_ent, std_ent, max_ent, min_ent, freq_std])


def extract_only_entropy(byte_group, window_size=65536):
    entropies = calculate_entropy_over_data(byte_group, window_size)
    if len(entropies) > 0:
        return np.array([np.mean(entropies),
                         np.std(entropies),
                         np.max(entropies),
                         np.min(entropies)])
    else:
        return np.array([0, 0, 0, 0])


def extract_only_frequency(byte_group, window_size=65536):
    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    return np.array([np.std(byte_freqs)])


################## 3) Build/Measure Decomposition (from second code) ##################
def transform_data(data_set_list, order='C'):
    """
    Flatten each 2D array (row-major or col-major), then concatenate into one byte array.
    """
    all_bytes = []
    for cmp in data_set_list:
        data_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        all_bytes.append(data_bytes)
    if not all_bytes:
        return np.array([], dtype=np.byte)
    return np.concatenate(all_bytes, axis=0)


def compress_data(data_set_list, compress_method, order='F'):
    """
    Flatten each 2D array in 'data_set_list' with the given 'order',
    compress it, sum total compressed size.
    Return (list_of_compressed, total_compressed_size).
    """
    compressed_data = []
    total_size = 0
    for cmp in data_set_list:
        bytes_ = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        c_ = compress_method(bytes_)
        compressed_data.append(c_)
        total_size += len(c_)
    return compressed_data, total_size


def analyze_data(data_set_list, data_set_original):
    """
    Similar to 'analyze_data' in second code: measure entropy, etc.
    """
    entropy_list = []
    WE = 0
    tot_size = 0
    max_rep_lst = []
    uniq_ratio_lst = []

    for cmp in data_set_list:
        tot_size += len(cmp.flatten().tobytes())

    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten('F').tobytes(), dtype=np.byte)
        e_val = compute_entropy(data_set_bytes)
        entropy_list.append(e_val)
        max_rep = find_max_consecutive_similar_values(data_set_bytes)
        max_rep_lst.append(max_rep)
        uniq_ratio_lst.append(len(set(data_set_bytes)) / len(data_set_bytes))
        if tot_size > 0:
            WE += e_val * (len(data_set_bytes) / tot_size)

    entropy_word = compute_entropy(data_set_original)
    return entropy_list, WE, entropy_word, max_rep_lst, uniq_ratio_lst


################## 4) Reorder by Cluster => Build comp_list (like second code) ##################
def build_comp_list_from_clusters(byte_groups, labels):
    """
    In the second code, each 'decomposition' is a list of 2D arrays.
    Here, 'labels' define which cluster each group belongs to.
    We'll gather groups with the same label into one 2D array of shape (#groups_in_label, group_length).
    Then return a list of these arrays.
    """
    # First figure out group length
    # Each group 'byte_groups[i]' is a 1D array of type uint8.
    group_length = [len(g) for g in byte_groups]
    # (We expect them all the same length if they came from a uniform split, but let's be safe.)
    # We'll store (cluster_label -> list of group indices)
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_dict[lab].append(i)

    # Build comp_list: each cluster => shape (#groups_in_label, length_of_each_group)
    comp_list = []
    for lab in sorted(cluster_dict.keys()):
        indices_in_cluster = cluster_dict[lab]
        # we create a 2D array of shape ( len(indices_in_cluster), length_of_each_group )
        # We'll pick the length from the first group in this cluster
        if not indices_in_cluster:
            continue
        first_idx = indices_in_cluster[0]
        length_g = len(byte_groups[first_idx])
        arr2d = np.zeros((len(indices_in_cluster), length_g), dtype=np.uint8)
        for row_idx, grp_i in enumerate(indices_in_cluster):
            arr2d[row_idx] = byte_groups[grp_i]
        comp_list.append(arr2d)
    return comp_list


def build_reordered_single_array1(comp_list, order='F'):
    """
    Create a single 2D array from the entire comp_list, so we treat it like "reordered" in second code.
    That is, all rows from each cluster appended vertically.
    """
    # Count total rows
    total_rows = 0
    length_g = 0
    for arr2d in comp_list:
        total_rows += arr2d.shape[0]
        length_g = arr2d.shape[1]  # assume consistent length

    # Make one big 2D array of shape (total_rows, length_g)
    big_arr = np.zeros((total_rows, length_g), dtype=np.uint8)
    start_row = 0
    for arr2d in comp_list:
        r_ = arr2d.shape[0]
        big_arr[start_row:start_row + r_] = arr2d
        start_row += r_
    return big_arr


###########################
def build_reordered_single_array(comp_list, order='F'):
    """
    1) Create one big 2D array by stacking each 2D array in 'comp_list' vertically.
       - The total row count is sum of row counts of each array in comp_list.
       - The column count (length_g) is taken from the first array (assumed consistent).
    2) Flatten that big 2D array (in 'C' or 'F' order).
    3) Return a 1D np.ndarray of dtype np.byte.

    This effectively combines the "reorder" step with the "transform_data" approach
    so you get a single flat array of bytes representing the entire cluster decomposition.
    """
    # Handle empty input gracefully
    if not comp_list:
        return np.array([], dtype=np.byte)

    # Determine total rows and the column length
    total_rows = 0
    length_g = comp_list[0].shape[1]  # assume at least one array, same number of columns
    for arr2d in comp_list:
        total_rows += arr2d.shape[0]

    # Allocate a big 2D array
    big_arr = np.zeros((total_rows, length_g), dtype=np.uint8)

    # Copy each 2D array in 'comp_list' into 'big_arr' sequentially
    start_row = 0
    for arr2d in comp_list:
        r_ = arr2d.shape[0]
        big_arr[start_row: start_row + r_] = arr2d
        start_row += r_

    # Now flatten that 2D array in the desired order and convert to bytes
    flattened_bytes = big_arr.flatten(order=order).tobytes()

    # Convert bytes to a 1D numpy array of dtype np.byte (== int8)
    data_bytes = np.frombuffer(flattened_bytes, dtype=np.byte)
    return data_bytes


################## 5) The Main run_analysis, merging both codes ##################
def run_analysis(folder_path):
    """
    For each .tsv file:
      1) We read it -> flatten -> split into 4 groups
      2) For each feature scenario (All Feat, Only Entropy, Only Freq):
         * Build feature_matrix
         * Linkage => for k in [2..n_groups], get labels => reorder
         * Construct 'comp_list' (like a decomposition).
         * Also measure 'standard' compression (the entire data as 1 chunk).
         * Then measure 'decomposed' & 'reordered' sizes & ratio, store in CSV
    """
    if not os.path.isdir(folder_path):
        print("Not a valid folder:", folder_path)
        return

    # Let's store results in a list of dict
    results_records = []

    # Some example compression tools from the second code
    comp_tools = {
        #  "blosc": blosc_comp,
        # "blosc_bit": blosc_comp_bit,
        "zstd": zstd_comp_cmd,  # If you want to also do external zstd command
        # "zlib": zlib_comp,
    }

    # Feature scenarios
    feature_scenarios = {
        "All Features (5D)": extract_5_features,
        "Only Entropy (4D)": extract_only_entropy,
        "Only Freq (1D)": extract_only_frequency
    }

    # List all .tsv
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print("No .tsv files found in", folder_path)
        return

    for fname in tsv_files:
        dataset_name = os.path.splitext(fname)[0]
        fpath = os.path.join(folder_path, fname)
        print(f"Processing: {dataset_name}")

        # read data
        try:
            df = pd.read_csv(fpath, sep='\t', header=None)
        except Exception as e:
            print("Failed to load", fname, e)
            continue

        # flatten numeric data
        numeric_vals = df.values[1:10, 1].to_numpy().astype(np.float32)  # skip col0 if not relevant
        flattened = numeric_vals.flatten(order='F').tobytes()

        # split into 4 groups
        arr = np.frombuffer(flattened, dtype=np.uint8)
        # If you always want exactly 4 interleaved groups:
        byte_groups = []
        for i in range(4):
            g = arr[i::4]
            byte_groups.append(g)
        n_groups = len(byte_groups)

        if n_groups < 2:
            print("Only 1 group => skip clustering.")
            continue

        # measure "standard" compression (the entire data as a single chunk):
        # We'll treat it like shape(1, total_bytes). For second code logic, we do:
        entire_arr_2d = [arr.reshape(1, -1)]  # a list with one 2D array

        # We'll store that in results for reference
        for ctool_name, ctool_func in comp_tools.items():
            # compress the entire data
            _, full_comp_size = compress_data(entire_arr_2d, ctool_func)
            std_ratio = ratio_or_inf(len(arr), full_comp_size)
            # record it as "k=0" => no clustering, scenario=Original
            results_records.append({
                "Dataset": dataset_name,
                "FeatureScenario": "Original",
                "k": 0,
                "Silhouette": None,
                "ClusterConfig": "(1,2,3,4) no clustering",
                "CompressionTool": ctool_name,
                "StandardSize(B)": full_comp_size,
                "StandardRatio": std_ratio,
                "DecomposedSize(B)": None,
                "DecomposedRatio": None,

                "RowBasedDecomposedSize(B)": None,
                "RowBasedDecomposedRatio": None,

            })

        # For each feature scenario
        for scenario_name, extractor in feature_scenarios.items():
            # build feature matrix
            feature_list = []
            for grp in byte_groups:
                fv = extractor(grp)
                feature_list.append(fv)
            feature_matrix = np.array(feature_list)
            if feature_matrix.shape[0] < 2:
                continue

            # hierarchical clustering
            linked = linkage(feature_matrix, method='complete')

            # For k in [2..4] (or up to n_groups)
            max_k = min(4, feature_matrix.shape[0])
            for k_val in range(2, max_k + 1):
                labels_k = fcluster(linked, k_val, criterion='maxclust')
                # compute silhouette
                # Only valid if n_clusters between 2 and n_samples-1
                n_clusters = len(np.unique(labels_k))
                if n_clusters < 2 or n_clusters > feature_matrix.shape[0] - 1:
                    sil_val = -1
                else:
                    try:
                        sil_val = silhouette_score(feature_matrix, labels_k)
                    except:
                        sil_val = -1

                # Build a cluster config string
                cluster_map = {}
                for i, lab_ in enumerate(labels_k):
                    cluster_map.setdefault(lab_, []).append(i + 1)
                cluster_str_parts = []
                for c_label in sorted(cluster_map.keys()):
                    cluster_str_parts.append("({})".format(",".join(str(x) for x in cluster_map[c_label])))
                config_str = "|".join(cluster_str_parts)

                # Now we build "comp_list" as in second code => each cluster => a 2D array
                comp_list = build_comp_list_from_clusters(byte_groups, labels_k)

                # measure compression for each tool
                for ctool_name, ctool_func in comp_tools.items():
                    # 1) standard = entire data
                    _, full_comp_size = compress_data(entire_arr_2d, ctool_func)
                    std_ratio = ratio_or_inf(len(arr), full_comp_size)

                    # 2) decomposed
                    #    "decomposed" means compress each cluster's 2D array individually
                    #    in "column-major" or "order='F'" by default
                    _, dec_size = compress_data(comp_list, ctool_func, order='F')
                    dec_ratio = ratio_or_inf(len(arr), dec_size)

                    # row-based decomposed
                    _, dec_size_row = compress_data(comp_list, ctool_func, order='C')
                    dec_ratio_row = ratio_or_inf(len(arr), dec_size_row)

                    # record
                    results_records.append({
                        "Dataset": dataset_name,
                        "FeatureScenario": scenario_name,
                        "k": k_val,
                        "Silhouette": sil_val,
                        "ClusterConfig": config_str,
                        "CompressionTool": ctool_name,

                        "StandardSize(B)": full_comp_size,
                        "StandardRatio": std_ratio,

                        "DecomposedSize(B)": dec_size,
                        "DecomposedRatio": dec_ratio,

                        "RowBasedDecomposedSize(B)": dec_size_row,
                        "RowBasedDecomposedRatio": dec_ratio_row,

                    })

    # Make a DataFrame
    df_results = pd.DataFrame(results_records)
    # Save
    out_csv = os.path.join(folder_path, "merged_clustering_compression_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"All done! Results saved to: {out_csv}")
    print(df_results.head(30))


################## 6) Entry Point ##################
if __name__ == "__main__":
    folder_path = r"C:\Users\jamalids\Downloads\dataset\High-Entropy\32\TEST"
    run_analysis(folder_path)