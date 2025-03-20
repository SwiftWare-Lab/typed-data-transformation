import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import zlib
import subprocess

# ===============  Your Existing Utilities & Tools  ===============
from modeling.utils import tuple_to_string, compute_entropy, list_to_string, find_max_consecutive_similar_values
from modeling.compression_tools import (
    zstd_comp, zlib_comp, bz2_comp, snappy_comp, fastlz_compress,
    rle_compress, huffman_compress, blosc_comp, blosc_comp_bit
)

# =============== 1) Basic Compression Helpers  ===============
def zstd_comp_cmd(data_bytes, level=3):
    """
    External call to the 'zstd' binary.
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

# =============== 2) Entropy & Feature Extraction  ===============
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
    Returns an array of 5 features: [avg_ent, std_ent, max_ent, min_ent, freq_std]
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
        return np.array([0,0,0,0])

def extract_only_frequency(byte_group, window_size=65536):
    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    return np.array([np.std(byte_freqs)])

# =============== 3) transform_data, compress_data, analyze_data  ===============
def transform_data(data_set_list, order='C'):
    """
    Flatten each 2D array in data_set_list (using order 'C' or 'F'),
    then concatenate into one 1D np.array of dtype np.byte.
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
    Flatten each 2D array in data_set_list with the given order,
    compress it, and sum total compressed size.
    Returns (list_of_compressed, total_compressed_size).
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
    Compute entropy metrics for each sub-array in data_set_list,
    plus a weighted entropy (WE) and overall entropy of data_set_original.
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

# =============== 4) Additional Cluster Metrics: WCSS, Gap Statistic  ===============
def compute_wcss(X, labels):
    """
    Compute within-cluster sum of squares (WCSS) for a given dataset X and labels.
    X: shape (n_samples, n_features)
    labels: cluster labels for each sample.
    """
    unique_labels = np.unique(labels)
    wcss = 0.0
    for lab in unique_labels:
        cluster_points = X[labels == lab]
        centroid = np.mean(cluster_points, axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def gap_statistic(X, k, B=10):
    """
    Compute Gap statistic for clustering dataset X into k clusters.
    X: (n_samples, n_features)
    B: number of reference datasets.
    Returns the gap value.
    """
    # Cluster the original data
    # Here, we assume hierarchical clustering using 'complete' linkage
    linked = linkage(X, method='complete')
    labels = fcluster(linked, k, criterion='maxclust')
    wcss = compute_wcss(X, labels)
    log_wcss = np.log(wcss)

    # Create B reference datasets
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    ref_log_wcss = []
    for b in range(B):
        random_ref = np.random.uniform(mins, maxs, size=X.shape)
        linked_ref = linkage(random_ref, method='complete')
        labels_ref = fcluster(linked_ref, k, criterion='maxclust')
        wcss_ref = compute_wcss(random_ref, labels_ref)
        ref_log_wcss.append(np.log(wcss_ref))
    gap = np.mean(ref_log_wcss) - log_wcss
    return gap

# ===============  5) Build comp_list from cluster labels  ===============
def build_comp_list_from_clusters(byte_groups, labels):
    """
    For each unique label, gather the corresponding groups (1D arrays) into a 2D array.
    Return a list of these 2D arrays (comp_list).
    """
    cluster_dict = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_dict[lab].append(i)
    comp_list = []
    for lab in sorted(cluster_dict.keys()):
        indices = cluster_dict[lab]
        if not indices:
            continue
        length_g = len(byte_groups[indices[0]])
        arr2d = np.zeros((len(indices), length_g), dtype=np.uint8)
        for row_idx, grp_i in enumerate(indices):
            arr2d[row_idx] = byte_groups[grp_i]
        comp_list.append(arr2d)
    return comp_list

# ===============  6) Main run_analysis (Clustering-based Decomposition)  ===============
def run_analysis(folder_path):
    """
    For each .tsv file:
      1) Read and flatten the numeric data.
      2) Split data into 4 groups.
      3) For each feature scenario (e.g., "All Features (5D)", "Only Entropy (4D)", "Only Freq (1D)"):
         - Build a feature matrix from the groups.
         - Perform hierarchical clustering (using 'complete' linkage).
         - For each k in [2..min(4, n_groups)]:
             * Obtain cluster labels and compute clustering metrics:
                 - Silhouette score,
                 - Gap statistic,
                 - Davies-Bouldin score,
                 - Calinski-Harabasz score.
             * Build a decomposition (comp_list) from the cluster labels.
             * Measure compression ratios:
                 - "Standard" (entire data as one chunk),
                 - "Decomposed" (compress each cluster individually, col-based),
                 - "RowBasedDecomposed" (row-major flattening).
         - Store results in a CSV.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    results_records = []

    # Define compression tools (using zstd_comp_cmd as an example)
    comp_tools = {
        "zstd_cmd": zstd_comp_cmd,
        "zstd": zstd_comp  # You can add more if needed
    }

    # Define feature scenarios
    feature_scenarios = {
        "All Features (5D)": extract_5_features,
        "Only Entropy (4D)": extract_only_entropy,
        "Only Freq (1D)": extract_only_frequency
    }

    # List all .tsv files in folder
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in '{folder_path}'.")
        return

    for fname in tsv_files:
        dataset_path = os.path.join(folder_path, fname)
        dataset_name = os.path.splitext(fname)[0]
        print(f"\n===== Processing Dataset: {dataset_name} =====")

        try:
            df = pd.read_csv(dataset_path, sep='\t', header=None)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # Flatten numeric data (skip col0 if not numeric)
        byte_cols = df.columns[1:]
        numeric_vals = df[byte_cols].to_numpy().astype(np.float32)
        # Use row-major order here (could also use 'F' if preferred)
        flattened_bytes = numeric_vals.flatten(order='C').tobytes()

        # Split into 4 groups (interleaved)
        arr = np.frombuffer(flattened_bytes, dtype=np.uint8)
        byte_groups = [arr[i::4] for i in range(4)]
        n_groups = len(byte_groups)
        if n_groups < 2:
            print("Fewer than 2 groups => skipping.")
            continue

        # Standard compression: treat entire data as one chunk (shape (1, -1))
        entire_arr_2d = [arr.reshape(1, -1)]
        for tool_name, comp_func in comp_tools.items():
            _, full_comp_size = compress_data(entire_arr_2d, comp_func)
            std_ratio = ratio_or_inf(len(arr), full_comp_size)
            results_records.append({
                "Dataset": dataset_name,
                "FeatureScenario": "Original",
                "k": 0,
                "Silhouette": None,
                "GapStatistic": None,
                "DaviesBouldin": None,
                "CalinskiHarabasz": None,
                "ClusterConfig": "(1,2,3,4) no clustering",
                "CompressionTool": tool_name,
                "StandardSize(B)": full_comp_size,
                "StandardRatio": std_ratio,
                "DecomposedSize(B)": None,
                "DecomposedRatio": None,
                "RowBasedDecomposedSize(B)": None,
                "RowBasedDecomposedRatio": None
            })

        # Process each feature scenario
        for scenario_name, extractor in feature_scenarios.items():
            # Build feature matrix from each group
            feature_list = [extractor(grp) for grp in byte_groups]
            feature_matrix = np.array(feature_list)
            if feature_matrix.shape[0] < 2:
                continue

            # Hierarchical clustering on feature matrix
            linked = linkage(feature_matrix, method='complete')

            # Try k from 2 up to min(4, number of groups)
            max_k = min(4, feature_matrix.shape[0])
            for k_val in range(2, max_k+1):
                labels_k = fcluster(linked, k_val, criterion='maxclust')
                unique_labels = np.unique(labels_k)
                # Compute silhouette score if valid
                if len(unique_labels) >= 2 and len(unique_labels) <= (feature_matrix.shape[0] - 1):
                    try:
                        sil_val = silhouette_score(feature_matrix, labels_k)
                    except:
                        sil_val = -1
                else:
                    sil_val = -1

                # Compute additional metrics (if possible)
                try:
                    db_score = davies_bouldin_score(feature_matrix, labels_k)
                except Exception:
                    db_score = None
                try:
                    ch_score = calinski_harabasz_score(feature_matrix, labels_k)
                except Exception:
                    ch_score = None
                try:
                    gap = gap_statistic(feature_matrix, k_val, B=10)
                except Exception:
                    gap = None

                # Build cluster configuration string (e.g., "(1,3)|(2,4)")
                cluster_map = {}
                for i, lab in enumerate(labels_k):
                    cluster_map.setdefault(lab, []).append(i+1)
                cluster_str_parts = []
                for c_label in sorted(cluster_map.keys()):
                    cluster_str_parts.append("({})".format(",".join(str(x) for x in cluster_map[c_label])))
                config_str = "|".join(cluster_str_parts)

                # Build comp_list from clusters (each cluster becomes a 2D array)
                comp_list = build_comp_list_from_clusters(byte_groups, labels_k)

                # Measure compression for each tool
                for tool_name, comp_func in comp_tools.items():
                    # Standard: compress entire data as one chunk
                    _, full_comp_size = compress_data(entire_arr_2d, comp_func)
                    std_ratio = ratio_or_inf(len(arr), full_comp_size)

                    # Decomposed (column-based)
                    _, dec_size = compress_data(comp_list, comp_func, order='F')
                    dec_ratio = ratio_or_inf(len(arr), dec_size)

                    # Decomposed (row-based)
                    _, dec_size_row = compress_data(comp_list, comp_func, order='C')
                    dec_ratio_row = ratio_or_inf(len(arr), dec_size_row)

                    results_records.append({
                        "Dataset": dataset_name,
                        "FeatureScenario": scenario_name,
                        "k": k_val,
                        "Silhouette": sil_val,
                        "GapStatistic": gap,
                        "DaviesBouldin": db_score,
                        "CalinskiHarabasz": ch_score,
                        "ClusterConfig": config_str,
                        "CompressionTool": tool_name,
                        "StandardSize(B)": full_comp_size,
                        "StandardRatio": std_ratio,
                        "DecomposedSize(B)": dec_size,
                        "DecomposedRatio": dec_ratio,
                        "RowBasedDecomposedSize(B)": dec_size_row,
                        "RowBasedDecomposedRatio": dec_ratio_row
                    })

    # Convert results to DataFrame and save to CSV
    df_results = pd.DataFrame(results_records)
    out_csv = os.path.join(folder_path, "merged_clustering_compression_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nDone! Results saved at: {out_csv}")
    print(df_results.head(30))

# =============== 6) Entry Point  ===============
if __name__ == "__main__":
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)
