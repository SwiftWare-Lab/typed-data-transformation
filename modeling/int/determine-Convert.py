import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import zlib
import subprocess

# Import custom compression utilities
from modeling.utils import compute_entropy, find_max_consecutive_similar_values
from modeling.compression_tools import zstd_comp, zlib_comp, bz2_comp,fastlz_compress

################## Compression Helper ##################

def ratio_or_inf(orig_size, comp_size):
    return float(orig_size) / comp_size if comp_size else float('inf')

################## Feature Extraction ##################

def extract_entropy_features(byte_group, window_size=65536):
    """
    Compute entropy for the given byte_group in windows,
    and return an array: [mean, std, max, min] entropy.
    """
    entropies = []
    for start_idx in range(0, len(byte_group), window_size):
        window = byte_group[start_idx:start_idx + window_size]
        entropies.append(compute_entropy(window))
    return np.array([np.mean(entropies), np.std(entropies), np.max(entropies), np.min(entropies)])

def extract_entropy(byte_group):
    """
    Baseline entropy extractor: overall entropy as a single feature.
    (Computes entropy on the entire group without windowing.)
    """
    freq = Counter(byte_group)
    total = len(byte_group)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return np.array([entropy])

def extract_entropy_mean(byte_group, window_size=65536):
    features = extract_entropy_features(byte_group, window_size)
    return np.array([features[0]])

def extract_entropy_std(byte_group, window_size=65536):
    features = extract_entropy_features(byte_group, window_size)
    return np.array([features[1]])

def extract_entropy_max(byte_group, window_size=65536):
    features = extract_entropy_features(byte_group, window_size)
    return np.array([features[2]])

def extract_entropy_min(byte_group, window_size=65536):
    features = extract_entropy_features(byte_group, window_size)
    return np.array([features[3]])

def extract_frequency(byte_group):
    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    return np.array([np.std(byte_freqs)])

def extract_all_features(byte_group, window_size=65536):
    """
    Concatenate all entropy features (mean, std, max, min) with frequency.
    Returns a 5-dimensional feature vector.
    """
    ent_features = extract_entropy_features(byte_group, window_size)
    freq_feature = extract_frequency(byte_group)
    return np.concatenate([ent_features, freq_feature])

################## Clustering Metrics ##################

# def compute_gap_statistic(feature_matrix, labels_k, k_val, n_refs=10):
#     actual_disp = np.sum([np.mean(cdist(feature_matrix[labels_k == c],
#                                         np.mean(feature_matrix[labels_k == c], axis=0, keepdims=True)))
#                           for c in np.unique(labels_k)])
#     random_disps = []
#     for _ in range(n_refs):
#         random_ref = np.random.uniform(np.min(feature_matrix, axis=0),
#                                        np.max(feature_matrix, axis=0),
#                                        feature_matrix.shape)
#         random_kmeans = KMeans(n_clusters=k_val, n_init=10, random_state=42)
#         random_labels = random_kmeans.fit_predict(random_ref)
#         ref_disp = np.sum([np.mean(cdist(random_ref[random_labels == c],
#                                          np.mean(random_ref[random_labels == c], axis=0, keepdims=True)))
#                            for c in np.unique(random_labels)])
#         random_disps.append(ref_disp)
#     gap_value = np.log(np.mean(random_disps)) - np.log(actual_disp)
#     return gap_value
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

def compute_gap_statistic(feature_matrix, labels_k, k_val, n_refs=10):
    # Actual dispersion (within-cluster sum of squares)
    actual_disp = 0.0
    for c in np.unique(labels_k):
        cluster_points = feature_matrix[labels_k == c]
        cluster_mean = np.mean(cluster_points, axis=0, keepdims=True)
        actual_disp += np.sum(cdist(cluster_points, cluster_mean, 'euclidean') ** 2)

    # Reference dispersion
    random_disps = []
    for _ in range(n_refs):
        random_ref = np.random.uniform(np.min(feature_matrix, axis=0),
                                       np.max(feature_matrix, axis=0),
                                       feature_matrix.shape)
        random_kmeans = KMeans(n_clusters=k_val, n_init=10, random_state=42)
        random_labels = random_kmeans.fit_predict(random_ref)

        ref_disp = 0.0
        for c in np.unique(random_labels):
            cluster_points = random_ref[random_labels == c]
            cluster_mean = np.mean(cluster_points, axis=0, keepdims=True)
            ref_disp += np.sum(cdist(cluster_points, cluster_mean, 'euclidean') ** 2)

        random_disps.append(ref_disp)

    # Gap statistic
    gap_value = np.log(np.mean(random_disps)) - np.log(actual_disp)
    return gap_value

################## Data Processing ##################

def build_comp_list_from_clusters(byte_groups, labels):
    """
    Groups byte groups according to clustering labels.
    To account for slight differences in lengths, compute the minimum length among groups
    in a cluster and truncate each group to that length.
    """
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_dict[lab].append(i)
    comp_list = []
    for lab in sorted(cluster_dict.keys()):
        indices_in_cluster = cluster_dict[lab]
        if not indices_in_cluster:
            continue
        min_length = min(len(byte_groups[i]) for i in indices_in_cluster)
        arr2d = np.zeros((len(indices_in_cluster), min_length), dtype=np.uint8)
        for row_idx, grp_i in enumerate(indices_in_cluster):
            arr2d[row_idx] = byte_groups[grp_i][:min_length]
        comp_list.append(arr2d)
    return comp_list

def compress_data(data_set_list, compress_method, order='F'):
    compressed_data = []
    total_size = 0
    for cmp in data_set_list:
        bytes_ = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        c_ = compress_method(bytes_)
        compressed_data.append(c_)
        total_size += len(c_)
    return compressed_data, total_size
############################################################
def floats_df_to_int64_allcols(df, min_exp_span=2):
    """
    Apply numeric float64 → int64 conversion to all columns in the DataFrame.
    Ensures exponent range is widened first for each column.

    Parameters
    ----------
    df : pandas DataFrame
        All columns should be float64 (or convertible).
    min_exp_span : int
        Minimum exponent span to ensure before casting.

    Returns
    -------
    int_df : DataFrame with int64 values
    metas  : dict of {col_name: {'shift': ..., 'pow2': ...}} per column
    """
    int_data = {}
    metas = {}

    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64, copy=True)

        # 1. center
        median = np.median(x)
        x -= median

        # 2. ensure exponent span
        nz = x[np.nonzero(x)]
        exp_span = (np.ptp(np.floor(np.log2(np.abs(nz)))) if nz.size else 0)

        k = max(0, min_exp_span - int(exp_span))
        if k:
            x *= 2.0 ** k

        # 3. cast safely
        x = np.clip(x, -9.22e18, 9.22e18)
        int_data[col] = x.astype(np.int64)
        metas[col] = {'shift': float(median), 'pow2': int(k)}

    return pd.DataFrame(int_data), metas

################## Main Analysis Function ##################

def run_analysis(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder:", folder_path)
        return

    results_records = []

    comp_tools = {
        "zstd": zstd_comp,
       # 'fastlz': fastlz_compress,
        # "zlib": zlib_comp,
        #"bz2": bz2_comp,
    }

    # Define feature scenarios including all individual ones and the "All_Features" scenario.
    feature_scenarios = {
        "Entropy": extract_entropy,
        "Entropy_Mean": extract_entropy_mean,
        "Entropy_Std": extract_entropy_std,
        "Entropy_Max": extract_entropy_max,
        "Entropy_Min": extract_entropy_min,
        "Frequency": extract_frequency,
        "All_Features": extract_all_features
    }

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print("No .tsv files found in", folder_path)
        return

    for fname in tsv_files:
        dataset_name = os.path.splitext(fname)[0]
        fpath = os.path.join(folder_path, fname)
        print(f"Processing: {dataset_name}")

        try:
            df = pd.read_csv(fpath, sep='\t', header=None)
        except Exception as e:
            print("Failed to load", fname, e)
            continue
        df = df.iloc[1:]  # removes the first row
        df = df.reset_index(drop=True)  # optional: reset index after removal
        int_df, meta_info = floats_df_to_int64_allcols(df, min_exp_span=4)

        print(int_df.head())  # All columns are now int64
        print(meta_info)  # Shows shift + scaling per column


        numeric_vals = int_df
        # 2. Determine 10% of the total number of float32 values:
        # total_len = len(numeric_vals1)
        # portion_len = int(total_len * 0.2)  # 10% of the total length
        #
        # # 3. Slice the first 10%:
        # numeric_vals = numeric_vals1[:portion_len]
        #4.
        flattened = numeric_vals.to_numpy().flatten().tobytes()

        arr = np.frombuffer(flattened, dtype=np.uint8)

        # Split into interleaved groups (using a stride of 8 in this example).
        byte_groups = [arr[i::8] for i in range(8)]
        n_groups = len(byte_groups)

        # Standard compression on entire array.
        entire_arr_2d = [arr.reshape(1, -1)]

        for scenario_name, extractor in feature_scenarios.items():
            # Build feature matrix from byte groups.
            feature_list = [extractor(grp) for grp in byte_groups]
            feature_matrix = np.array(feature_list)
            if feature_matrix.shape[0] < 2:
                continue

            linked = linkage(feature_matrix, method='complete')
            max_k = min(8, feature_matrix.shape[0])
            for k_val in range(2, max_k + 1):
                labels_k = fcluster(linked, k_val, criterion='maxclust')

                try:
                    sil_val = silhouette_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                    db_score = davies_bouldin_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                    ch_score = calinski_harabasz_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                    gap_stat = compute_gap_statistic(feature_matrix, labels_k, k_val) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                except Exception as e:
                    sil_val, db_score, ch_score, gap_stat = -1, -1, -1, -1

                # Create a string representation of the clustering configuration.
                cluster_str = "|".join([f"({','.join(str(x) for x in np.where(labels_k == c)[0] + 1)})"
                                         for c in sorted(set(labels_k))])
                comp_list = build_comp_list_from_clusters(byte_groups, labels_k)

                for ctool_name, ctool_func in comp_tools.items():
                    # Standard compression (entire data).
                    _, full_comp_size = compress_data(entire_arr_2d, ctool_func)
                    std_ratio = ratio_or_inf(len(arr), full_comp_size)

                    # Decomposed compression using column-order.
                    _, dec_size = compress_data(comp_list, ctool_func, order='F')
                    dec_ratio = ratio_or_inf(len(arr), dec_size)

                    # Decomposed compression using row-order.
                    _, dec_size_row = compress_data(comp_list, ctool_func, order='C')
                    dec_ratio_row = ratio_or_inf(len(arr), dec_size_row)

                    results_records.append({
                        "Dataset": dataset_name,
                        "FeatureScenario": scenario_name,
                        "k": k_val,
                        "Silhouette": sil_val,
                        "DaviesBouldin": db_score,
                        "CalinskiHarabasz": ch_score,
                        "GapStatistic": gap_stat,
                        "ClusterConfig": cluster_str,
                        "CompressionTool": ctool_name,
                        "StandardSize(B)": full_comp_size,
                        "StandardRatio": std_ratio,
                        "DecomposedSize(B)_ColOrder": dec_size,
                        "DecomposedRatio_ColOrder": dec_ratio,
                        "DecomposedSize(B)_RowOrder": dec_size_row,
                        "DecomposedRatio_RowOrder": dec_ratio_row,
                    })

    df_results = pd.DataFrame(results_records)
    out_csv = os.path.join("/home/jamalids/Documents", "clustering_compression_results64.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")

if __name__ == "__main__":
    folder_path ="/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64/NEW"
    run_analysis(folder_path)
