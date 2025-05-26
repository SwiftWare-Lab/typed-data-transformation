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
from modeling.compression_tools import zstd_comp, zlib_comp, bz2_comp

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
def extract_delta(global_stream, byte_group):
    """
    For one byte_group:
      ΔH0 = H0(global_stream) - (|byte_group|/|global_stream|) * H0(byte_group)
    Returns a 1D array so it fits into your feature matrix.
    """
    total = len(global_stream)
    H0_global = compute_entropy(global_stream)
    H0_grp = compute_entropy(byte_group)
    weighted_H0 = (len(byte_group) / total) * H0_grp
    return np.array([H0_global - weighted_H0])




################## Clustering Metrics ##################

def compute_gap_statistic(feature_matrix, labels_k, k_val, n_refs=10):
    actual_disp = np.sum([np.mean(cdist(feature_matrix[labels_k == c],
                                        np.mean(feature_matrix[labels_k == c], axis=0, keepdims=True)))
                          for c in np.unique(labels_k)])
    random_disps = []
    for _ in range(n_refs):
        random_ref = np.random.uniform(np.min(feature_matrix, axis=0),
                                       np.max(feature_matrix, axis=0),
                                       feature_matrix.shape)
        random_kmeans = KMeans(n_clusters=k_val, n_init=10, random_state=42)
        random_labels = random_kmeans.fit_predict(random_ref)
        ref_disp = np.sum([np.mean(cdist(random_ref[random_labels == c],
                                         np.mean(random_ref[random_labels == c], axis=0, keepdims=True)))
                           for c in np.unique(random_labels)])
        random_disps.append(ref_disp)
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

# ################## Main Analysis Function ##################
#def run_analysis(folder_path, window_size=1048576):
def run_analysis(folder_path, window_size=262144):
    if not os.path.isdir(folder_path):
        print("Invalid folder:", folder_path)
        return

    comp_tools = {
        "zstd": zstd_comp,
        # "zlib": zlib_comp,
        # "bz2": bz2_comp,
    }

    feature_scenarios = {
        "Entropy": extract_entropy,
        "Entropy_Mean": extract_entropy_mean,
        "Entropy_Std": extract_entropy_std,
        "Entropy_Max": extract_entropy_max,
        "Entropy_Min": extract_entropy_min,
        "Frequency": extract_frequency,
        "All_Features": extract_all_features,
        "Delta": extract_delta
    }

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print("No .tsv files found in", folder_path)
        return

    for fname in tsv_files:
        dataset_name = os.path.splitext(fname)[0]
        fpath = os.path.join(folder_path, fname)
        print(f"\nProcessing dataset: {dataset_name}")

        try:
            df = pd.read_csv(fpath, sep='\t', header=None)
        except Exception as e:
            print("  Failed to load", fname, e)
            continue

        numeric_vals = df.values[:, 1].astype(np.float32)
        flattened   = numeric_vals.flatten().tobytes()
        arr         = np.frombuffer(flattened, dtype=np.uint8)

        # Determine number of full blocks
        num_blocks = max(1, len(arr) // window_size)

        for block_idx in range(num_blocks):
            start    = block_idx * window_size
            end      = start + window_size
            block_arr = arr[start:end]

            print(f"  Block {block_idx+1}/{num_blocks} (bytes {start}-{end}):")

            # Prepare streams & groups
            global_stream = block_arr.copy()
            byte_groups   = [block_arr[i::4] for i in range(4)]
            entire_arr_2d = [block_arr.reshape(1, -1)]

            records = []
            for scenario_name, extractor in feature_scenarios.items():
                # build feature matrix
                if scenario_name == "Delta":
                    feature_list = [extractor(global_stream, grp) for grp in byte_groups]
                else:
                    feature_list = [extractor(grp) for grp in byte_groups]
                feature_matrix = np.array(feature_list)
                if feature_matrix.shape[0] < 2:
                    continue

                linked = linkage(feature_matrix, method='complete')
                max_k  = min(4, feature_matrix.shape[0])

                for k_val in range(2, max_k + 1):
                    labels_k = fcluster(linked, k_val, criterion='maxclust')
                    # clustering metrics
                    try:
                        sil = silhouette_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                        db  = davies_bouldin_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                        ch  = calinski_harabasz_score(feature_matrix, labels_k) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                        gap = compute_gap_statistic(feature_matrix, labels_k, k_val) if 2 <= len(set(labels_k)) < len(feature_matrix) else -1
                    except:
                        sil, db, ch, gap = -1, -1, -1, -1

                    cluster_str = "|".join(
                        f"({','.join(str(i) for i in np.where(labels_k == c)[0] + 1)})"
                        for c in sorted(set(labels_k))
                    )
                    comp_list = build_comp_list_from_clusters(byte_groups, labels_k)

                    for ctool_name, ctool_func in comp_tools.items():
                        # standard
                        _, full_size = compress_data(entire_arr_2d, ctool_func)
                        # decomposed column‐order
                        _, col_size  = compress_data(comp_list, ctool_func, order='F')
                        # decomposed row‐order
                        _, row_size  = compress_data(comp_list, ctool_func, order='C')

                        records.append({
                            "Dataset": dataset_name,
                            "BlockIdx": block_idx,
                            "FeatureScenario": scenario_name,
                            "k": k_val,
                            "Silhouette": sil,
                            "DaviesBouldin": db,
                            "CalinskiHarabasz": ch,
                            "GapStatistic": gap,
                            "ClusterConfig": cluster_str,
                            "CompressionTool": ctool_name,
                            "StandardSize(B)": full_size,
                            "StandardRatio": ratio_or_inf(len(block_arr), full_size),
                            "DecomposedSize(B)_ColOrder": col_size,
                            "DecomposedRatio_ColOrder": ratio_or_inf(len(block_arr), col_size),
                            "DecomposedSize(B)_RowOrder": row_size,
                            "DecomposedRatio_RowOrder": ratio_or_inf(len(block_arr), row_size),
                        })

            # write per‐block CSV
            df_block = pd.DataFrame(records)
            out_csv = os.path.join(folder_path,
                                   f"{dataset_name}-block{block_idx}.csv")
            df_block.to_csv(out_csv, index=False)
            print(f"    → saved: {out_csv}")

if __name__ == "__main__":
    folder_path =  '/mnt/c/Users/jamalids/Downloads/dataset/HPC/block2'
    run_analysis(folder_path)
