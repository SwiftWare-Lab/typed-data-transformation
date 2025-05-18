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
from sklearn.metrics import davies_bouldin_score

# Import custom compression utilities
from modeling.utils import compute_entropy, find_max_consecutive_similar_values
from modeling.compression_tools import zstd_comp

################## Compression Helper ##################

def ratio_or_inf(orig_size, comp_size):
    return float(orig_size) / comp_size if comp_size else float('inf')

################## Feature Extraction ##################

def extract_entropy(byte_group):
    freq = Counter(byte_group)
    total = len(byte_group)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return np.array([entropy])

def extract_frequency(byte_group):
    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    return np.array([np.std(byte_freqs)])

def extract_entropy_features(byte_group, window_size=65536):
    entropies = []
    for start_idx in range(0, len(byte_group), window_size):
        window = byte_group[start_idx:start_idx + window_size]
        entropies.append(compute_entropy(window))
    return np.array([np.mean(entropies), np.std(entropies), np.max(entropies), np.min(entropies)])

def extract_all_features(byte_group, window_size=65536):
    ent_features = extract_entropy_features(byte_group, window_size)
    freq_feature = extract_frequency(byte_group)
    return np.concatenate([ent_features, freq_feature])

################## Data Processing ##################

def build_comp_list_from_clusters(byte_groups, labels):
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
    return comp_list, cluster_dict

def compress_data(data_set_list, compress_method, order='F'):
    compressed_data = []
    total_size = 0
    all_bytes = bytearray()
    for cmp in data_set_list:
        all_bytes.extend(cmp.flatten(order))
    compressed = compress_method(np.frombuffer(all_bytes, dtype=np.byte))
    total_size = len(compressed)
    return compressed, total_size

################## Entropy Metrics ##################

def compute_joint_entropy(groups):
    combined = list(zip(*groups))
    freq = Counter(combined)
    total = len(combined)
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in freq.values())

def compute_mutual_information(groups):
    H_joint = compute_joint_entropy(groups)
    H_individual = sum(compute_entropy(g) for g in groups)
    return H_individual - H_joint

def compute_kth_entropy(byte_group, k=2):
    if len(byte_group) < k:
        return 0
    freq = Counter(tuple(byte_group[i:i+k]) for i in range(len(byte_group)-k+1))
    total = len(byte_group) - k + 1
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in freq.values())

################## Main Analysis Function ##################

def run_analysis(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder:", folder_path)
        return

    results_records = []
    comp_tools = {"zstd": zstd_comp}
    feature_scenarios = {
        "Entropy": extract_entropy,
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

        numeric_vals = df.values[1:500000, 1].astype(np.float32)
        arr = np.frombuffer(numeric_vals.tobytes(), dtype=np.uint8)
        byte_groups = [arr[i::4] for i in range(4)]
        entire_arr_2d = [arr.reshape(1, -1)]

        for scenario_name, extractor in feature_scenarios.items():
            feature_list = [extractor(grp) for grp in byte_groups]
            feature_matrix = np.array(feature_list)
            linked = linkage(feature_matrix, method='complete')
            max_k = feature_matrix.shape[0]

            for k_val in range(1, max_k + 1):
                if k_val == 1:
                    labels_k = np.ones(feature_matrix.shape[0], dtype=int)
                else:
                    labels_k = fcluster(linked, k_val, criterion='maxclust')

                n_labels = len(np.unique(labels_k))
                db_score = davies_bouldin_score(feature_matrix, labels_k) if 2 <= n_labels < feature_matrix.shape[0] else -1

                comp_list, cluster_dict = build_comp_list_from_clusters(byte_groups, labels_k)
                decomp_groups = [grp.flatten(order='F') for grp in comp_list]

                cluster_config_str = "|".join([f"({','.join(str(i + 1) for i in cluster_dict[c])})" for c in sorted(cluster_dict)])

                inner_stds = []
                cluster_means = []
                full_entropies = []
                for lab in sorted(cluster_dict.keys()):
                    indices = cluster_dict[lab]
                    if not indices:
                        continue
                    entropies = [compute_entropy(byte_groups[i]) for i in indices]
                    inner_stds.append(np.std(entropies))
                    cluster_means.append(np.mean(entropies))
                    full_entropies.extend(entropies)
                inner_cluster_std = np.mean(inner_stds)
                between_cluster_std = np.std(cluster_means)
                full_entropy_std = np.std(full_entropies)

                joint_entropy = compute_joint_entropy(decomp_groups)
                mutual_info = compute_mutual_information(decomp_groups)
                std_entropy = np.std([compute_entropy(g) for g in decomp_groups])
                kth_entropy = np.mean([compute_kth_entropy(g) for g in decomp_groups])

                for ctool_name, ctool_func in comp_tools.items():
                    _, full_comp_size = compress_data(entire_arr_2d, ctool_func)
                    std_ratio = ratio_or_inf(len(arr), full_comp_size)
                    _, dec_size = compress_data(comp_list, ctool_func, order='F')
                    dec_ratio = ratio_or_inf(len(arr), dec_size)

                    results_records.append({
                        "Dataset": dataset_name,
                        "FeatureScenario": scenario_name,
                        "k": k_val,
                        "ClusterConfig": cluster_config_str,
                        "DaviesBouldin": db_score,
                        "CompressionTool": ctool_name,
                        "StandardRatio": std_ratio,
                        "DecomposedRatio_ColOrder": dec_ratio,
                        "JointEntropy": joint_entropy,
                        "MutualInfo": mutual_info,
                        "StdEntropy": std_entropy,
                        "KthEntropy": kth_entropy,
                        "InnerClusterStd": inner_cluster_std,
                        "BetweenClusterStd": between_cluster_std,
                        "FullClusterStd": full_entropy_std
                    })

    df_results = pd.DataFrame(results_records)
    out_csv = os.path.join("/home/jamalids/Documents", "clustering_compression_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")

if __name__ == "__main__":
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)