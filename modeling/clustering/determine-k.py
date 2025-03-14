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

# <-- Adjust to your own library path
# e.g. if your code is at "modeling/compression_tools.py"
from modeling.compression_tools import zstd_comp  # or add more if you like

##############################################
#         Decomposed vs. Reordered
##############################################

def compress_and_sum(byte_groups, compress_method):
    """
    DECOMPOSED approach:
      Compress each cluster in 'byte_groups' individually,
      sum up the compressed sizes.
    Returns (total_compressed_size, compression_ratio).
    """
    # Merge them to know the original_size
    merged = b''.join([grp.tobytes() if hasattr(grp, 'tobytes') else grp for grp in byte_groups])
    original_size = len(merged)

    total_comp_size = 0
    for grp in byte_groups:
        grp_bytes = grp.tobytes() if hasattr(grp, 'tobytes') else grp
        c = compress_method(grp_bytes)
        total_comp_size += len(c)

    ratio = (original_size / total_comp_size) if total_comp_size != 0 else float('inf')
    return total_comp_size, ratio


def compress_concat(byte_groups, compress_method):
    """
    REORDERED approach:
      Concatenate all groups in 'byte_groups' in the new order,
      compress as one single chunk.
    Returns (compressed_size, compression_ratio).
    """
    merged = b''.join([grp.tobytes() if hasattr(grp, 'tobytes') else grp for grp in byte_groups])
    original_size = len(merged)
    comp = compress_method(merged)
    comp_size = len(comp)
    ratio = (original_size / comp_size) if comp_size != 0 else float('inf')
    return comp_size, ratio


##############################################
#           Entropy / Feature Extraction
##############################################
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
    if entropies:
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
    if entropies:
        return np.array([
            np.mean(entropies),
            np.std(entropies),
            np.max(entropies),
            np.min(entropies)
        ])
    else:
        return np.array([0,0,0,0])

def extract_only_frequency(byte_group, window_size=65536):
    freq_counter = Counter(byte_group)
    byte_freqs = np.array([freq_counter.get(i, 0) / len(byte_group) for i in range(256)])
    freq_std = np.std(byte_freqs)
    return np.array([freq_std])


##############################################
#         Hierarchical Clustering
##############################################
def perform_hierarchical_clustering(feature_matrix, method='complete'):
    return linkage(feature_matrix, method=method)

def reorder_groups_by_labels(byte_groups, labels):
    """
    Reorder the list of groups by ascending cluster label (and by original index).
    Returns (reordered_groups, cluster_config_str)
        e.g. cluster_config_str => "(1,2)|(3)|(4,5)"
    """
    labeled = list(zip(range(len(byte_groups)), labels))  # (orig_idx, cluster_label)
    labeled_sorted = sorted(labeled, key=lambda x: (x[1], x[0]))

    reordered = []
    cluster_dict = {}
    for (orig_idx, c_label) in labeled_sorted:
        reordered.append(byte_groups[orig_idx])
        cluster_dict.setdefault(c_label, []).append(orig_idx + 1)  # +1 for 1-based group IDs

    cluster_str_parts = []
    for c_label in sorted(cluster_dict.keys()):
        members = cluster_dict[c_label]
        cluster_str_parts.append("({})".format(",".join(str(x) for x in members)))
    cluster_config_str = "|".join(cluster_str_parts)

    return reordered, cluster_config_str


##############################################
#        Splitting Bytes into Groups
##############################################
def split_bytes_into_components(byte_array, component_sizes=[1,1,1,1]):
    """
    Example: if we have a big byte array, we split it into 4 interleaved groups.
    """
    arr = np.frombuffer(byte_array, dtype=np.uint8)
    comps = []
    num_components = len(component_sizes)
    for i in range(num_components):
        c = arr[i::num_components]
        comps.append(c)
    return comps


##############################################
#        Main Workflow with "Auto K"
##############################################
def run_analysis(folder_path):
    """
    For each .tsv file:
      1) Flatten => split into 4 groups.
      2) For each FeatureScenario, do hierarchical clustering => for k in [2..4].
      3) Reorder => measure decomposed ratio & reordered ratio => store in CSV.
      4) Mark best k by "max decomposed ratio".
    """

    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    results_records = []

    # We define only zstd_comp for demonstration, from your modeling.compression_tools
    comp_tools = {
        "zstd": zstd_comp,
        # You can add more if you want
    }

    feature_scenarios = {
        "All Features (5D)": extract_5_features,
        "Only Entropy (4D)": extract_only_entropy,
        "Only Freq (1D)": extract_only_frequency
    }

    # find .tsv files
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in '{folder_path}'.")
        return

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        dataset_name = os.path.splitext(tsv_file)[0]
        print(f"\n===== Processing Dataset: {dataset_name} =====")

        # Load data
        try:
            df = pd.read_csv(dataset_path, sep='\t', header=None)
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
            continue

        # Flatten numeric data as bytes
        # skip col 0 if not relevant
        byte_cols = df.columns[1:]
        numeric_vals = df[byte_cols].to_numpy().astype(np.float64)
        flattened_bytes = numeric_vals.flatten(order='F').tobytes()

        # Split into 4 groups
        byte_groups = split_bytes_into_components(flattened_bytes, [1,1,1,1])
        n_groups = len(byte_groups)
        if n_groups < 2:
            print("Fewer than 2 groups => skipping.")
            continue

        # Measure "original" scenario (k=0 => no clustering)
        # We'll do "decomposed" = compress each group as-is, plus "reordered" = single chunk
        for tool_name, comp_func in comp_tools.items():
            dec_size, dec_ratio = compress_and_sum(byte_groups, comp_func)
            ro_size, ro_ratio = compress_concat(byte_groups, comp_func)
            results_records.append({
                "Dataset": dataset_name,
                "FeatureScenario": "Original",
                "k": 0,
                "Silhouette": None,
                "ClusterConfig": "(1,2,3,4) no clustering",
                "CompressionTool": tool_name,
                "DecomposedSize(B)": dec_size,
                "DecomposedRatio": dec_ratio,
                "ReorderedSize(B)": ro_size,
                "ReorderedRatio": ro_ratio,
                "BestByDecomposedRatio": False
            })

        # Now do each feature scenario
        for scenario_name, feature_extractor in feature_scenarios.items():
            # Build feature matrix
            feature_list = [feature_extractor(grp) for grp in byte_groups]
            feature_matrix = np.array(feature_list)
            if feature_matrix.shape[0] < 2:
                continue

            linked = perform_hierarchical_clustering(feature_matrix, 'complete')

            # For k=2..(up to 4 or n_groups)
            max_k = min(n_groups, 4)
            scenario_rows = []
            for k_val in range(2, max_k+1):
                labels_k = fcluster(linked, k_val, criterion='maxclust')

                # compute silhouette if valid
                unique_labels = np.unique(labels_k)
                if (len(unique_labels) >= 2) and (len(unique_labels) <= (n_groups - 1)):
                    try:
                        sil_val = silhouette_score(feature_matrix, labels_k)
                    except:
                        sil_val = -1
                else:
                    sil_val = -1

                # reorder
                reordered_groups, config_str = reorder_groups_by_labels(byte_groups, labels_k)

                # measure decomposed + reordered for each compression tool
                for tool_name, comp_func in comp_tools.items():
                    dec_size, dec_ratio = compress_and_sum(reordered_groups, comp_func)
                    ro_size, ro_ratio = compress_concat(reordered_groups, comp_func)

                    row = {
                        "Dataset": dataset_name,
                        "FeatureScenario": scenario_name,
                        "k": k_val,
                        "Silhouette": sil_val,
                        "ClusterConfig": config_str,
                        "CompressionTool": tool_name,
                        "DecomposedSize(B)": dec_size,
                        "DecomposedRatio": dec_ratio,
                        "ReorderedSize(B)": ro_size,
                        "ReorderedRatio": ro_ratio,
                        "BestByDecomposedRatio": False
                    }
                    scenario_rows.append(row)

            if scenario_rows:
                df_scenario = pd.DataFrame(scenario_rows)
                # We'll pick best k by the highest decomposed ratio, per (Dataset,FeatureScenario,CompressionTool)
                group_cols = ["Dataset","FeatureScenario","CompressionTool"]
                grouped = df_scenario.groupby(group_cols)
                for grp_key, sub_df in grouped:
                    best_idx = sub_df["DecomposedRatio"].idxmax()
                    df_scenario.at[best_idx, "BestByDecomposedRatio"] = True

                scenario_records = df_scenario.to_dict(orient="records")
                results_records.extend(scenario_records)

    # Convert to DF & save
    results_df = pd.DataFrame(results_records)
    out_path = os.path.join(folder_path, "clustering_compression_results_AUTO_K.csv")
    results_df.to_csv(out_path, index=False)
    print(f"All done! Results saved at: {out_path}")
    print(results_df.head(30))


if __name__ == "__main__":
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)
