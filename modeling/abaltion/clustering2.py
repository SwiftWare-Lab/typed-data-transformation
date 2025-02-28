#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score)


###############################################################################
#                           GAP STATISTIC HELPER FUNCTIONS                    #
###############################################################################

def within_cluster_dispersion(X, labels):
    """
    Compute within-cluster dispersion (sum of squared distances to cluster centroid).
    Used in Gap statistic computations.
    """
    total_disp = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            centroid = np.mean(cluster_points, axis=0)
            ssd = np.sum((cluster_points - centroid) ** 2)
            total_disp += ssd
    return total_disp


def compute_gap_statistic(feature_matrix, cluster_func, k, n_refs=5, random_seed=42):
    """
    Compute the Gap Statistic for a given k.
    Reference: Tibshirani, Walther, Hastie (2001).

    :param feature_matrix: (n_samples, n_features) data.
    :param cluster_func: A callable that takes (X, k) -> labels,
                        e.g., a lambda using fcluster on a linkage.
    :param k: Number of clusters (integer).
    :param n_refs: Number of reference datasets to sample.
    :param random_seed: For reproducible random generation.
    :return: gap value (float).
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    labels = cluster_func(feature_matrix, k)
    disp_original = within_cluster_dispersion(feature_matrix, labels)

    # Generate reference data
    mins = np.min(feature_matrix, axis=0)
    maxs = np.max(feature_matrix, axis=0)
    ref_disps = np.zeros(n_refs)

    for i in range(n_refs):
        ref_data = np.zeros_like(feature_matrix)
        for dim in range(feature_matrix.shape[1]):
            ref_data[:, dim] = np.random.uniform(mins[dim], maxs[dim],
                                                 size=feature_matrix.shape[0])
        ref_labels = cluster_func(ref_data, k)
        ref_disps[i] = within_cluster_dispersion(ref_data, ref_labels)

    # Gap = mean(log(disp_ref)) - log(disp_original)
    log_disp_ref_mean = np.mean(np.log(ref_disps + 1e-12))
    gap_value = log_disp_ref_mean - np.log(disp_original + 1e-12)
    return gap_value


###############################################################################
#                       CLUSTER METRIC EVALUATION + TIE DETECTION            #
###############################################################################

def evaluate_cluster_metrics(feature_matrix, linked,
                             min_clusters=1, max_clusters=4,
                             compute_gap=True, n_refs=5):
    """
    Evaluate Silhouette, Davies-Bouldin, Calinski-Harabasz, and Gap
    for k in [min_clusters..max_clusters].

    Returns a dict that includes:
      - single best k for each metric
      - all tie options for each metric
      - the raw scores for each k
    """

    silhouette_scores = {}
    db_scores = {}
    ch_scores = {}
    gap_scores = {}

    def cluster_func_for_gap(X, k):
        if X.shape == feature_matrix.shape:
            return fcluster(linked, k, criterion='maxclust')
        else:
            return np.ones(X.shape[0], dtype=int)

    for k in range(min_clusters, max_clusters + 1):
        labels = fcluster(linked, k, criterion='maxclust')

        # Silhouette
        if k >= 2 and feature_matrix.shape[0] > k:
            try:
                sil_val = silhouette_score(feature_matrix, labels)
            except:
                sil_val = np.nan
        else:
            sil_val = np.nan
        silhouette_scores[k] = sil_val

        # Davies-Bouldin
        if k >= 2 and feature_matrix.shape[0] > k:
            try:
                db_val = davies_bouldin_score(feature_matrix, labels)
            except:
                db_val = np.nan
        else:
            db_val = np.nan
        db_scores[k] = db_val

        # Calinski-Harabasz
        if k >= 2 and feature_matrix.shape[0] > k:
            try:
                ch_val = calinski_harabasz_score(feature_matrix, labels)
            except:
                ch_val = np.nan
        else:
            ch_val = np.nan
        ch_scores[k] = ch_val

        # Gap
        if compute_gap and k >= 1:
            gap_val = compute_gap_statistic(feature_matrix, cluster_func_for_gap, k,
                                            n_refs=n_refs)
        else:
            gap_val = np.nan
        gap_scores[k] = gap_val

    # find best single k
    def argmax_valid(d):
        valid = [(kk, vv) for kk, vv in d.items() if not np.isnan(vv)]
        if not valid:
            return None
        return max(valid, key=lambda x: x[1])[0]

    def argmin_valid(d):
        valid = [(kk, vv) for kk, vv in d.items() if not np.isnan(vv)]
        if not valid:
            return None
        return min(valid, key=lambda x: x[1])[0]

    # find all ties
    def argmax_all(d):
        valid = [(kk, vv) for kk, vv in d.items() if not np.isnan(vv)]
        if not valid:
            return []
        best_val = max(vv for (_, vv) in valid)
        return [kk for (kk, vv) in valid if vv == best_val]

    def argmin_all(d):
        valid = [(kk, vv) for kk, vv in d.items() if not np.isnan(vv)]
        if not valid:
            return []
        best_val = min(vv for (_, vv) in valid)
        return [kk for (kk, vv) in valid if vv == best_val]

    best_k_sil = argmax_valid(silhouette_scores)
    best_k_db = argmin_valid(db_scores)
    best_k_ch = argmax_valid(ch_scores)
    best_k_gap = argmax_valid(gap_scores)

    tie_sil = argmax_all(silhouette_scores)
    tie_db = argmin_all(db_scores)
    tie_ch = argmax_all(ch_scores)
    tie_gap = argmax_all(gap_scores)

    return {
        'silhouette': silhouette_scores,
        'davies_bouldin': db_scores,
        'calinski_harabasz': ch_scores,
        'gap': gap_scores,

        'best_k_silhouette': best_k_sil,
        'best_k_db': best_k_db,
        'best_k_ch': best_k_ch,
        'best_k_gap': best_k_gap,

        'tie_silhouette': tie_sil,
        'tie_db': tie_db,
        'tie_ch': tie_ch,
        'tie_gap': tie_gap,
    }


###############################################################################
#    PLOT 2x2 METRICS, SHOW TIES, PLUS CLUSTER LABELS FOR EACH TIED K         #
###############################################################################

def plot_cluster_metrics_fixed_k(metric_dict, dataset_name, save_path,
                                 best_k_sil, best_k_db, best_k_ch, best_k_gap,
                                 final_k, cluster_labels, feature_matrix, linked):
    """
    Create bar plots (2x2) for Silhouette, DB, CH, and Gap with x-axis = [1,2,3,4].
    Annotate with:
      - best k for each metric
      - final chosen k
      - cluster labels for final_k
      - TIE k-values (and their labels) for each metric
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ks = [1, 2, 3, 4]

    silhouette_dict = metric_dict['silhouette']
    db_dict = metric_dict['davies_bouldin']
    ch_dict = metric_dict['calinski_harabasz']
    gap_dict = metric_dict['gap']

    tie_sil = metric_dict['tie_silhouette']  # e.g. [2, 3]
    tie_db = metric_dict['tie_db']
    tie_ch = metric_dict['tie_ch']
    tie_gap = metric_dict['tie_gap']

    # A helper to produce labels for each tie K
    def compute_labels_for_ties(tie_list):
        """
        For each k in tie_list, compute cluster labels,
        return a dict k -> string, e.g. {2: '[1 1 2 2]', 3: '[1 2 2 3]'}
        """
        tie_labels = {}
        for k_ in tie_list:
            labs_ = fcluster(linked, k_, criterion='maxclust')
            # Convert numeric labels to a short string, truncating if large:
            if len(labs_) > 12:
                short_str = np.array2string(labs_[:12], separator=' ', max_line_width=999)
                short_str += " ... (truncated)"
            else:
                short_str = np.array2string(labs_, separator=' ', max_line_width=999)
            tie_labels[k_] = short_str
        return tie_labels

    # Precompute tie label dicts for each metric
    tie_sil_labels = compute_labels_for_ties(tie_sil)
    tie_db_labels = compute_labels_for_ties(tie_db)
    tie_ch_labels = compute_labels_for_ties(tie_ch)
    tie_gap_labels = compute_labels_for_ties(tie_gap)

    # Convert best_k_sil + tie_sil into a string, e.g. "2 (ties: [2, 3])"
    # We'll also show the cluster labels for each tie K
    def fmt_metric_line(metric_name, single_k, tie_ks, tie_k_labels_dict):
        """
        e.g. 'Silhouette => 2 (ties: [2, 3])\n  tie cluster labels:\n    K=2 => [1 1 2 2]\n    K=3 => [1 2 2 3]'
        """
        if single_k is None:
            return f"{metric_name} => None"

        # Build tie info if there's more than 1 in tie_ks
        if len(tie_ks) > 1:
            # show all tie K plus their label arrays
            lines = [f"{metric_name} => {single_k} (ties: {tie_ks})", "  tie cluster labels:"]
            for k_ in tie_ks:
                lines.append(f"    K={k_} => {tie_k_labels_dict[k_]}")
            return "\n".join(lines)
        else:
            # No tie or single tie => just show the single best
            return f"{metric_name} => {single_k}"

    line_sil = fmt_metric_line("Silhouette", best_k_sil, tie_sil, tie_sil_labels)
    line_db = fmt_metric_line("Davies-Bouldin", best_k_db, tie_db, tie_db_labels)
    line_ch = fmt_metric_line("Calinski-Harabasz", best_k_ch, tie_ch, tie_ch_labels)
    line_gap = fmt_metric_line("Gap Statistic", best_k_gap, tie_gap, tie_gap_labels)

    # Build y-values for each metric in the order k=1..4, defaulting to NaN if missing
    silhouette_vals = [silhouette_dict.get(k, np.nan) for k in ks]
    db_vals = [db_dict.get(k, np.nan) for k in ks]
    ch_vals = [ch_dict.get(k, np.nan) for k in ks]
    gap_vals = [gap_dict.get(k, np.nan) for k in ks]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # ---------- (1) Silhouette (Higher=Better) ----------
    ax = axes[0]
    bars = ax.bar(ks, silhouette_vals, color='C0')
    ax.set_title("Silhouette (Higher=Better)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(ks)
    ax.set_xticklabels(str(k) for k in ks)
    for x_pos, val in zip(ks, silhouette_vals):
        if not np.isnan(val):
            ax.text(x_pos, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    # ---------- (2) Davies-Bouldin (Lower=Better) ----------
    ax = axes[1]
    bars = ax.bar(ks, db_vals, color='C1')
    ax.set_title("Davies-Bouldin (Lower=Better)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("DB Index")
    ax.set_xticks(ks)
    ax.set_xticklabels(str(k) for k in ks)
    for x_pos, val in zip(ks, db_vals):
        if not np.isnan(val):
            ax.text(x_pos, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    # ---------- (3) Calinski-Harabasz (Higher=Better) ----------
    ax = axes[2]
    bars = ax.bar(ks, ch_vals, color='C2')
    ax.set_title("Calinski-Harabasz (Higher=Better)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("CH Index")
    ax.set_xticks(ks)
    ax.set_xticklabels(str(k) for k in ks)
    for x_pos, val in zip(ks, ch_vals):
        if not np.isnan(val):
            ax.text(x_pos, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    # ---------- (4) Gap Statistic (Higher=Better) ----------
    ax = axes[3]
    bars = ax.bar(ks, gap_vals, color='C3')
    ax.set_title("Gap Statistic (Higher=Better)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Gap Value")
    ax.set_xticks(ks)
    ax.set_xticklabels(str(k) for k in ks)
    for x_pos, val in zip(ks, gap_vals):
        if not np.isnan(val):
            ax.text(x_pos, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    # ---------- Prepare final annotation text ----------
    # We also want to show cluster labels for final_k
    # The user might also want to see the final cluster_labels array
    # which we already have as "cluster_labels"
    # We'll convert it to a short string too, in case it's large.
    def short_label_str(labs):
        if len(labs) > 12:
            return np.array2string(labs[:12], separator=' ') + " ... (truncated)"
        else:
            return np.array2string(labs, separator=' ')

    final_labels_str = short_label_str(cluster_labels)

    ann_text = (
        "===== Best k for each metric =====\n"
        f"{line_sil}\n"
        f"{line_db}\n"
        f"{line_ch}\n"
        f"{line_gap}\n"
        "==================================\n\n"
        f"Final chosen k = {final_k}\n"
        f"Cluster labels (final_k) = {final_labels_str}\n"
    )

    plt.figtext(
        0.5, 0.01, ann_text,
        ha='center', va='bottom',
        fontsize=10, wrap=True,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.suptitle(f"Cluster Metrics for {dataset_name}", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster metrics plot saved at '{save_path}'")


###############################################################################
#               DETERMINE OPTIMAL k AND (OPTIONALLY) RETURN LABELS            #
###############################################################################

def determine_optimal_clusters(linked, feature_matrix,
                               min_clusters=1, max_clusters=4):
    """
    Determine the best k for Silhouette, DB, CH, and Gap, then pick a final k.
    Returns (final_k, metrics_dict).

    Ties are stored in metrics_dict['tie_silhouette'], etc.
    """
    metrics_dict = evaluate_cluster_metrics(
        feature_matrix, linked,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        compute_gap=True,
        n_refs=5
    )

    # Single best
    best_k_sil = metrics_dict['best_k_silhouette']
    best_k_db = metrics_dict['best_k_db']
    best_k_ch = metrics_dict['best_k_ch']
    best_k_gap = metrics_dict['best_k_gap']

    # Example logic to pick final_k (fallback=2 if everything is None)
    final_k = 2
    if best_k_sil is not None:
        final_k = best_k_sil
    elif best_k_gap is not None:
        final_k = best_k_gap
    elif best_k_ch is not None:
        final_k = best_k_ch
    elif best_k_db is not None:
        final_k = best_k_db

    return final_k, metrics_dict


###############################################################################
#              BASIC ENTROPY + FEATURE EXTRACTION (AS PER YOUR EXAMPLE)       #
###############################################################################

def compute_entropy(data_window):
    """Compute the Shannon entropy of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_entropy_over_data(data, window_size=256):
    """
    Slide a window of size `window_size` across data and compute entropy for each window.
    """
    entropies = []
    data_len = len(data)
    for start_idx in range(0, data_len - window_size + 1, window_size):
        window = data[start_idx:start_idx + window_size]
        ent = compute_entropy(window)
        entropies.append(ent)
    return entropies


def extract_features(byte_group, window_size=256):
    """
    Example feature extraction: average/std/max/min entropies + byte frequency (256).
    """
    entropies = calculate_entropy_over_data(byte_group, window_size)
    if len(entropies) > 0:
        avg_ent = np.mean(entropies)
        std_ent = np.std(entropies)
        max_ent = np.max(entropies)
        min_ent = np.min(entropies)
    else:
        avg_ent = std_ent = max_ent = min_ent = 0.0

    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])

    features = np.concatenate(([avg_ent, std_ent, max_ent, min_ent], byte_freq))
    return features


###############################################################################
#          HIERARCHICAL CLUSTERING HELPER (SIMPLE "complete" LINKAGE)         #
###############################################################################

def perform_hierarchical_clustering(feature_matrix, method='complete'):
    """
    Perform hierarchical clustering with SciPy linkage.
    :param feature_matrix: (n_samples, n_features).
    :param method: 'single', 'complete', 'average', 'ward', ...
    :return: Linkage matrix.
    """
    linked = linkage(feature_matrix, method=method)
    return linked


def plot_dendrogram_custom(linked, labels, save_path):
    """
    Plot and save a dendrogram.
    """
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, orientation='top',
               distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Groups')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Dendrogram saved at {save_path}")


###############################################################################
#                       COMPRESSION & ORDERING EXAMPLE                        #
###############################################################################

def group_and_reorder(byte_groups, cluster_labels):
    """
    Return the reordered list of group indices (1-based), sorted by cluster label then group index.
    """
    group_info = []
    for idx, label in enumerate(cluster_labels):
        group_info.append({'group_index': idx + 1, 'cluster_label': label})

    sorted_groups = sorted(group_info, key=lambda x: (x['cluster_label'], x['group_index']))
    ordered_indices = [g['group_index'] for g in sorted_groups]
    return ordered_indices


def compress_and_evaluate(byte_groups, ordered_indices):
    """
    Reorder groups, merge, and compress with zlib. Return (ratio, original_size, compressed_size).
    """
    merged_bytes = b''.join(byte_groups[i - 1].tobytes() for i in ordered_indices)
    compressed_data = zlib.compress(merged_bytes)
    orig_size = len(merged_bytes)
    comp_size = len(compressed_data)
    ratio = comp_size / orig_size if orig_size > 0 else 1.0
    return ratio, orig_size, comp_size


###############################################################################
#                   EXAMPLE MAIN "run_analysis" FUNCTION                      #
###############################################################################

def run_analysis(folder_path):
    """
    Example workflow: load each .tsv, compute features, hierarchical clustering,
    pick best k, reorder, compress, and plot metrics (showing tie info + tie labels).
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'.\n")

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        dataset_name = os.path.splitext(tsv_file)[0]

        print("=" * 30)
        print(f"Processing: {tsv_file}")
        print("=" * 30)

        # Load dataset
        try:
            ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
        except Exception as e:
            print(f"Error loading {tsv_file}: {e}")
            continue

        # For demonstration, assume first column is an ID, skip it
        byte_columns = ts_data.columns[1:]
        data_values = ts_data[byte_columns].to_numpy().astype(np.float64)

        # Flatten to bytes
        byte_data = data_values.flatten(order='F').tobytes()

        # Split into 4 groups
        component_sizes = [1, 1, 1, 1]
        byte_groups = split_into_components(byte_data, component_sizes=component_sizes)

        # Extract features
        features = []
        for grp in byte_groups:
            feat = extract_features(grp, window_size=65536)
            features.append(feat)
        feature_matrix = np.array(features)

        # Hierarchical clustering
        linked = perform_hierarchical_clustering(feature_matrix, method='complete')
        dendro_path = f"/home/jamalids/Documents/cluster2/{dataset_name}_dendrogram.png"
        plot_dendrogram_custom(linked, [f"G{i + 1}" for i in range(len(byte_groups))], dendro_path)

        # Find best k
        optimal_k, metrics_dict = determine_optimal_clusters(linked, feature_matrix,
                                                             min_clusters=1, max_clusters=4)
        # Get final labels
        cluster_labels = fcluster(linked, optimal_k, criterion='maxclust')

        # Reorder groups
        ordered_indices = group_and_reorder(byte_groups, cluster_labels)

        # Evaluate compression with new order
        best_ratio, orig_size, comp_size = compress_and_evaluate(byte_groups, ordered_indices)
        print(f"Best grouping => ratio={best_ratio:.4f} (orig={orig_size}, comp={comp_size})")

        # Compare to original ordering
        original_indices = [1, 2, 3, 4]
        orig_ratio, _, _ = compress_and_evaluate(byte_groups, original_indices)
        print(f"Original ordering => ratio={orig_ratio:.4f}")

        # Now plot the 2x2 metrics, tie info, + show each tie's cluster labels
        metrics_plot_path = f"/home/jamalids/Documents/cluster2/{dataset_name}_cluster_metrics.png"
        plot_cluster_metrics_fixed_k(
            metric_dict=metrics_dict,
            dataset_name=dataset_name,
            save_path=metrics_plot_path,
            best_k_sil=metrics_dict['best_k_silhouette'],
            best_k_db=metrics_dict['best_k_db'],
            best_k_ch=metrics_dict['best_k_ch'],
            best_k_gap=metrics_dict['best_k_gap'],
            final_k=optimal_k,
            cluster_labels=cluster_labels,
            feature_matrix=feature_matrix,
            linked=linked
        )

        print(f"Done with {tsv_file}\n")

    print("All analyses completed.")


def split_into_components(byte_data, component_sizes):
    """
    Helper: break the raw byte_data into separate numpy arrays, each capturing
    selected bytes according to the 'component_sizes' pattern.
    """
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        component = arr[offset::total_bytes]
        components.append(component)
    return components


###############################################################################
#                          SCRIPT ENTRY POINT                                 #
###############################################################################

if __name__ == "__main__":
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)
