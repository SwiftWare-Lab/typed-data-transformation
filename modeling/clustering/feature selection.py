import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import zlib
import matplotlib.cm as cm

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples


# -------------------- Helper: Plot Silhouette -------------------- #
def plot_silhouette(feature_matrix, labels,
                    dataset_name, scenario, k, save_dir):
    """
    Create and save a silhouette plot for the given cluster labeling.
    Checks if the number of clusters is valid for silhouette.
    If invalid, we skip rather than crash.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = feature_matrix.shape[0]

    # silhouette is only valid if 2 <= n_clusters <= n_samples - 1
    if n_clusters < 2 or n_clusters > n_samples - 1:
        print(f"Skipping silhouette plot for k={k} in {scenario}: "
              f"{n_clusters} cluster(s), {n_samples} samples.")
        return

    # Calculate silhouette samples & average
    silhouette_avg = silhouette_score(feature_matrix, labels)
    sample_sil_values = silhouette_samples(feature_matrix, labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10

    # We'll color each cluster differently
    colors = cm.nipy_spectral(np.linspace(0, 1, n_clusters))

    # Sort clusters by label (just for consistent plotting order)
    for i, clust_label in enumerate(sorted(unique_labels), start=1):
        ith_cluster_values = sample_sil_values[labels == clust_label]
        ith_cluster_values.sort()
        size_cluster_i = len(ith_cluster_values)
        y_upper = y_lower + size_cluster_i

        color = colors[i - 1]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        # Label on the left side
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(clust_label))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette plot for {scenario}, k={k}")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f"Avg = {silhouette_avg:.3f}")
    ax.legend(loc="best")

    ax.set_yticks([])  # Clear the y-axis
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, y_lower + 10])

    # Save figure
    outfile = os.path.join(save_dir, f"{dataset_name}_{scenario}_k{k}_silhouette.png")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"Silhouette plot saved at {outfile}")


# -------------------- Hierarchical Clustering Helpers -------------------- #
def perform_hierarchical_clustering(feature_matrix, method='complete'):
    return linkage(feature_matrix, method=method)


def plot_dendrogram_custom(linked, labels, save_path):
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, orientation='top',
               distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Byte Groups')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Dendrogram saved at {save_path}")


def group_and_reorder(byte_groups, cluster_labels):
    """
    Sort byte groups by (cluster_label, group_index).
    Returns ordered indices.
    """
    group_info = []
    for idx, label in enumerate(cluster_labels):
        group_info.append({
            'group_index': idx + 1,
            'cluster_label': label
        })
    sorted_groups = sorted(group_info, key=lambda x: (x['cluster_label'], x['group_index']))
    ordered_indices = [grp['group_index'] for grp in sorted_groups]
    return ordered_indices


# -------------------- Compression Evaluation -------------------- #
def compress_and_evaluate(byte_groups, ordered_indices, component_sizes=[1, 1, 1, 1]):
    ordered_groups = [byte_groups[i - 1].tobytes() for i in ordered_indices]
    merged_bytes = b''.join(ordered_groups)
    compressed_data = zlib.compress(merged_bytes)
    original_size = len(merged_bytes)
    compressed_size = len(compressed_data)
    ratio = compressed_size / original_size if original_size else 1
    return ratio, original_size, compressed_size


# -------------------- Entropy & Feature Extraction -------------------- #
def compute_entropy(data_window):
    from collections import Counter
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_entropy_over_data(data, window_size=256):
    entropies = []
    for start_idx in range(0, len(data) - window_size + 1, window_size):
        window = data[start_idx:start_idx + window_size]
        ent = compute_entropy(window)
        entropies.append(ent)
    return entropies


def extract_features(byte_group, window_size=256):
    """
    4 entropy metrics + 256 byte frequency bins => feature vector of length 260.
    """
    entropies = calculate_entropy_over_data(byte_group, window_size)
    if len(entropies) > 0:
        avg_ent = np.mean(entropies)
        std_ent = np.std(entropies)
        max_ent = np.max(entropies)
        min_ent = np.min(entropies)
    else:
        # If the group is smaller than window_size, fallback:
        avg_ent = std_ent = max_ent = min_ent = 0

    from collections import Counter
    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])
    return np.concatenate(([avg_ent, std_ent, max_ent, min_ent], byte_freq))


# -------------------- Data Splitting -------------------- #
def split_bytes_into_components(byte_array, component_sizes):
    """
    Splits the byte array into multiple interleaved components.
    """
    arr = np.frombuffer(byte_array, dtype=np.uint8)
    components = []
    num_components = len(component_sizes)
    for i in range(num_components):
        c = arr[i::num_components]
        components.append(c)
    return components


# -------------------- Multi-Metric Evaluate & Choose K -------------------- #
def evaluate_cluster_metrics(feature_matrix, linked,
                             min_clusters=1, max_clusters=4):
    """
    Dummy function: we only compute Silhouette here,
    but you could also compute DB, CH, Gap, etc.
    Returns a dict with best k for each metric + tie info.
    """
    silhouette_scores = {}
    cluster_assignments = {}

    n_samples = feature_matrix.shape[0]

    for k in range(min_clusters, max_clusters + 1):
        labels = fcluster(linked, k, criterion='maxclust')
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Only valid if 2 <= n_clusters <= n_samples-1
        if n_clusters >= 2 and n_clusters <= n_samples - 1:
            try:
                sil = silhouette_score(feature_matrix, labels)
            except Exception:
                sil = -1
        else:
            sil = -1

        silhouette_scores[k] = sil
        cluster_assignments[k] = labels

    if len(silhouette_scores) == 0:
        best_k_sil = None
    else:
        best_k_sil = max(silhouette_scores, key=silhouette_scores.get)

    return {
        'best_k_silhouette': best_k_sil,
        'tie_silhouette': [k for k, v in silhouette_scores.items()
                           if v == silhouette_scores[best_k_sil]] if best_k_sil else [],
        'silhouette_scores': silhouette_scores,
        'cluster_assignments': cluster_assignments
    }


def determine_optimal_clusters(feature_matrix, linked,
                               min_clusters=2, max_clusters=4):
    """
    Uses evaluate_cluster_metrics and picks the best K by silhouette.
    If none are valid, fallback to 2.
    """
    results = evaluate_cluster_metrics(feature_matrix, linked, min_clusters, max_clusters)
    best_k_sil = results['best_k_silhouette']
    final_k = best_k_sil if best_k_sil else 2
    return final_k, results


# -------------------- Plotting Helpers -------------------- #
def plot_entropy_profiles(components_entropy, dataset_name, save_path):
    df = pd.DataFrame(components_entropy).T
    df.columns = [f'Group {i + 1}' for i in range(len(components_entropy))]
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(f'Entropy Profiles for {dataset_name}')
    plt.xlabel('Window Index')
    plt.ylabel('Entropy (bits)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Entropy profiles plot saved at {save_path}")


def plot_correlation_matrix(components_entropy, dataset_name, save_path):
    df = pd.DataFrame(components_entropy).T
    df.columns = [f'Group {i + 1}' for i in range(len(components_entropy))]
    corr_mat = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix of Group Entropies for {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation matrix plot saved at {save_path}")


def plot_best_grouping(best_grouping, dataset_name, save_path):
    """
    Bar plot showing how many original groups are in each "cluster group".
    """
    groups = [f"Group {grp}" for grp in best_grouping]
    sizes = [len(grp) for grp in best_grouping]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=groups, y=sizes, palette='pastel')
    plt.title(f'Best Grouping Configuration for {dataset_name}')
    plt.xlabel('Byte Groups')
    plt.ylabel('Number of Original Groups')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Best grouping plot saved at {save_path}")


def plot_compression_comparison(best_ratio, original_ratio, dataset_name, save_path):
    labels = ['Best Grouping', 'Original Ordering']
    ratios = [best_ratio, original_ratio]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=ratios, palette='muted')
    plt.ylabel('Compression Ratio')
    plt.title(f'Compression Ratio Comparison for {dataset_name}')
    plt.ylim(0, max(ratios) * 1.2)
    for i, v in enumerate(ratios):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Compression comparison plot saved at {save_path}")


# -------------------- Feature Ablation Study -------------------- #
def run_feature_ablation(feature_matrix, dataset_name, save_dir,
                         silhouette_records):
    """
    Remove subsets of features, cluster them, and record the silhouette
    for each scenario. Then save dendrograms & silhouette plots as well.

    We accept silhouette_records so we can append (Scenario, k, Silhouette) data.
    """
    scenarios = {
        "No avg_entropy": np.delete(np.arange(feature_matrix.shape[1]), 0),
        "No std_entropy": np.delete(np.arange(feature_matrix.shape[1]), 1),
        "No max_entropy": np.delete(np.arange(feature_matrix.shape[1]), 2),
        "No min_entropy": np.delete(np.arange(feature_matrix.shape[1]), 3),
        "No entropy features": np.arange(4, feature_matrix.shape[1]),  # only byte freq
        "No byte frequencies": np.arange(0, 4)  # only the 4 entropy metrics
    }

    ablation_results = {}

    for scenario, indices in scenarios.items():
        ablated_features = feature_matrix[:, indices]
        linked = linkage(ablated_features, method='complete')

        # We'll test k=2,3 just as an example
        silhouette_scores = {}
        cluster_assignments = {}

        for k in range(2, 4):
            labels = fcluster(linked, k, criterion='maxclust')
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_samples = ablated_features.shape[0]

            # Only compute silhouette if valid
            if 2 <= n_clusters <= n_samples - 1:
                try:
                    s_val = silhouette_score(ablated_features, labels)
                except:
                    s_val = -1
            else:
                s_val = -1

            silhouette_scores[k] = s_val
            cluster_assignments[k] = labels

            # Record
            silhouette_records.append({
                "Dataset": dataset_name,
                "Scenario": f"Ablation: {scenario}",
                "k": k,
                "Silhouette": s_val
            })

            # Also plot silhouette
            plot_silhouette(ablated_features, labels,
                            dataset_name=dataset_name,
                            scenario=f"Ablation_{scenario.replace(' ', '_')}",
                            k=k, save_dir=save_dir)

        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        ablation_results[scenario] = {
            "optimal_k": optimal_k,
            "silhouette_score": silhouette_scores[optimal_k],
            "cluster_labels": cluster_assignments[optimal_k],
            "linkage_matrix": linked
        }

        # Plot dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linked, labels=[f"Group {i + 1}" for i in range(ablated_features.shape[0])])
        plt.title(f'Dendrogram - {scenario}')
        plt.xlabel('Byte Groups')
        plt.ylabel('Distance')
        plt.tight_layout()
        dendro_file = os.path.join(save_dir, f"{dataset_name}_Ablation_{scenario.replace(' ', '_')}_dendrogram.png")
        plt.savefig(dendro_file)
        plt.close()
        print(f"Dendrogram for scenario '{scenario}' saved at {dendro_file}")

    print("\nFeature Ablation Study Results:")
    for scenario, res in ablation_results.items():
        print(f"Scenario: {scenario}, Optimal k={res['optimal_k']}, "
              f"Silhouette={res['silhouette_score']:.4f}")

    return ablation_results


# -------------------- Main Analysis -------------------- #
def run_analysis(folder_path):
    """
    1) Reads .tsv in the folder
    2) Preps data, calculates entropy
    3) Hierarchical clustering w/ all features
    4) Only byte frequency
    5) Feature ablation
    6) Saves silhouette results in a CSV
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    tsv_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'. Starting analysis...\n")

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        dataset_name = os.path.splitext(tsv_file)[0]
        save_dir = os.path.join(os.getcwd(), f"{dataset_name}_analysis")
        os.makedirs(save_dir, exist_ok=True)

        print(f"==============================\nProcessing: {tsv_file}\n==============================")
        print(f"Loading dataset from {dataset_path}...")
        try:
            ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
            print("Dataset loaded successfully.\n")
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}. Skipping this file.\n")
            continue

        # Prepare data
        print("Preparing data...")
        try:
            # Assume col 0 is something else, actual bytes in columns 1..end
            byte_cols = ts_data.columns[1:]
            byte_values = ts_data[byte_cols].to_numpy().astype(np.float64)
            byte_values = byte_values.flatten(order='F').tobytes()
            byte_groups = split_bytes_into_components(byte_values, [1, 1, 1, 1])
            print("Data preparation completed.\n")
        except Exception as e:
            print(f"Error in data prep: {e}")
            continue

        # Calculate entropy
        print("Calculating entropy profiles...")
        components_entropy = []
        for i, group_bytes in enumerate(byte_groups):
            ent_list = calculate_entropy_over_data(group_bytes, window_size=65536)
            components_entropy.append(ent_list)
            print(f"Group {i + 1} => {len(ent_list)} entropy windows")

        # Plot entropy
        ent_plot_file = os.path.join(save_dir, f"{dataset_name}_entropy_profiles.png")
        plot_entropy_profiles(components_entropy, dataset_name, ent_plot_file)

        # Correlation
        corr_file = os.path.join(save_dir, f"{dataset_name}_correlation_matrix.png")
        plot_correlation_matrix(components_entropy, dataset_name, corr_file)

        # Feature extraction
        print("Extracting features for clustering...")
        features = []
        for gb in byte_groups:
            fv = extract_features(gb, window_size=65536)
            features.append(fv)
        feature_matrix = np.array(features)
        print(f"Extracted {feature_matrix.shape[0]} rows, {feature_matrix.shape[1]} features each.\n")

        # We'll store silhouette results in a list of dicts & then CSV
        silhouette_records = []

        # ---------- 1) Clustering with All Features ----------
        print("Performing hierarchical clustering (All Features)...")
        linked = perform_hierarchical_clustering(feature_matrix, method='complete')

        # Plot dendrogram
        dend_file = os.path.join(save_dir, f"{dataset_name}_All_Features_dendrogram.png")
        group_labels = [f"Group {i + 1}" for i in range(len(byte_groups))]
        plot_dendrogram_custom(linked, group_labels, dend_file)

        # For k in [2..4], do silhouette
        scenario_name = "All_Features"
        for k in range(2, 5):
            labels_k = fcluster(linked, k, criterion='maxclust')
            # how many distinct clusters?
            n_clusters = len(np.unique(labels_k))
            n_samples = feature_matrix.shape[0]
            if 2 <= n_clusters <= n_samples - 1:
                try:
                    sil_val = silhouette_score(feature_matrix, labels_k)
                except:
                    sil_val = -1
            else:
                sil_val = -1

            silhouette_records.append({
                "Dataset": dataset_name,
                "Scenario": scenario_name,
                "k": k,
                "Silhouette": sil_val
            })

            # Plot silhouette
            plot_silhouette(feature_matrix, labels_k,
                            dataset_name=dataset_name,
                            scenario=scenario_name,
                            k=k,
                            save_dir=save_dir)

        # Determine best k
        print("Determining best k (All Features)...")
        best_k_all, metrics_all = determine_optimal_clusters(feature_matrix, linked, 2, 4)
        cluster_labels_all = fcluster(linked, best_k_all, criterion='maxclust')
        print(f"Best k (All Features) = {best_k_all}, silhouette scores:")
        print(metrics_all['silhouette_scores'])

        # Group & reorder
        reorder_indices_all = group_and_reorder(byte_groups, cluster_labels_all)
        best_groups = []
        for grp_label in sorted(set(cluster_labels_all)):
            idxs = [i + 1 for i, lab in enumerate(cluster_labels_all) if lab == grp_label]
            best_groups.append(tuple(idxs))

        best_group_file = os.path.join(save_dir, f"{dataset_name}_All_Features_best_grouping.png")
        plot_best_grouping(best_groups, dataset_name, best_group_file)

        # Compression
        ratio_all, orig_size_all, comp_size_all = compress_and_evaluate(byte_groups, reorder_indices_all)
        print(f"All-Features compression ratio: {ratio_all:.4f} ({comp_size_all}/{orig_size_all})")

        # Compare original
        orig_order = list(range(1, len(byte_groups) + 1))
        ratio_orig, orig_sz, comp_sz = compress_and_evaluate(byte_groups, orig_order)
        print(f"Original ordering ratio: {ratio_orig:.4f} ({comp_sz}/{orig_sz})")

        comp_plot_file = os.path.join(save_dir, f"{dataset_name}_compression_comparison.png")
        plot_compression_comparison(ratio_all, ratio_orig, dataset_name, comp_plot_file)

        # ---------- 2) Only Byte Frequency ----------
        print("Performing clustering (Only Byte Frequency)...")
        byte_freq_features = feature_matrix[:, 4:]
        linked_freq = perform_hierarchical_clustering(byte_freq_features, 'complete')

        # Dendrogram
        dend_file_freq = os.path.join(save_dir, f"{dataset_name}_Only_Byte_Freq_dendrogram.png")
        plot_dendrogram_custom(linked_freq, group_labels, dend_file_freq)

        scenario_name = "Only_Byte_Freq"
        for k in range(2, 5):
            labels_k = fcluster(linked_freq, k, criterion='maxclust')
            n_clusters = len(np.unique(labels_k))
            n_samples = byte_freq_features.shape[0]
            if 2 <= n_clusters <= n_samples - 1:
                try:
                    sil_val = silhouette_score(byte_freq_features, labels_k)
                except:
                    sil_val = -1
            else:
                sil_val = -1

            silhouette_records.append({
                "Dataset": dataset_name,
                "Scenario": scenario_name,
                "k": k,
                "Silhouette": sil_val
            })

            plot_silhouette(byte_freq_features, labels_k,
                            dataset_name=dataset_name,
                            scenario=scenario_name,
                            k=k,
                            save_dir=save_dir)

        # Evaluate best k
        best_k_freq, metrics_freq = determine_optimal_clusters(byte_freq_features, linked_freq, 2, 4)
        print(f"Best k (Only Byte Freq) = {best_k_freq}, silhouette scores:")
        print(metrics_freq['silhouette_scores'])

        # ---------- 3) Feature Ablation ----------
        print("Running Feature Ablation Study...")
        ablation_results = run_feature_ablation(feature_matrix, dataset_name,
                                                save_dir,
                                                silhouette_records)
        print("Feature ablation done.\n")

        # ---------- Finally: Save Silhouette CSV for This Dataset ----------
        df_sil = pd.DataFrame(silhouette_records)
        csv_path = os.path.join(save_dir, f"{dataset_name}_silhouette_results.csv")
        df_sil.to_csv(csv_path, index=False)
        print(f"Saved all silhouette scores to {csv_path}\n")

        print(f"Analysis for '{tsv_file}' completed.\n{'=' * 30}\n")

    print("All analyses completed successfully.")


# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    # Replace with your folder containing .tsv files
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)
