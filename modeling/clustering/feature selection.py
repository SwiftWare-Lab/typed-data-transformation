import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import zlib


# --------------------------- Entropy Calculation --------------------------- #

def compute_entropy(data_window):
    """Compute the entropy of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_entropy_over_data(data, window_size=256):
    """Slide a window of size `window_size` across data and compute entropy for each window."""
    entropies = []
    data_len = len(data)
    for start_idx in range(0, data_len - window_size + 1, window_size):
        window = data[start_idx:start_idx + window_size]
        ent = compute_entropy(window)
        entropies.append(ent)
    return entropies


# --------------------------- Feature Extraction --------------------------- #

def extract_features(byte_group, window_size=256):
    """
    Extract features from a byte group.

    The feature vector consists of 4 entropy metrics (average, std, max, min)
    followed by 256 normalized byte frequencies.

    :param byte_group: Byte array of the group.
    :param window_size: Size of each window for entropy calculation.
    :return: 1D numpy array of features.
    """
    entropies = calculate_entropy_over_data(byte_group, window_size)
    average_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)

    # Byte frequency
    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])

    # Combine all features into a single array
    features = np.concatenate(([average_entropy, std_entropy, max_entropy, min_entropy], byte_freq))
    return features


# --------------------------- Data Splitting --------------------------- #

def split_bytes_into_components(byte_array, component_sizes):
    """
    Split the byte array into individual components based on specified sizes.

    :param byte_array: The original byte array.
    :param component_sizes: List indicating the size of each component.
    :return: List of numpy arrays, each representing a component.
    """
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    # Assume components are interleaved
    components = []
    num_components = len(component_sizes)
    for i in range(num_components):
        component = byte_array[i::num_components]
        components.append(component)
    return components


# --------------------------- Hierarchical Clustering --------------------------- #

def perform_hierarchical_clustering(feature_matrix, method='complete'):
    """
    Perform hierarchical clustering.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param method: Linkage method.
    :return: Linkage matrix.
    """
    linked = linkage(feature_matrix, method=method)
    return linked


def plot_dendrogram_custom(linked, labels, save_path):
    """
    Plot and save a dendrogram.

    :param linked: Linkage matrix.
    :param labels: Labels for each data point.
    :param save_path: File path to save the dendrogram.
    """
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Byte Groups')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Dendrogram saved at {save_path}")


def determine_optimal_clusters(linked, feature_matrix, max_clusters=3):
    """
    Determine the optimal number of clusters based on silhouette scores.

    :param linked: Linkage matrix.
    :param feature_matrix: Original feature matrix.
    :param max_clusters: Maximum number of clusters to consider.
    :return: Optimal number of clusters.
    """
    cluster_assignments = {}
    for k in range(1, max_clusters + 1):
        labels = fcluster(linked, k, criterion='maxclust')
        cluster_assignments[k] = labels

    silhouette_scores = {}
    for k, labels in cluster_assignments.items():
        try:
            score = silhouette_score(feature_matrix, labels)
            silhouette_scores[k] = score
        except Exception as e:
            silhouette_scores[k] = -1
            print(f"Silhouette score calculation failed for K={k}: {e}")

    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    else:
        optimal_k = 2

    print(f"Optimal number of clusters based on silhouette scores: {optimal_k}")
    return optimal_k


def group_and_reorder(byte_groups, cluster_labels):
    """
    Group and reorder byte groups based on cluster labels.

    :param byte_groups: List of byte arrays.
    :param cluster_labels: Cluster labels for each group.
    :return: Ordered list of group indices.
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


# --------------------------- Compression Evaluation --------------------------- #

def compress_and_evaluate(byte_groups, ordered_indices, component_sizes=[1, 1, 1, 1]):
    """
    Compress byte groups based on ordered indices and evaluate compression ratio.

    :param byte_groups: List of original byte arrays.
    :param ordered_indices: List indicating the order of groups.
    :param component_sizes: List indicating the size of each group.
    :return: Compression ratio, original size, compressed size.
    """
    ordered_groups = [byte_groups[i - 1].tobytes() for i in ordered_indices]
    merged_bytes = b''.join(ordered_groups)
    compressed_data = zlib.compress(merged_bytes)
    original_size = len(merged_bytes)
    compressed_size = len(compressed_data)
    compression_ratio = compressed_size / original_size
    return compression_ratio, original_size, compressed_size


# --------------------------- Plotting Functions --------------------------- #

def plot_entropy_profiles(components_entropy, dataset_name, save_path):
    df = pd.DataFrame(components_entropy).T
    df.columns = [f'Group {i + 1}' for i in range(len(components_entropy))]
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(f'Entropy Profiles for {dataset_name}')
    plt.xlabel('Window Index')
    plt.ylabel('Entropy (bits)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Entropy profiles plot saved as {save_path}")


def plot_correlation_matrix(components_entropy, dataset_name, save_path):
    df = pd.DataFrame(components_entropy).T
    df.columns = [f'Group {i + 1}' for i in range(len(components_entropy))]
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix of Group Entropies for {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation matrix plot saved as {save_path}")


def plot_best_grouping(best_grouping, dataset_name, save_path):
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
    print(f"Best grouping plot saved as {save_path}")


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
    print(f"Compression comparison plot saved as {save_path}")


# --------------------------- Feature Ablation Study --------------------------- #

def run_feature_ablation(feature_matrix, dataset_name, save_dir):
    """
    Perform a feature ablation study by removing specific features and evaluating clustering performance.

    For each scenario (e.g., removing a particular entropy metric or the entire byte frequency set),
    hierarchical clustering is performed and silhouette scores are computed.
    A dendrogram is saved for each ablation scenario.

    :param feature_matrix: The full feature matrix (each row corresponds to a byte group).
    :param dataset_name: Name of the dataset (used in plot titles and file names).
    :param save_dir: Directory where plots will be saved.
    :return: Dictionary with ablation results.
    """
    scenarios = {
        "All Features": np.arange(feature_matrix.shape[1]),
        "No avg_entropy": np.delete(np.arange(feature_matrix.shape[1]), 0),
        "No std_entropy": np.delete(np.arange(feature_matrix.shape[1]), 1),
        "No max_entropy": np.delete(np.arange(feature_matrix.shape[1]), 2),
        "No min_entropy": np.delete(np.arange(feature_matrix.shape[1]), 3),
        "No entropy features": np.arange(4, feature_matrix.shape[1]),  # only byte frequencies
        "No byte frequencies": np.arange(0, 4)  # only entropy metrics
    }

    ablation_results = {}

    for scenario, indices in scenarios.items():
        ablated_features = feature_matrix[:, indices]
        scaler = StandardScaler()
        ablated_features_scaled = scaler.fit_transform(ablated_features)

        # Perform hierarchical clustering using complete linkage
        linked = linkage(ablated_features_scaled, method='complete')

        # Evaluate silhouette scores for k = 2 and 3 clusters
        silhouette_scores = {}
        cluster_assignments = {}
        for k in range(2, 4):
            labels = fcluster(linked, k, criterion='maxclust')
            try:
                score = silhouette_score(ablated_features_scaled, labels)
            except Exception as e:
                score = -1
            silhouette_scores[k] = score
            cluster_assignments[k] = labels

        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        optimal_labels = cluster_assignments[optimal_k]

        ablation_results[scenario] = {
            "optimal_k": optimal_k,
            "silhouette_score": silhouette_scores[optimal_k],
            "cluster_labels": optimal_labels,
            "linkage_matrix": linked
        }

        # Plot dendrogram for the current ablation scenario
        plt.figure(figsize=(10, 7))
        dendrogram(linked, labels=[f"Group {i + 1}" for i in range(feature_matrix.shape[0])])
        plt.title(f'Dendrogram - {scenario}')
        plt.xlabel('Byte Groups')
        plt.ylabel('Distance')
        plt.tight_layout()
        dendro_save_path = os.path.join(save_dir, f"{dataset_name}_{scenario.replace(' ', '_')}_dendrogram.png")
        plt.savefig(dendro_save_path)
        plt.close()
        print(f"Dendrogram for scenario '{scenario}' saved at {dendro_save_path}")

    print("\nFeature Ablation Study Results:")
    for scenario, res in ablation_results.items():
        print(f"{scenario}: Optimal Clusters = {res['optimal_k']}, Silhouette Score = {res['silhouette_score']:.4f}")

    return ablation_results


# --------------------------- Main Analysis Function --------------------------- #

def run_analysis(folder_path):
    """
    Perform analysis on all .tsv files within the specified folder.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]

    if not tsv_files:
        print(f"No .tsv files found in the folder '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'. Starting analysis...\n")

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        dataset_name = os.path.splitext(tsv_file)[0]
        save_dir = os.path.join(os.getcwd(), f"{dataset_name}_analysis")
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------- Data Loading --------------------------- #
        print(f"==============================\nProcessing: {tsv_file}\n==============================")
        print(f"Loading dataset from {dataset_path}...")
        try:
            ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
            print("Dataset loaded successfully.\n")
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}. Skipping this file.\n")
            continue

        # --------------------------- Data Preparation --------------------------- #
        print("Preparing data...")
        try:
            byte_columns = ts_data.columns[1:]
            byte_data = ts_data[byte_columns].to_numpy().astype(np.float64)
            byte_data = byte_data.flatten(order='F').tobytes()
            byte_groups = split_bytes_into_components(byte_data, [1, 1, 1, 1])
            print("Data preparation completed.\n")
        except Exception as e:
            print(f"An error occurred during data preparation: {e}. Skipping this file.\n")
            continue

        # --------------------------- Entropy Calculation --------------------------- #
        print("Calculating entropy profiles...")
        components_entropy = []
        for i, group_bytes in enumerate(byte_groups):
            entropies = calculate_entropy_over_data(group_bytes, window_size=65536)
            components_entropy.append(entropies)
            print(f"Group {i + 1} entropy calculation completed. {len(entropies)} windows processed.\n")

        plot_entropy_profiles_path = os.path.join(save_dir, f"{dataset_name}_entropy_profiles.png")
        plot_entropy_profiles(components_entropy, dataset_name, plot_entropy_profiles_path)

        print("Analyzing correlations between group entropies...")
        plot_correlation_matrix_path = os.path.join(save_dir, f"{dataset_name}_correlation_matrix.png")
        plot_correlation_matrix(components_entropy, dataset_name, plot_correlation_matrix_path)

        # --------------------------- Feature Extraction --------------------------- #
        print("Extracting features for clustering...")
        features = []
        for group_bytes in byte_groups:
            feature_vector = extract_features(group_bytes, window_size=65536)
            features.append(feature_vector)
        feature_matrix = np.array(features)
        print(f"Extracted {feature_matrix.shape[0]} feature vectors with {feature_matrix.shape[1]} features each.\n")

        # --------------------------- Clustering Using All Features --------------------------- #
        print("Performing hierarchical clustering using all features...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        linked = perform_hierarchical_clustering(features_scaled, method='complete')
        dendrogram_save_path = os.path.join(save_dir, f"{dataset_name}_dendrogram.png")
        labels = [f'Group {i + 1}' for i in range(len(byte_groups))]
        plot_dendrogram_custom(linked, labels, dendrogram_save_path)

        print("Determining the optimal number of clusters...")
        optimal_k = determine_optimal_clusters(linked, features_scaled, max_clusters=3)
        cluster_labels = fcluster(linked, optimal_k, criterion='maxclust')
        print(f"Cluster labels for each group: {cluster_labels}\n")

        print("Grouping and reordering byte groups based on clustering...")
        ordered_indices = group_and_reorder(byte_groups, cluster_labels)
        print(f"Optimal group order (Group indices): {ordered_indices}\n")

        print("Plotting best grouping configuration...")
        best_grouping = []
        for grp in sorted(set(cluster_labels)):
            indices = [i + 1 for i, label in enumerate(cluster_labels) if label == grp]
            best_grouping.append(tuple(indices))
        plot_best_grouping(best_grouping, dataset_name, os.path.join(save_dir, f"{dataset_name}_best_grouping.png"))

        print("Evaluating compression ratio with the determined group order...")
        compression_ratio, original_size, compressed_size = compress_and_evaluate(byte_groups, ordered_indices, [1, 1, 1, 1])
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.4f} ({(compressed_size / original_size) * 100:.2f}%)\n")

        print("Comparing with original group ordering...")
        original_ordering = list(range(1, len(byte_groups) + 1))
        original_compression_ratio, orig_size, orig_compressed_size = compress_and_evaluate(byte_groups, original_ordering, [1, 1, 1, 1])
        print(f"Original Ordering Compression ratio: {original_compression_ratio:.4f} ({orig_compressed_size}/{orig_size})\n")
        plot_compression_comparison_path = os.path.join(save_dir, f"{dataset_name}_compression_comparison.png")
        plot_compression_comparison(compression_ratio, original_compression_ratio, dataset_name, plot_compression_comparison_path)

        # --------------------------- Clustering Using Only Byte Frequency Features --------------------------- #
        print("Performing hierarchical clustering using only byte frequency features (removing entropy metrics)...")
        # Select only byte frequency features (remove first 4 entropy features)
        byte_freq_features = feature_matrix[:, 4:]
        scaler_byte_freq = StandardScaler()
        byte_freq_scaled = scaler_byte_freq.fit_transform(byte_freq_features)
        linked_byte_freq = perform_hierarchical_clustering(byte_freq_scaled, method='complete')
        dendrogram_byte_freq_path = os.path.join(save_dir, f"{dataset_name}_byte_frequency_dendrogram.png")
        labels_byte_freq = [f'Group {i + 1}' for i in range(len(byte_groups))]
        plot_dendrogram_custom(linked_byte_freq, labels_byte_freq, dendrogram_byte_freq_path)
        print("Clustering with only byte frequency features completed.\n")

        # --------------------------- Feature Ablation Study --------------------------- #
        print("Running feature ablation study to evaluate the impact of each feature on clustering...")
        ablation_results = run_feature_ablation(feature_matrix, dataset_name, save_dir)
        print("Feature ablation study completed.\n")

        print(f"Analysis for '{tsv_file}' completed successfully.\n{'=' * 30}\n")

    print("All analyses completed successfully.")


if __name__ == "__main__":
    # Update this path to point to the folder containing your .tsv files
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    run_analysis(folder_path)
