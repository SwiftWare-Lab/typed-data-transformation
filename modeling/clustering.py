import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
def compute_kth_entropy(arr, k=2):
    if len(arr) < k: return 0.0
    grams = [tuple(arr[i:i+k]) for i in range(len(arr)-k+1)]
    freq = Counter(grams); total = len(grams)
    return -sum((cnt/total) * math.log2(cnt/total) for cnt in freq.values())

def delta_k_entropy(global_stream, cluster_streams, k=2):
    H_gl = compute_kth_entropy(global_stream, k)
    total = len(global_stream)
    Hw = sum(len(c) * compute_kth_entropy(c, k) for c in cluster_streams) / total
    return H_gl - Hw
def delta_k_entropy(global_stream, cluster_streams, k=2):
    H_gl = compute_kth_entropy(global_stream, k)
    total = len(global_stream)
    Hw = sum(len(c) * compute_kth_entropy(c, k) for c in cluster_streams) / total
    return H_gl - Hw

def delta_H0(global_stream, cluster_streams):
    H0 = compute_entropy(global_stream)
    total = len(global_stream)
    Hw0 = sum(len(c) * compute_entropy(c) for c in cluster_streams) / total
    return H0 - Hw0
#----------------------
def compute_byte_frequency(data_window):
    """Compute the normalized byte frequency of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    byte_freq = np.array([freq.get(i, 0) / total for i in range(256)])
    return byte_freq


def extract_features1(byte_group, window_size=256):
    """
    Extract features from a byte group.

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

def extract_features(byte_group, global_stream, window_size=256):
    # 1) Entropy windows
    entropies = calculate_entropy_over_data(byte_group, window_size)
    avg_ent, std_ent, max_ent, min_ent = (
        np.mean(entropies),
        np.std(entropies),
        np.max(entropies),
        np.min(entropies),
    )

    # 2) Byte-frequency stddev
    freq = Counter(byte_group)
    probs = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])
    freq_std = float(np.std(probs))

    # 3) Delta metrics versus the full stream
    #    first flatten this group to bytes
    grp_bytes = np.frombuffer(byte_group.tobytes(), dtype=np.uint8)
    delta_match = delta_k_entropy(global_stream, [grp_bytes], k=2)
    delta_h0    = delta_H0(global_stream, [grp_bytes])

    # 4) Concatenate into one feature vector
    return np.array([
        avg_ent, std_ent, max_ent, min_ent,
        freq_std,
        delta_match, delta_h0
    ])

# --------------------------- Data Splitting --------------------------- #

def split_bytes_into_components(byte_array, component_sizes):
    """
    Split the byte array into individual components based on specified sizes.

    :param byte_array: The original byte array.
    :param component_sizes: List indicating the size of each component.
    :return: List of numpy arrays, each representing a component.
    """
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    num_elements = len(byte_array) // total_bytes

    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        component = byte_array[offset::total_bytes]
        components.append(component)

    return components


# --------------------------- Hierarchical Clustering --------------------------- #

def perform_hierarchical_clustering(feature_matrix, method='complete'):
    """
    Perform hierarchical clustering.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param method: Linkage method.
    :param metric: Distance metric.
    :return: Linkage matrix.
    """
    linked = linkage(feature_matrix, method=method)
    return linked
from scipy.spatial.distance import squareform

def perform_hierarchical_clustering1(feature_matrix, method='complete'):
    """
    Perform hierarchical clustering using correlation-based distance.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param method: Linkage method (e.g., 'single', 'complete', 'average', 'ward').
    :return: Linkage matrix.
    """
    # 1. Compute correlation among rows (data points)
    #    np.corrcoef expects data in rows => shape: (N, D), returns (N, N) correlation matrix
    corr_matrix = np.corrcoef(feature_matrix)

    # 2. Convert correlation to distance: distance = 1 - correlation
    dist_matrix = 1 - corr_matrix

    # 3. Convert the square distance matrix to condensed form
    #    (this is what 'linkage' expects if you supply a "distance matrix" instead of raw observations)
    dist_condensed = squareform(dist_matrix, checks=False)

    # 4. Perform hierarchical clustering using the condensed distance
    linked = linkage(dist_condensed, method=method)
    return linked
####################################

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
    :param feature_matrix: Original feature matrix before clustering.
    :param max_clusters: Maximum number of clusters to consider.
    :return: Optimal number of clusters.
    """
    # Extract cluster assignments for K=2 and K=3
    cluster_assignments = {}
    for k in range(1, max_clusters + 1):
        labels = fcluster(linked, k, criterion='maxclust')
        cluster_assignments[k] = labels

    # Compute silhouette scores
    silhouette_scores = {}
    for k, labels in cluster_assignments.items():
        try:
            score = silhouette_score(feature_matrix, labels)
            silhouette_scores[k] = score
        except Exception as e:
            silhouette_scores[k] = -1  # Invalid score
            print(f"Silhouette score calculation failed for K={k}: {e}")

    # Select K with the highest silhouette score
    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    else:
        optimal_k = 2  # Default to 2 if no valid scores

    print(f"Optimal number of clusters based on silhouette scores: {optimal_k}")
    return optimal_k


def group_and_reorder(byte_groups, cluster_labels):
    """
    Group and reorder byte groups based on cluster labels.

    :param byte_groups: List of byte arrays.
    :param cluster_labels: Cluster labels for each group.
    :return: Ordered list of group indices.
    """
    # Combine group indices with their cluster labels and average entropy
    group_info = []
    for idx, label in enumerate(cluster_labels):
        group_info.append({
            'group_index': idx + 1,
            'cluster_label': label
            # Add more metrics if needed
        })

    # Sort groups by cluster label and then by group index
    sorted_groups = sorted(group_info, key=lambda x: (x['cluster_label'], x['group_index']))

    # Extract ordered group indices
    ordered_indices = [grp['group_index'] for grp in sorted_groups]

    return ordered_indices


# --------------------------- Compression Evaluation --------------------------- #

def compress_and_evaluate(byte_groups, ordered_indices, component_sizes=[1, 1, 1, 1]):
    """
    Compress byte groups based on ordered indices and evaluate compression ratio.

    :param byte_groups: List of original byte arrays (numpy.ndarray).
    :param ordered_indices: List indicating the order of groups.
    :param component_sizes: List indicating the size of each group.
    :return: Compression ratio, original size, compressed size.
    """
    # Reorder groups and convert to bytes
    ordered_groups = [byte_groups[i - 1].tobytes() for i in ordered_indices]  # Convert to bytes

    # Merge ordered groups
    merged_bytes = b''.join(ordered_groups)

    # Compress using zlib
    compressed_data = zlib.compress(merged_bytes)

    # Calculate sizes
    original_size = len(merged_bytes)
    compressed_size = len(compressed_data)
    compression_ratio = compressed_size / original_size

    return compression_ratio, original_size, compressed_size


# --------------------------- Plotting Functions --------------------------- #

def plot_dendrogram_custom_func(linked, labels, save_path):
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


def plot_best_grouping(best_grouping, dataset_name, save_path):
    """
    Plot the best grouping configuration.

    :param best_grouping: Best grouping tuple.
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
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
    print(f"Best grouping plot saved as {save_path}")


def plot_compression_comparison(best_ratio, original_ratio, dataset_name, save_path):
    """
    Plot a comparison of compression ratios between best grouping and original ordering.

    :param best_ratio: Compression ratio for best grouping.
    :param original_ratio: Compression ratio for original ordering.
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
    """
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


# # --------------------------- Execute Script --------------------------- #


def run_analysis(folder_path):
    """
    Perform analysis on all .tsv files within the specified folder.

    :param folder_path: Path to the folder containing .tsv files.
    """
    # --------------------------- Configuration --------------------------- #
    # Verify that the provided folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    # Retrieve all .tsv files in the folder
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]

    if not tsv_files:
        print(f"No .tsv files found in the folder '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'. Starting analysis...\n")

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        dataset_name = os.path.splitext(tsv_file)[0]

        folder_path1 = ('/mnt/c/Users/jamalids/Downloads')
        save_dir = os.path.join(folder_path1, f"{dataset_name}_normal")
        os.makedirs(save_dir, exist_ok=True)

        # Parameters
        window_size = 65536
        #window_size =921600
       # window_size = 1024
        component_sizes = [1, 1, 1, 1]  # Assuming four groups, each of size 1 byte

        print(f"==============================\nProcessing: {tsv_file}\n==============================")

        # --------------------------- Data Loading --------------------------- #
        print(f"Loading dataset from {dataset_path}...")
        try:
            ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
            print("Dataset loaded successfully.\n")
        except FileNotFoundError:
            print(f"Error: File not found at {dataset_path}. Skipping this file.\n")
            continue
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}. Skipping this file.\n")
            continue

        # --------------------------- Data Preparation --------------------------- #
        print("Preparing data...")
        try:
            # Assuming the first column is an index or identifier
            byte_columns = ts_data.columns[1:]
            # byte_data = ts_data[byte_columns].to_numpy(order='F').astype(np.float32).tobytes()

            byte_data = ts_data[byte_columns].to_numpy().astype(np.float64)
            byte_data = byte_data.flatten(order='F').tobytes()

            # Split into four groups
            byte_groups = split_bytes_into_components(byte_data, component_sizes)
            print("Data preparation completed.\n")
        except Exception as e:
            print(f"An error occurred during data preparation: {e}. Skipping this file.\n")
            continue




        # --------------------------- Feature Extraction for Clustering --------------------------- #
        # --------------------------- Feature Extraction for Clustering --------------------------- #
        print("Extracting features for clustering...")

        # Build the global byte stream (concatenate all groups)
        full_stream_bytes = b''.join(g.tobytes() for g in byte_groups)
        global_stream = np.frombuffer(full_stream_bytes, dtype=np.uint8)

        features = []
        for group_bytes in byte_groups:
            # Pass global_stream into extract_features
            feature_vector = extract_features(group_bytes,
                                              global_stream,
                                              window_size=window_size)
            features.append(feature_vector)

        feature_matrix = np.vstack(features)
        print(f"Extracted {feature_matrix.shape[0]} feature vectors "
              f"with {feature_matrix.shape[1]} features each.\n")


        # --------------------------- Hierarchical Clustering --------------------------- #
        print("Performing hierarchical clustering...")
        linked = perform_hierarchical_clustering(feature_matrix, method='complete')
        dendrogram_save_path = os.path.join(save_dir, f"{dataset_name}_average.png")
        labels = [f'Group {i + 1}' for i in range(len(byte_groups))]
        plot_dendrogram_custom(linked, labels, dendrogram_save_path)

        # Determine optimal number of clusters
        print("Determining the optimal number of clusters...")
        optimal_k = determine_optimal_clusters(linked, feature_matrix, max_clusters=3)
        print()

        # Assign cluster labels based on optimal_k
        cluster_labels = fcluster(linked, optimal_k, criterion='maxclust')
        print(f"Cluster labels for each group: {cluster_labels}\n")

        # --------------------------- Grouping and Reordering --------------------------- #
        print("Grouping and reordering byte groups based on clustering...")
        ordered_indices = group_and_reorder(byte_groups, cluster_labels)
        print(f"Optimal group order (Group indices): {ordered_indices}\n")

        # Plot best grouping
        best_grouping = []
        for grp in sorted(set(cluster_labels)):
            indices = [i + 1 for i, label in enumerate(cluster_labels) if label == grp]
            best_grouping.append(tuple(indices))
        plot_best_grouping(best_grouping, dataset_name, os.path.join(save_dir, f"{dataset_name}_best_grouping.png"))

        # --------------------------- Compression Evaluation --------------------------- #
        print("Evaluating compression ratio with the determined group order...")
        compression_ratio, original_size, compressed_size = compress_and_evaluate(
            byte_groups, ordered_indices, component_sizes)
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.4f} ({(compressed_size / original_size) * 100:.2f}%)\n")

        # Compare with original ordering
        print("Comparing with original group ordering (1,2,3,4)...")
        original_ordering = list(range(1, len(byte_groups) + 1))
        original_compression_ratio, orig_size, orig_compressed_size = compress_and_evaluate(
            byte_groups, original_ordering, component_sizes)
        print(
            f"Original Ordering Compression ratio: {original_compression_ratio:.4f} ({orig_compressed_size}/{orig_size})\n")

        # --------------------------- Compression Comparison Plot --------------------------- #
        plot_compression_comparison_path = os.path.join(
            save_dir, f"{dataset_name}_compression_comparison.png")
        plot_compression_comparison(compression_ratio, original_compression_ratio, dataset_name,
                                    plot_compression_comparison_path)

        print(f"Analysis for '{tsv_file}' completed successfully.\n{'='*30}\n")

    print("All analyses completed successfully.")

if __name__ == "__main__":
    # Specify the folder containing .tsv files
    folder_path =  ('/mnt/c/Users/jamalids/Downloads/dataset/OBS/test')   # Update as needed
    run_analysis(folder_path)