import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
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

def compute_byte_frequency(data_window):
    """Compute the normalized byte frequency of a given byte window."""
    freq = Counter(data_window)
    total = len(data_window)
    byte_freq = np.array([freq.get(i, 0) / total for i in range(256)])
    return byte_freq


def compute_run_length_stats(byte_group):
    """
    Compute run-length statistics for a byte group.

    :param byte_group: Byte array (numpy.ndarray).
    :return: Average run length, maximum run length.
    """
    if len(byte_group) == 0:
        return 0, 0
    runs = 1
    current_run = 1
    max_run = 1
    for i in range(1, len(byte_group)):
        if byte_group[i] == byte_group[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            runs += 1
            current_run = 1
    average_run_length = len(byte_group) / runs
    return average_run_length, max_run


def compute_byte_pair_frequencies(byte_group):
    """
    Compute byte pair frequencies for a byte group.

    :param byte_group: Byte array (numpy.ndarray).
    :return: Array of byte pair frequencies (size 256*256).
    """
    pairs = zip(byte_group[:-1], byte_group[1:])
    pair_counts = Counter(pairs)
    total_pairs = len(byte_group) - 1 if len(byte_group) > 1 else 1
    byte_pair_freq = np.array([pair_counts.get((i, j), 0) / total_pairs for i in range(256) for j in range(256)])
    return byte_pair_freq


def compute_transition_probabilities(byte_group):
    """
    Compute transition probabilities between bytes for a byte group.

    :param byte_group: Byte array (numpy.ndarray).
    :return: Array of transition probabilities (size 256*256).
    """
    pairs = zip(byte_group[:-1], byte_group[1:])
    pair_counts = Counter(pairs)
    transition_prob = np.array([pair_counts.get((i, j), 0) / sum(pair_counts.get((i, k), 0) for k in range(256)) if sum(
        pair_counts.get((i, k), 0) for k in range(256)) > 0 else 0 for i in range(256) for j in range(256)])
    return transition_prob


def extract_features(byte_group, window_size=256):
    """
    Extract features from a byte group.

    :param byte_group: Byte array of the group.
    :param window_size: Size of each window for entropy calculation.
    :return: 1D numpy array of features.
    """
    # Entropy Features
    entropies = calculate_entropy_over_data(byte_group, window_size)
    average_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    max_entropy = np.max(entropies)
    min_entropy = np.min(entropies)

    # Byte Frequency Features
    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])

    # Run-Length Encoding Features
    avg_run_length, max_run_length = compute_run_length_stats(byte_group)

    # Byte Pair Frequencies
    byte_pair_freq = compute_byte_pair_frequencies(byte_group)

    # Transition Probabilities
    transition_prob = compute_transition_probabilities(byte_group)

    # Compression Ratio Feature
    compression_ratio = compute_individual_compression_ratio(byte_group)

    # Combine all features into a single array
    features = np.concatenate((
        [average_entropy, std_entropy, max_entropy, min_entropy],
        byte_freq,
        [avg_run_length, max_run_length],
        byte_pair_freq,
        transition_prob,
        [compression_ratio]
    ))

    return features


def compute_individual_compression_ratio(byte_group):
    """
    Compress a byte group individually and compute its compression ratio.

    :param byte_group: Byte array (numpy.ndarray).
    :return: Compression ratio (compressed_size / original_size).
    """
    if len(byte_group) == 0:
        return 0
    byte_data = byte_group.tobytes()
    compressed_data = zlib.compress(byte_data)
    original_size = len(byte_data)
    compressed_size = len(compressed_data)
    return compressed_size / original_size if original_size > 0 else 0


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

def perform_hierarchical_clustering(feature_matrix, method='ward', metric='euclidean'):
    """
    Perform hierarchical clustering.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param method: Linkage method.
    :param metric: Distance metric.
    :return: Linkage matrix.
    """
    linked = linkage(feature_matrix, method=method, metric=metric)
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


def determine_optimal_clusters(linked, feature_matrix, max_clusters=5):
    """
    Determine the optimal number of clusters based on silhouette scores.

    :param linked: Linkage matrix.
    :param feature_matrix: Original feature matrix before clustering.
    :param max_clusters: Maximum number of clusters to consider.
    :return: Optimal number of clusters.
    """
    # Extract cluster assignments for K=2 to K=max_clusters
    cluster_assignments = {}
    for k in range(2, max_clusters + 1):
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
    # Combine group indices with their cluster labels and additional metrics if needed
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


# --------------------------- Byte Frequency Measurement --------------------------- #

def measure_byte_frequencies(byte_groups):
    """
    Measure the frequency of each byte (0-255) in each group.

    :param byte_groups: List of byte arrays (numpy.ndarray).
    :return: Dictionary with group indices as keys and byte frequency arrays as values.
    """
    byte_frequencies = {}
    for idx, group_bytes in enumerate(byte_groups):
        freq = Counter(group_bytes)
        byte_freq = np.array([freq.get(i, 0) for i in range(256)])
        byte_frequencies[f'Group {idx + 1}'] = byte_freq
    return byte_frequencies


def plot_byte_frequencies(byte_frequencies, dataset_name, save_dir):
    """
    Plot byte frequency distributions for each group.

    :param byte_frequencies: Dictionary with group indices as keys and byte frequency arrays as values.
    :param dataset_name: Name of the dataset for titling.
    :param save_dir: Directory to save the plots.
    """
    for group, freq in byte_frequencies.items():
        plt.figure(figsize=(12, 6))
        sns.barplot(x=np.arange(256), y=freq, palette='viridis')
        plt.title(f'Byte Frequency Distribution for {group} in {dataset_name}')
        plt.xlabel('Byte Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{dataset_name}_{group}_byte_frequency.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Byte frequency plot for {group} saved as {save_path}")


# --------------------------- Plotting Functions --------------------------- #

def plot_entropy_profiles(components_entropy, dataset_name, save_path):
    """Plot entropy profiles for each byte group."""
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
    """
    Plot the correlation matrix of component entropies.

    :param components_entropy: List of entropy lists for each component.
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
    """
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


# --------------------------- Main Function --------------------------- #

def run_analysis():
    # --------------------------- Configuration --------------------------- #
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/High-Entropy/32/jw_mirimage_f32.tsv"  # Update path as needed
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    save_dir = f"/home/jamalids/Documents/{dataset_name}_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    window_size = 65536
    component_sizes = [1, 1, 1, 1]  # Assuming four groups, each of size 1 byte

    # --------------------------- Data Loading --------------------------- #
    print(f"Loading dataset from {dataset_path}...")
    try:
        ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
        print("Dataset loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}. Please check the path and try again.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # --------------------------- Data Preparation --------------------------- #
    print("Preparing data...")
    try:
        # Assuming the first column is an index or identifier
        byte_columns = ts_data.columns[1:]
        byte_data = ts_data[byte_columns].to_numpy().astype(np.float32).tobytes()

        # Split into four groups
        byte_groups = split_bytes_into_components(byte_data, component_sizes)
        print("Data preparation completed.\n")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return

    # --------------------------- Entropy Calculation --------------------------- #
    print("Calculating entropy profiles...")
    components_entropy = []
    for i, group_bytes in enumerate(byte_groups):
        entropies = calculate_entropy_over_data(group_bytes, window_size=window_size)
        components_entropy.append(entropies)
        print(f"Group {i + 1} entropy calculation completed. {len(entropies)} windows processed.\n")

    # Plot entropy profiles
    plot_entropy_profiles_path = os.path.join(save_dir, f"{dataset_name}_entropy_profiles.png")
    plot_entropy_profiles(components_entropy, dataset_name, plot_entropy_profiles_path)

    # --------------------------- Correlation Analysis --------------------------- #
    print("Analyzing correlations between group entropies...")
    # Create a DataFrame where each column is a group and each row is a window
    df_entropies = pd.DataFrame(components_entropy).T
    df_entropies.columns = [f'Group {i + 1}' for i in range(len(components_entropy))]
    correlation_matrix = df_entropies.corr()

    # Plot correlation matrix
    plot_correlation_matrix_path = os.path.join(save_dir, f"{dataset_name}_correlation_matrix.png")
    plot_correlation_matrix(components_entropy, dataset_name, plot_correlation_matrix_path)
    print()

    # --------------------------- Feature Extraction for Clustering --------------------------- #
    print("Extracting features for clustering...")
    features = []
    for group_bytes in byte_groups:
        feature_vector = extract_features(group_bytes, window_size=window_size)
        features.append(feature_vector)
    feature_matrix = np.array(features)
    print(f"Extracted {feature_matrix.shape[0]} feature vectors with {feature_matrix.shape[1]} features each.\n")

    # Optional: Feature Scaling (commented out as not required for hierarchical clustering)
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(feature_matrix)

    # --------------------------- Hierarchical Clustering --------------------------- #
    print("Performing hierarchical clustering...")
    linked = perform_hierarchical_clustering(feature_matrix, method='ward', metric='euclidean')
    dendrogram_save_path = os.path.join(save_dir, f"{dataset_name}_dendrogram.png")
    labels = [f'Group {i + 1}' for i in range(len(byte_groups))]
    plot_dendrogram_custom(linked, labels, dendrogram_save_path)

    # Determine optimal number of clusters
    print("Determining the optimal number of clusters...")
    optimal_k = determine_optimal_clusters(linked, feature_matrix, max_clusters=5)
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
    compression_ratio, original_size, compressed_size = compress_and_evaluate(byte_groups, ordered_indices,
                                                                              component_sizes)
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.4f} ({(compressed_size / original_size) * 100:.2f}%)\n")

    # Compare with original ordering
    print("Comparing with original group ordering (1,2,3,4)...")
    original_ordering = list(range(1, len(byte_groups) + 1))
    original_compression_ratio, orig_size, orig_compressed_size = compress_and_evaluate(byte_groups, original_ordering,
                                                                                        component_sizes)
    print(
        f"Original Ordering Compression ratio: {original_compression_ratio:.4f} ({orig_compressed_size}/{orig_size})\n")

    # --------------------------- Byte Frequency Measurement --------------------------- #
    print("Measuring byte frequencies in each group...")
    byte_frequencies = measure_byte_frequencies(byte_groups)
    plot_byte_frequencies(byte_frequencies, dataset_name, save_dir)
    print()

    # --------------------------- Compression Comparison Plot --------------------------- #
    plot_compression_comparison_path = os.path.join(save_dir, f"{dataset_name}_compression_comparison.png")
    plot_compression_comparison(compression_ratio, original_compression_ratio, dataset_name,
                                plot_compression_comparison_path)

    print("Analysis completed successfully.")


# --------------------------- Execute Script --------------------------- #

if __name__ == "__main__":
    run_analysis()
