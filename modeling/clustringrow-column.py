import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import zlib
import itertools
from multiprocessing import Pool


# --------------------------- Entropy Calculation --------------------------- #

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
    """Slide a window across data and compute entropy for each window."""
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


def extract_features(byte_group, window_size=256):
    """
    Extract features from a byte group.

    Features:
        - Average Entropy
        - Standard Deviation of Entropy
        - Maximum Entropy
        - Minimum Entropy
        - Byte Frequency (256 features)

    :param byte_group: Byte array of the group.
    :param window_size: Size of each window for entropy calculation.
    :return: 1D numpy array of features.
    """
    entropies = calculate_entropy_over_data(byte_group, window_size)
    average_entropy = np.mean(entropies) if len(entropies) > 0 else 0
    std_entropy = np.std(entropies) if len(entropies) > 0 else 0
    max_entropy = np.max(entropies) if len(entropies) > 0 else 0
    min_entropy = np.min(entropies) if len(entropies) > 0 else 0

    # Byte frequency
    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)]) if len(byte_group) > 0 else np.zeros(
        256)

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


# --------------------------- Clustering Evaluation Metrics --------------------------- #

def compute_evaluation_metrics(feature_matrix, cluster_labels):
    """
    Compute various clustering evaluation metrics.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param cluster_labels: Cluster labels for each data point.
    :return: Dictionary of evaluation metrics.
    """
    metrics = {}
    try:
        metrics['Silhouette Score'] = silhouette_score(feature_matrix, cluster_labels)
    except:
        metrics['Silhouette Score'] = -1  # Invalid score

    try:
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(feature_matrix, cluster_labels)
    except:
        metrics['Calinski-Harabasz Index'] = -1

    try:
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(feature_matrix, cluster_labels)
    except:
        metrics['Davies-Bouldin Index'] = -1

    return metrics


# --------------------------- Compression Evaluation --------------------------- #

def compress_and_evaluate(byte_groups, ordered_indices):
    """
    Compress byte groups based on ordered indices and evaluate compression ratio.

    :param byte_groups: List of original byte arrays (numpy.ndarray).
    :param ordered_indices: List indicating the order of groups.
    :return: (compression_ratio, original_size, compressed_size)
    """
    # Reorder groups and convert to bytes
    ordered_groups = [byte_groups[i - 1].tobytes() for i in ordered_indices]

    # Merge ordered groups
    merged_bytes = b''.join(ordered_groups)

    # Compress using zlib
    compressed_data = zlib.compress(merged_bytes)

    # Calculate sizes
    original_size = len(merged_bytes)
    compressed_size = len(compressed_data)

    if compressed_size == 0:
        compression_ratio = float('inf')  # Avoid division by zero
    else:
        compression_ratio = original_size / compressed_size  # Higher is better

    return compression_ratio, original_size, compressed_size


# --------------------------- Exhaustive Reordering Search --------------------------- #

def evaluate_permutation(perm, byte_groups):
    """Helper function for multiprocessing."""
    ratio, _, _ = compress_and_evaluate(byte_groups, perm)
    return (perm, ratio)


def find_best_clustering_by_reordering(byte_groups, pool_size=4):
    """
    Try all permutations of the byte_groups to find the ordering with the best (highest)
    compression ratio using parallel processing. Returns (best_order, best_ratio).

    :param byte_groups: List of byte arrays.
    :param pool_size: Number of worker processes.
    :return: (best_order, best_ratio)
    """
    best_ratio = -float('inf')
    best_order = None
    n = len(byte_groups)

    print("Starting exhaustive search for the best ordering (this may take some time)...")
    perms = itertools.permutations(range(1, n + 1))
    with Pool(pool_size) as pool:
        for perm, ratio in pool.starmap(evaluate_permutation, [(perm, byte_groups) for perm in perms]):
            if ratio > best_ratio:
                best_ratio = ratio
                best_order = perm
    print("Exhaustive search completed.")
    return best_order, best_ratio


# --------------------------- Clustering Evaluation Function --------------------------- #

def evaluate_clustering_compression(feature_matrix, byte_groups, cluster_range, save_dir, dataset_name):
    """
    Evaluate clustering options over a range of cluster counts based on compression ratio and clustering metrics.

    :param feature_matrix: 2D numpy array where each row is a data point.
    :param byte_groups: List of byte arrays.
    :param cluster_range: Iterable of cluster counts to evaluate.
    :param save_dir: Directory to save plots and results.
    :param dataset_name: Name of the dataset for titling.
    :return: DataFrame containing compression and clustering metrics for each k.
    """
    results = []

    # Perform hierarchical clustering once for the entire dataset
    linked = perform_hierarchical_clustering(feature_matrix, method='ward', metric='euclidean')

    # Plot dendrogram
    dendrogram_path = os.path.join(save_dir, f"{dataset_name}_dendrogram.png")
    plot_dendrogram_custom(linked, labels=[f'Group {i + 1}' for i in range(len(byte_groups))],
                           save_path=dendrogram_path)

    for k in cluster_range:
        print(f"\nEvaluating for k = {k} clusters...")

        # Assign cluster labels
        cluster_labels = fcluster(linked, k, criterion='maxclust')

        # Group and reorder based on cluster labels
        ordered_indices = group_and_reorder(byte_groups, cluster_labels)

        # Compress and evaluate
        compression_ratio, original_size, compressed_size = compress_and_evaluate(
            byte_groups, ordered_indices)

        # Compute clustering metrics
        clustering_metrics = compute_evaluation_metrics(feature_matrix, cluster_labels)

        # Store the results
        results.append({
            'k': k,
            'Compression Ratio (Original/Compressed)': compression_ratio,
            'Silhouette Score': clustering_metrics['Silhouette Score'],
            'Calinski-Harabasz Index': clustering_metrics['Calinski-Harabasz Index'],
            'Davies-Bouldin Index': clustering_metrics['Davies-Bouldin Index']
        })

        print(
            f"Compression Ratio (Original/Compressed): {compression_ratio:.4f} ({original_size}/{compressed_size} bytes)")
        print(f"Silhouette Score: {clustering_metrics['Silhouette Score']:.4f}")
        print(f"Calinski-Harabasz Index: {clustering_metrics['Calinski-Harabasz Index']:.4f}")
        print(f"Davies-Bouldin Index: {clustering_metrics['Davies-Bouldin Index']:.4f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(os.path.join(save_dir, f"{dataset_name}_compression_evaluation.csv"), index=False)
    print(
        f"\nClustering compression evaluation results saved to {os.path.join(save_dir, f'{dataset_name}_compression_evaluation.csv')}")

    return results_df


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


def plot_best_grouping(best_grouping, dataset_name, save_path):
    """
    Plot the best grouping configuration.

    :param best_grouping: Best grouping tuple (list of tuples).
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
    """
    # Flatten group indices for a simple bar plot
    groups = []
    sizes = []
    for grp in best_grouping:
        group_name = "Cluster " + str(grp[0])  # or more sophisticated naming
        group_size = len(grp)
        groups.append(group_name)
        sizes.append(group_size)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=groups, y=sizes, palette='pastel')
    plt.title(f'Best Grouping Configuration for {dataset_name}')
    plt.xlabel('Byte Groups')
    plt.ylabel('Number of Original Groups')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Best grouping plot saved as {save_path}")


def plot_compression_comparison(optimal_ratio, original_ratio, best_ratio, dataset_name, save_path):
    """
    Plot a comparison of compression ratios between different methods.

    :param optimal_ratio: Compression ratio for optimal clustering.
    :param original_ratio: Compression ratio for original ordering.
    :param best_ratio: Compression ratio for best ordering via exhaustive search.
    :param dataset_name: Name of the dataset for titling.
    :param save_path: File path to save the plot.
    """
    methods = ['Optimal Clustering', 'Original Ordering', 'Best Ordering (Exhaustive)']
    ratios = [optimal_ratio, original_ratio, best_ratio]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=methods, y=ratios, palette='muted')
    plt.ylabel('Compression Ratio (Original / Compressed)')
    plt.title(f'Compression Ratio Comparison for {dataset_name}')
    plt.ylim(0, max(ratios) * 1.2)
    for i, v in enumerate(ratios):
        plt.text(i, v + 0.005, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Compression comparison plot saved as {save_path}")


def plot_elbow_method(compression_results_df, save_path):
    """
    Plot the Elbow Method graph.

    :param compression_results_df: DataFrame containing compression ratios for each k.
    :param save_path: File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='k', y='Compression Ratio (Original/Compressed)', marker='o', data=compression_results_df)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Compression Ratio (Original / Compressed)')
    plt.xticks(compression_results_df['k'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Elbow Method plot saved as {save_path}")


# --------------------------- Helper Functions --------------------------- #

def group_and_reorder(byte_groups, cluster_labels):
    """
    Reorder byte groups based on cluster assignments.

    :param byte_groups: List of byte arrays.
    :param cluster_labels: Cluster labels for each byte group.
    :return: List of ordered group indices.
    """
    # Pair each group with its cluster label
    paired = list(zip(range(1, len(byte_groups) + 1), cluster_labels))
    # Sort groups by cluster label
    paired_sorted = sorted(paired, key=lambda x: x[1])
    # Extract ordered group indices
    ordered_indices = [item[0] for item in paired_sorted]
    return ordered_indices


# --------------------------- Main Function --------------------------- #

def run_analysis():
    # --------------------------- Configuration --------------------------- #
    dataset_path = r"C:\Users\jamalids\Downloads\dataset\High-Entropy\jw_mirimage_f32.tsv"  # Update path as needed
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    save_dir = f"C:\\Users\\jamalids\\Downloads\\{dataset_name}_analysisF"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    window_size = 65536
    component_sizes = [1, 1, 1, 1]  # Create 4 byte groups

    # Cluster Range
    cluster_range = range(2, 5)  # Evaluate k=2,3,4

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
        # Convert numeric data to float32, then flatten and convert to bytes
        byte_data = ts_data[byte_columns].to_numpy().astype(np.float32)
        byte_data = byte_data.flatten(order='C').tobytes()

        # Split into 4 groups
        byte_groups = split_bytes_into_components(byte_data, component_sizes)
        print(f"Data preparation completed. Created {len(byte_groups)} byte groups.\n")
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

    # --------------------------- Feature Scaling and Dimensionality Reduction --------------------------- #
    print("Scaling and reducing feature dimensions...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Ensure n_components does not exceed number of samples
    n_components = min(4, feature_matrix.shape[0])  # min(n_samples=4, n_features=260)=4
    pca = PCA(n_components=n_components, random_state=42)
    feature_matrix_preprocessed = pca.fit_transform(scaled_features)
    print(f"Feature matrix shape after PCA: {feature_matrix_preprocessed.shape}\n")

    # --------------------------- Evaluate Clustering Options Based on Compression --------------------------- #
    print("Evaluating clustering options based on compression ratio and clustering metrics...")
    compression_results_df = evaluate_clustering_compression(
        feature_matrix_preprocessed, byte_groups, cluster_range, save_dir, dataset_name)

    # --------------------------- Plot Elbow Method --------------------------- #
    print("\nPlotting the Elbow Method...")
    plot_elbow_method_path = os.path.join(save_dir, f"{dataset_name}_elbow_method.png")
    plot_elbow_method(compression_results_df, plot_elbow_method_path)

    # --------------------------- Determine Optimal k --------------------------- #
    print("\nDetermining the optimal number of clusters based on multiple criteria...")

    # The criteria include:
    # - Highest Compression Ratio
    # - Highest Silhouette Score
    # - Highest Calinski-Harabasz Index
    # - Lowest Davies-Bouldin Index

    # Normalize metrics
    metrics_to_normalize = ['Compression Ratio (Original/Compressed)', 'Silhouette Score', 'Calinski-Harabasz Index',
                            'Davies-Bouldin Index']
    normalized_df = compression_results_df.copy()

    for metric in metrics_to_normalize:
        if metric != 'Davies-Bouldin Index':  # For metrics where higher is better
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            if max_val - min_val != 0:
                normalized_df[f'Normalized {metric}'] = (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'Normalized {metric}'] = 0
        else:
            # For Davies-Bouldin Index, lower is better
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            if max_val - min_val != 0:
                normalized_df[f'Normalized {metric}'] = (max_val - normalized_df[metric]) / (max_val - min_val)
            else:
                normalized_df[f'Normalized {metric}'] = 0

    # Compute a composite score with assigned weights
    # Corrected column name for Compression Ratio
    try:
        normalized_df['Composite Score'] = (
                0.4 * normalized_df['Normalized Compression Ratio (Original/Compressed)'] +
                0.3 * normalized_df['Normalized Silhouette Score'] +
                0.2 * normalized_df['Normalized Calinski-Harabasz Index'] +
                0.1 * normalized_df['Normalized Davies-Bouldin Index']
        )
    except KeyError as e:
        print(f"Error in composite score calculation: {e}")
        print("Available columns:", normalized_df.columns)
        return

    # Select the k with the highest composite score
    optimal_row = normalized_df.loc[normalized_df['Composite Score'].idxmax()]
    optimal_k = int(optimal_row['k'])

    print(f"Optimal number of clusters based on multiple criteria: {optimal_k}")

    # --------------------------- Assign Optimal Cluster Labels --------------------------- #
    print(f"\nAssigning cluster labels based on optimal k = {optimal_k}...")
    # Perform hierarchical clustering again for optimal k
    linked_optimal = perform_hierarchical_clustering(feature_matrix_preprocessed, method='ward', metric='euclidean')
    cluster_labels = fcluster(linked_optimal, optimal_k, criterion='maxclust')

    # --------------------------- Grouping and Reordering (Optimal k) --------------------------- #
    print("Grouping and reordering byte groups based on optimal clustering...")
    cluster_based_ordered_indices = group_and_reorder(byte_groups, cluster_labels)
    print(f"Cluster-based group order (Group indices): {cluster_based_ordered_indices}\n")

    # Plot grouping by cluster
    best_grouping = []
    for grp in sorted(set(cluster_labels)):
        indices = [i + 1 for i, label in enumerate(cluster_labels) if label == grp]
        best_grouping.append(tuple(indices))
    plot_best_grouping(best_grouping, dataset_name, os.path.join(save_dir, f"{dataset_name}_best_grouping.png"))

    # --------------------------- Compression Evaluation: Optimal Ordering --------------------------- #
    print("Evaluating compression ratio with optimal group order...")
    cluster_based_ratio, cb_original_size, cb_compressed_size = compress_and_evaluate(
        byte_groups, cluster_based_ordered_indices)
    print(f"Optimal ordering based on clustering:")
    print(f"  Original size: {cb_original_size} bytes")
    print(f"  Compressed size: {cb_compressed_size} bytes")
    print(
        f"  Compression ratio (Original/Compressed): {cluster_based_ratio:.4f} ({cb_original_size}/{cb_compressed_size} bytes)\n")

    # --------------------------- Compression Evaluation: Original Ordering --------------------------- #
    print("Evaluating compression ratio with original group ordering (1,2,3,4)...")
    original_ordering = list(range(1, len(byte_groups) + 1))
    original_compression_ratio, orig_size, orig_compressed_size = compress_and_evaluate(
        byte_groups, original_ordering)
    print(f"Original ordering:")
    print(f"  Original size: {orig_size} bytes")
    print(f"  Compressed size: {orig_compressed_size} bytes")
    print(
        f"  Compression ratio (Original/Compressed): {original_compression_ratio:.4f} ({orig_size}/{orig_compressed_size} bytes)\n")

    # --------------------------- Exhaustive Best Reordering --------------------------- #
    print("Searching for the best possible ordering by exhaustive reordering...")
    best_order, best_ratio = find_best_clustering_by_reordering(byte_groups)
    print(f"Best ordering found by brute force: {best_order}")
    print(f"Best compression ratio (Original/Compressed): {best_ratio:.4f}\n")

    # --------------------------- Compression Comparison Plot --------------------------- #
    # Compare optimal clustering ratio, original ratio, and exhaustive best ratio
    plot_compression_comparison_path = os.path.join(save_dir, f"{dataset_name}_compression_comparison.png")
    plot_compression_comparison(cluster_based_ratio, original_compression_ratio, best_ratio, dataset_name,
                                plot_compression_comparison_path)

    print("Analysis completed successfully.")


# --------------------------- Execute Script --------------------------- #

if __name__ == "__main__":
    run_analysis()
