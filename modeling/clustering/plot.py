import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # File paths (adjust as needed)
    combine_file = '/home/jamalids/Documents/all-config-cluster/logs-zstd/combine.csv'
    clustering_file = '/home/jamalids/Documents/all-config-cluster/logs-zstd/clustering_vs_exhaustive_all.csv'

    # Load CSV files
    df_combine = pd.read_csv(combine_file)
    df_cluster = pd.read_csv(clustering_file)

    # Tag each DataFrame with a source label (optional)
    df_combine['source'] = 'combine'
    df_cluster['source'] = 'clustering'

    # --- Prepare combine data ---
    # Sort combine data by 'dataset name' and 'decomposition'
    df_combine = df_combine.sort_values(['dataset name', 'decomposition']).reset_index(drop=True)
    # Create a global x-axis index for combine points
    df_combine['x'] = np.arange(len(df_combine))

    # Map decomposition values to a numeric index for color-coding
    if df_combine['decomposition'].dtype == object:
        unique_configs = sorted(df_combine['decomposition'].unique())
        config_mapping = {config: i for i, config in enumerate(unique_configs)}
        df_combine['config_num'] = df_combine['decomposition'].map(config_mapping)
    else:
        df_combine['config_num'] = df_combine['decomposition']

    # --- Highlighting ---
    # For each dataset, get the best clustering ratio from df_cluster, then find the combine point
    # (from df_combine) whose "decomposed zstd compression ratio" is closest to that best clustering ratio.
    highlight_indices = []
    for ds in df_cluster['dataset name'].unique():
        # Subset clustering data for this dataset
        cluster_subset = df_cluster[df_cluster['dataset name'] == ds]
        if cluster_subset.empty:
            continue
        best_clust_ratio = cluster_subset['clustering_ratio'].max()

        # Subset combine data for this dataset
        combine_subset = df_combine[df_combine['dataset name'] == ds]
        if combine_subset.empty:
            continue

        # Calculate absolute difference between each combine ratio and the best clustering ratio
        diff = abs(combine_subset['decomposed zstd compression ratio'] - best_clust_ratio)
        diff = diff.dropna()
        if diff.empty:
            continue
        idx = diff.idxmin()
        highlight_indices.append(idx)

    df_highlight = df_combine.loc[highlight_indices]

    # --- Plotting ---
    plt.figure(figsize=(14, 8))

    # Plot combine points with a colormap (using decomposition values)
    scatter1 = plt.scatter(
        df_combine['x'],
        df_combine['decomposed zstd compression ratio'],
        c=df_combine['config_num'],
        cmap='tab20',  # Use a colormap with many distinct colors
        s=50,
        label='Combine (Exhaustive)',
        zorder=1
    )

    # Add a colorbar to show the mapping for decomposition
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Decomposition Config Index')

    # Overlay highlighted points (the combine points matching best clustering ratio) with a large black circle
    plt.scatter(
        df_highlight['x'],
        df_highlight['decomposed zstd compression ratio'],
        facecolors='none',
        edgecolors='black',
        marker='o',
        s=200,
        linewidth=3,
        label='Match (Best Clustering Ratio)',
        zorder=4
    )

    # Set x-axis ticks to show dataset names (positioned at the average x for each dataset)
    unique_datasets = df_combine['dataset name'].unique()
    tick_positions = []
    tick_labels = []
    for ds in unique_datasets:
        subset = df_combine[df_combine['dataset name'] == ds]
        tick_positions.append(subset['x'].mean())
        tick_labels.append(ds)

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.xlabel('Dataset')
    plt.ylabel('Decomposed Zstd Compression Ratio')
    plt.title('Comparison: Best Clustering vs. Exhaustive Compression Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig('compare_clustering_vs_exhaustive.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
