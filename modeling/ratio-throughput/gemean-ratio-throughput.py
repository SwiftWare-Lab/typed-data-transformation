import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def geometric_mean(series):
    """
    Computes the geometric mean of a Pandas Series
    containing only positive values.
    """
    return np.exp(np.log(series).mean())

data = pd.read_csv('/home/jamalids/Documents/combined_all_data.csv')

thread_values = [1, 8, 16]

special_blocks = [
    (640 * 1024,       'L1'),   #  640 * 1024
    (24 * 1024 * 1024, 'L2'),   # 24 * 1024 * 1024
    (30 * 1024 * 1024, 'L3')    # 30 * 1024 * 1024
]

# Create a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for idx, thread_val in enumerate(thread_values):
    filtered = data[
        (data['RunType'] == 'Decompose_Chunk_Parallel') &
        (data['Threads'] == thread_val)
    ]

    if filtered.empty:
        axes[idx].set_title(f'Threads = {thread_val}\n(No Data)')
        continue

    grouped = filtered.groupby('BlockSize').agg({
        'CompressionThroughput': geometric_mean,
        'DecompressionThroughput': geometric_mean,
        'CompressionRatio': geometric_mean
    }).reset_index()

    grouped.sort_values('BlockSize', inplace=True)

    grouped['Avg_Throughput'] = (
        grouped['CompressionThroughput'] +
        grouped['DecompressionThroughput']
    ) / 2.0

    # Simple score to identify best block size
    grouped['Score'] = grouped['Avg_Throughput'] + grouped['CompressionRatio']
    best_row = grouped.loc[grouped['Score'].idxmax()]
    best_blocksize = best_row['BlockSize']

    print(f"[Threads = {thread_val}] Best BlockSize: {best_blocksize}")
    print("Metrics for Best BlockSize:")
    print(best_row)
    print("-----------")

    x = np.arange(len(grouped))

    ax1 = axes[idx]      # Left y-axis (Compression Ratio)
    ax2 = ax1.twinx()    # Right y-axis (Throughput)

    # Plot on left axis
    ax1.plot(x, grouped['CompressionRatio'], 'o-', label='Compression Ratio')
    ax1.set_ylabel('Gmean Compression Ratio')
    # Move the left y-axis label slightly to the left
    ax1.yaxis.set_label_coords(-0.10, 0.5)

    # Plot on right axis
    ax2.plot(x, grouped['Avg_Throughput'], 's-', label='Avg Throughput (GB/s)', color='red')
    ax2.set_ylabel('Avg (gemean Throughput)')
    # Move the right y-axis label slightly to the right
    ax2.yaxis.set_label_coords(1.10, 0.5)

    ax1.set_title(f'Threads = {thread_val}')

    # Custom x-axis tick labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped['BlockSize'].astype(str), rotation=45, ha='right')
    ax1.set_xlabel('BlockSize')

    # Draw vertical lines for L1, L2, L3
    for sb_size, sb_label in special_blocks:
        if sb_size in grouped['BlockSize'].values:
            x_loc = grouped.index[grouped['BlockSize'] == sb_size][0]
            ax1.axvline(x=x_loc, color='gray', linestyle='--', alpha=0.8)
            y_top = grouped['CompressionRatio'].max() * 1.05
            ax1.text(x_loc, y_top, sb_label, rotation=90,
                     va='bottom', ha='center', color='gray', fontsize=9)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

# Adjust space between subplots to reduce label overlap
plt.subplots_adjust(wspace=0.3)

plt.tight_layout()

output_plot_path = '/home/jamalids/Documents/plot_special_blocksize_gmean_adjusted.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Plot saved to {output_plot_path}")
