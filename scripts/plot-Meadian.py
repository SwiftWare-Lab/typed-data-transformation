import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


# Load the processed CSV file
file_path ='/home/jamalids/Documents/results/combine/combined_median.csv'
data = pd.read_csv(file_path)

# Create a new column for combined labels
data['Dataset_ComponentSizes'] = data['DatasetName'].astype(str) + ' (' + data['ConfigString'].astype(str) + ')'

# Ensure dataset names are categorical for proper x-axis ordering
data['Dataset_ComponentSizes'] = pd.Categorical(data['Dataset_ComponentSizes'], categories=data['Dataset_ComponentSizes'].unique(), ordered=True)

# Sort the data by dataset for consistent plotting
data = data.sort_values('Dataset_ComponentSizes')

# Metrics for Full and Parallel
parallel_data = data[data['RunType'] == 'Parallel']
full_data = data[data['RunType'] == 'Full']

# Aggregate data by taking the median for each dataset to ensure one-to-one mapping
parallel_data_grouped = parallel_data.groupby("Dataset_ComponentSizes").median(numeric_only=True).reset_index()
full_data_grouped = full_data.groupby("Dataset_ComponentSizes").median(numeric_only=True).reset_index()

# Ensure the same datasets exist in both sets
common_datasets = list(set(parallel_data_grouped['Dataset_ComponentSizes']) & set(full_data_grouped['Dataset_ComponentSizes']))
parallel_data_grouped = parallel_data_grouped[parallel_data_grouped['Dataset_ComponentSizes'].isin(common_datasets)]
full_data_grouped = full_data_grouped[full_data_grouped['Dataset_ComponentSizes'].isin(common_datasets)]

# Re-sort data for consistent x-axis alignment
parallel_data_grouped = parallel_data_grouped.sort_values('Dataset_ComponentSizes')
full_data_grouped = full_data_grouped.sort_values('Dataset_ComponentSizes')

# X-axis positions for datasets
x = np.arange(len(common_datasets))  # the label locations
width = 0.35  # the width of the bars

# Create subplots in a single frame
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot 1: Compression Ratio
bars_parallel_0 = axs[0].bar(x - width / 2, parallel_data_grouped['CompressionRatio'], width, label='Decompose', color='red')
bars_full_0 = axs[0].bar(x + width / 2, full_data_grouped['CompressionRatio'], width, label='zstd', color='cyan')
axs[0].set_title('Compression Ratio by Dataset')
axs[0].set_ylabel('Compression Ratio')
axs[0].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_0, parallel_data_grouped['Threads']):
    axs[0].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Plot 2: Compression Throughput
bars_parallel_1 = axs[1].bar(x - width / 2, parallel_data_grouped['CompressionThroughput'], width, label='Decompose+zstd', color='red')
bars_full_1 = axs[1].bar(x + width / 2, full_data_grouped['CompressionThroughput'], width, label='zstd', color='cyan')
axs[1].set_title('Compression Throughput by Dataset')
axs[1].set_ylabel('Compression Throughput (GB/s)')
axs[1].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_1, parallel_data_grouped['Threads']):
    axs[1].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Plot 3: Decompression Throughput
bars_parallel_2 = axs[2].bar(x - width / 2, parallel_data_grouped['DecompressionThroughput'], width, label='Decompose+zstd', color='red')
bars_full_2 = axs[2].bar(x + width / 2, full_data_grouped['DecompressionThroughput'], width, label='zstd', color='cyan')
axs[2].set_title('Decompression Throughput by Dataset')
axs[2].set_ylabel('Decompression Throughput (GB/s)')
axs[2].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_2, parallel_data_grouped['Threads']):
    axs[2].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Set common x-axis labels and positions
axs[2].set_xlabel('Dataset (ComponentSizes)')
plt.xticks(x, common_datasets, rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.savefig("/home/jamalids/Documents/results/combine/compression_analysis.png", dpi=300, bbox_inches='tight')

