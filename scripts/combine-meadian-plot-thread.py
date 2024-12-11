import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the processed CSV file
file_path = '/home/jamalids/Documents/2D/data1/Fcbench/logCpp/log-zstd-thread/max_comp_ratio_parallel_full_32L.csv'
data = pd.read_csv(file_path)

# Create a new column for combined labels
data['Dataset_ComponentSizes'] = data['dataset'].astype(str) + ' (' + data['ComponentSizes'].astype(str) + ')'

# Ensure dataset names are categorical for proper x-axis ordering
data['Dataset_ComponentSizes'] = pd.Categorical(data['Dataset_ComponentSizes'], categories=data['Dataset_ComponentSizes'].unique(), ordered=True)

# Sort the data by dataset for consistent plotting
data = data.sort_values('Dataset_ComponentSizes')

# Unique datasets for the x-axis
datasets = data['Dataset_ComponentSizes'].unique()

# Metrics for Full and Parallel
parallel_data = data[data['RunType'] == 'Parallel']
full_data = data[data['RunType'] == 'Full']

# X-axis positions for datasets
x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

# Create subplots in a single frame
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot 1: Compression Ratio
bars_parallel_0 = axs[0].bar(x - width / 2, parallel_data['CompressionRatio'], width, label='Decompose+zstd', color='red')
bars_full_0 = axs[0].bar(x + width / 2, full_data['CompressionRatio'], width, label='zstd', color='cyan')
axs[0].set_title('Compression Ratio by Dataset')
axs[0].set_ylabel('Compression Ratio')
axs[0].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_0, parallel_data['ThreadCount']):
    axs[0].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Plot 2: Compression Throughput
bars_parallel_1 = axs[1].bar(x - width / 2, parallel_data['CompressionThroughput'], width, label='Decompose+zstd', color='red')
bars_full_1 = axs[1].bar(x + width / 2, full_data['CompressionThroughput'], width, label='zstd', color='cyan')
axs[1].set_title('Compression Throughput by Dataset')
axs[1].set_ylabel('Compression Throughput (GB/s)')
axs[1].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_1, parallel_data['ThreadCount']):
    axs[1].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Plot 3: Decompression Throughput
bars_parallel_2 = axs[2].bar(x - width / 2, parallel_data['DecompressionThroughput'], width, label='Decompose+zstd', color='red')
bars_full_2 = axs[2].bar(x + width / 2, full_data['DecompressionThroughput'], width, label='zstd', color='cyan')
axs[2].set_title('Decompression Throughput by Dataset')
axs[2].set_ylabel('Decompression Throughput (GB/s)')
axs[2].legend()

# Annotate ThreadCount for parallel bars
for bar, thread_count in zip(bars_parallel_2, parallel_data['ThreadCount']):
    axs[2].text(
        bar.get_x() + bar.get_width() / 2,  # x position
        bar.get_height() + 0.02,  # y position above the bar
        f'Th:{int(thread_count)}',  # ThreadCount label
        ha='center', va='bottom', fontsize=8
    )

# Set common x-axis labels and positions
axs[2].set_xlabel('Dataset (ComponentSizes)')
plt.xticks(x, datasets, rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Save the plots to a file
output_path = '/home/jamalids/Documents/2D/data1/Fcbench/logCpp/log-zstd-thread/compression_analysis_plots32L.png'
plt.savefig(output_path)

# Show the plots
plt.show()

print(f'Plots saved to {output_path}')
