import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the processed CSV file
file_path = '/media/samira/sa/result-compression/logzstd/max_comp_ratio_parallel_full_32L.csv'
data = pd.read_csv(file_path)

# Ensure dataset names are categorical for proper x-axis ordering
data['dataset'] = pd.Categorical(data['dataset'], categories=data['dataset'].unique(), ordered=True)

# Sort the data by dataset for consistent plotting
data = data.sort_values('dataset')

# Unique datasets for the x-axis
datasets = data['dataset'].unique()

# Metrics for Full and Parallel
parallel_data = data[data['RunType'] == 'Parallel']
full_data = data[data['RunType'] == 'Full']

# X-axis positions for datasets
x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

# Create subplots in a single frame
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot 1: Compression Ratio
axs[0].bar(x - width / 2, parallel_data['CompressionRatio'], width, label='Decompose+zstd', color='blue')
axs[0].bar(x + width / 2, full_data['CompressionRatio'], width, label='zstd', color='cyan')
axs[0].set_title('Compression Ratio by Dataset')
axs[0].set_ylabel('Compression Ratio')
axs[0].legend()

# Plot 2: Compression Throughput
axs[1].bar(x - width / 2, parallel_data['CompressionThroughput'], width, label='Decompose+zstd', color='green')
axs[1].bar(x + width / 2, full_data['CompressionThroughput'], width, label='zstd', color='lightgreen')
axs[1].set_title('Compression Throughput by Dataset')
axs[1].set_ylabel('Compression Throughput (GB/s)')
axs[1].legend()

# Plot 3: Decompression Throughput
axs[2].bar(x - width / 2, parallel_data['DecompressionThroughput'], width, label='Decompose+zstd', color='orange')
axs[2].bar(x + width / 2, full_data['DecompressionThroughput'], width, label='zstd', color='gold')
axs[2].set_title('Decompression Throughput by Dataset')
axs[2].set_ylabel('Decompression Throughput (GB/s)')
axs[2].legend()

# Set common x-axis labels and positions
axs[2].set_xlabel('Dataset')
plt.xticks(x, datasets, rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Save the plots to a file
output_path = '/media/samira/sa/result-compression/logzstd/compression_analysis_plots32L.png'
plt.savefig(output_path)

# Show the plots
plt.show()

print(f'Plots saved to {output_path}')
