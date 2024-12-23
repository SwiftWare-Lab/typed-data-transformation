import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the processed CSV file
file_path = '/home/jamalids/Documents/2D/CR-Ct-DT/zlib-CPP/max_comp_ratio_parallel_full.csv'
data = pd.read_csv(file_path)

# Ensure dataset names are categorical for proper x-axis ordering
data['dataset'] = pd.Categorical(data['dataset'], categories=data['dataset'].unique(), ordered=True)

# Sort the data by dataset for consistent plotting
data = data.sort_values('dataset')

# Separate data by RunType
parallel_data = data[data['RunType'] == 'Parallel'].set_index('dataset')
full_data = data[data['RunType'] == 'Full'].set_index('dataset')

# Merge parallel and full data for accurate alignment
merged_data = parallel_data.join(full_data, lsuffix='_parallel', rsuffix='_full')

# X-axis positions for datasets
x = np.arange(len(merged_data))  # the label locations
width = 0.35  # the width of the bars

# Define font sizes
title_fontsize = 18
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 12

# Create subplots in a single frame
fig, axs = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Adjust matplotlib defaults to set font sizes globally
plt.rcParams.update({'font.size': tick_fontsize})

# Plot 1: Compression Ratio with logarithmic scale
axs[0].bar(x, merged_data['CompressionRatio_parallel'], width, label='Decompose+bz2 (Parallel)', color='red', alpha=0.8)
axs[0].bar(x, merged_data['CompressionRatio_full'], width, label='Standard bz2 (Full)', color='blue', alpha=0.8)
axs[0].set_ylabel('log(Compression Ratio)', fontsize=label_fontsize)
axs[0].set_title('Compression Ratio by Dataset', fontsize=title_fontsize)
axs[0].set_yscale('log')  # Set the y-axis to logarithmic scale
axs[0].legend(fontsize=legend_fontsize)

# Plot 2: Compression Throughput
axs[1].bar(x, merged_data['CompressionThroughput_parallel'], width, label='Decompose+bz2', color='red', alpha=0.8)
axs[1].bar(x, merged_data['CompressionThroughput_full'], width, label='Standard bz2', color='blue', alpha=0.8)
axs[1].set_ylabel('Compression Throughput (GB/s)', fontsize=label_fontsize)
axs[1].set_title('Compression Throughput by Dataset', fontsize=title_fontsize)
axs[1].legend(fontsize=legend_fontsize)

# Plot 3: Decompression Throughput
axs[2].bar(x, merged_data['DecompressionThroughput_parallel'], width, label='Decompose+bz2 ', color='red', alpha=0.8)
#axs[2].bar(x, merged_data['DecompressionThroughput_full'], width, label='Standard bz2', color='cyan', alpha=0.8)
axs[2].bar(x, merged_data['DecompressionThroughput_full'], width, label='Standard bz2', color='blue', alpha=0.8)
axs[2].set_xlabel('Dataset', fontsize=label_fontsize)
axs[2].set_ylabel('Decompression Throughput (GB/s)', fontsize=label_fontsize)
axs[2].set_title('Decompression Throughput by Dataset', fontsize=title_fontsize)
axs[2].legend(fontsize=legend_fontsize)

# Set x-axis labels with rotation
plt.xticks(x, merged_data.index, rotation=45, ha='right', fontsize=tick_fontsize)

# Adjust layout and show plot
plt.tight_layout()

# Save the plots to a file
output_path = '/home/jamalids/Documents/2D/CR-Ct-DT/zlib-CPP/compression_analysis_plots_zlib.png'
plt.savefig(output_path)
plt.show()

print(f'Plots saved to {output_path}')
