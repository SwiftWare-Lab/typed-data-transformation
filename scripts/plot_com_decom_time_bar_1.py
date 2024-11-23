import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file (update the path as needed)
file_path = '/home/jamalids/Documents/2D/data1/Fcbench/llog-gzip-H/combined_median_rows.csv'
df = pd.read_csv(file_path)

# Set up the plot style
sns.set_theme(style="whitegrid")

# Create a 3-in-1 figure with subplots for CompressionRatio, TotalTimeCompressed, and TotalTimeDecompressed
fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# Plot Compression Ratio
sns.barplot(ax=axes[0], x='dataset', y='CompressionRatio', hue='Type', data=df)
axes[0].set_title('Compression Ratio by Dataset and Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
axes[0].legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].tick_params(axis='x', rotation=45)

# Plot Total Time Compressed
# sns.barplot(ax=axes[1], x='dataset', y='TotalTimeCompressed', hue='Type', data=df)
# axes[1].set_title('Total Compression Time by Dataset and Type', fontsize=14, fontweight='bold')
# axes[1].set_xlabel('')
# axes[1].set_ylabel('Total Compression Time (s)', fontsize=12, fontweight='bold')
# axes[1].legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
# axes[1].tick_params(axis='x', rotation=45)

# Plot Total Time Decompressed
# sns.barplot(ax=axes[2], x='dataset', y='TotalTimeDecompressed', hue='Type', data=df)
# axes[2].set_title('Total Decompression Time by Dataset and Type', fontsize=14, fontweight='bold')
# axes[2].set_xlabel('Dataset', fontsize=12, fontweight='bold')
# axes[2].set_ylabel('Total Decompression Time (s)', fontsize=12, fontweight='bold')
# axes[2].legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
# axes[2].tick_params(axis='x', rotation=45)

# Plot Compression Throughput

sns.barplot(ax=axes[1], x='dataset', y='compression_throughput', hue='Type', data=df)
axes[1].set_title('Compression Throughput by Dataset and Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Compression Throughput (GB/s)', fontsize=12, fontweight='bold')
axes[1].legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].tick_params(axis='x', rotation=45)



# Plot Decompression Throughput

sns.barplot(ax=axes[2], x='dataset', y='decompression_throughput', hue='Type', data=df)
axes[2].set_title('Decompression Throughput by Dataset and Type', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Dataset', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Decompression Throughput (GB/s)', fontsize=12, fontweight='bold')
axes[2].legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[2].tick_params(axis='x', rotation=45)




# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/llog-gzip-H/compression_ratios_comparison_by_method-22.png', bbox_inches='tight')
plt.show()
