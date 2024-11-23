import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file (update the path as needed)
file_path = '/home/jamalids/Documents/2D/data1/Fcbench/logH-22/combined_data_1.csv'
df = pd.read_csv(file_path)

# Set up the plot style
sns.set_theme(style="whitegrid")

# Plot and save each metric separately

# 1. Compression Ratio
plt.figure(figsize=(10, 6))
sns.barplot(x='dataset', y='CompressionRatio', hue='Type', data=df)
plt.title('Compression Ratio by Dataset and Type', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('Compression Ratio', fontsize=12, fontweight='bold')
plt.legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/logH-22/compression_ratio.png', bbox_inches='tight')
plt.show()

# 2. Total Time Compressed
plt.figure(figsize=(10, 6))
sns.barplot(x='dataset', y='TotalTimeCompressed', hue='Type', data=df)
plt.title('Total Compression Time by Dataset and Type', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('Total Compression Time (s)', fontsize=12, fontweight='bold')
plt.legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/logH-22/total_time_compressed.png', bbox_inches='tight')
plt.show()

# 3. Total Time Decompressed
plt.figure(figsize=(10, 6))
sns.barplot(x='dataset', y='TotalTimeDecompressed', hue='Type', data=df)
plt.title('Total Decompression Time by Dataset and Type', fontsize=14, fontweight='bold')
plt.xlabel('Dataset', fontsize=12, fontweight='bold')
plt.ylabel('Total Decompression Time (s)', fontsize=12, fontweight='bold')
plt.legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/logH-22/total_time_decompressed.png', bbox_inches='tight')
plt.show()

# 4. Compression Throughput
plt.figure(figsize=(10, 6))
sns.barplot(x='dataset', y='compression_throughput', hue='Type', data=df)
plt.title('Compression Throughput by Dataset and Type', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('Compression Throughput (GB/s)', fontsize=12, fontweight='bold')
plt.legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/logH-22/compression_throughput.png', bbox_inches='tight')
plt.show()

# 5. Decompression Throughput
plt.figure(figsize=(10, 6))
sns.barplot(x='dataset', y='decompression_throughput', hue='Type', data=df)
plt.title('Decompression Throughput by Dataset and Type', fontsize=14, fontweight='bold')
plt.xlabel('Dataset', fontsize=12, fontweight='bold')
plt.ylabel('Decompression Throughput (GB/s)', fontsize=12, fontweight='bold')
plt.legend(title='Compression Type', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/2D/data1/Fcbench/logH-22/decompression_throughput.png', bbox_inches='tight')
plt.show()
