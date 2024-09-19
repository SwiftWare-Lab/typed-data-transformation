import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '/home/jamalids/Documents/compression-part3/big-data-compression/modeling/zstd.csv'
df = pd.read_csv(file_path)

# Extract necessary data from the dataframe
dataset_names = df['dataset_name']
XOR_comp_default = df['XOR_comp_ratio_zstd_default']
XOR_comp_22 = df['XOR_comp_ratio_zstd_22']
comp_default = df['comp_ratio_zstd_default']
comp_22 = df['comp_ratio_zstd_22']
entropy_float = df['entropy_float']
entropy_float_XOR = df['entropy_float_XOR']

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar chart for compression ratios
bar_width = 0.2
index = np.arange(len(dataset_names))

ax1.bar(index - bar_width*1.5, XOR_comp_default, bar_width, label='XOR Comp Ratio Zstd Default')
ax1.bar(index - bar_width*0.5, XOR_comp_22, bar_width, label='XOR Comp Ratio Zstd 22')
ax1.bar(index + bar_width*0.5, comp_default, bar_width, label='Comp Ratio Zstd Default')
ax1.bar(index + bar_width*1.5, comp_22, bar_width, label='Comp Ratio Zstd 22')

# Customize the x-axis and labels
ax1.set_xlabel('Dataset Name')
ax1.set_ylabel('Compression Ratios')
ax1.set_title('Compression Ratios and Entropy for Different Datasets')
ax1.set_xticks(index)
ax1.set_xticklabels(dataset_names, rotation=90)

# Second y-axis for two entropy values
ax2 = ax1.twinx()
ax2.plot(index, entropy_float, color='red', marker='o', label='Entropy Float', linestyle='-', linewidth=2)
ax2.plot(index, entropy_float_XOR, color='blue', marker='x', label='Entropy Float XOR', linestyle='--', linewidth=2)
ax2.set_ylabel('Entropy')

# Adding legends for both bar and line plots
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Tight layout for better display
plt.tight_layout()
plt.savefig("zstd.png")

# Show plot
plt.show()
