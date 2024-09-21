import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
file_path = '/home/jamalids/Documents/compression-part3/big-data-compression/modeling/Decom+zstd+gzip(all).csv'  # Change this to your actual path
df = pd.read_csv(file_path)
gmean_com_ratio_zstd = np.power(np.prod(df['comp_ratio_zstd_default']), 1/len(df['comp_ratio_zstd_default']))
gmean_decom = np.power(np.prod(df['max_Decom+zstd_com_ratio']), 1/len(df['max_Decom+zstd_com_ratio']))

print(gmean_com_ratio_zstd)
print(gmean_decom)

# Adjusting the plot to add space between the bars of different datasets
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart data
bar_width = 0.15
index = range(len(df) * 2)  # Adding space by expanding the range

# Plotting the bars with space between datasets
ax1.bar(index[::2], df['comp_ratio_zstd_default'], bar_width, label='Zstd Default')
ax1.bar([i + bar_width for i in index[::2]], df['comp_ratio_l22'], bar_width, label='Zstd_22')
ax1.bar([i + 2*bar_width for i in index[::2]], df['max_Decom+zstd_com_ratio'], bar_width, label='Decompose+Zstd ')
ax1.bar([i + 3*bar_width for i in index[::2]], df['max_Decom+zstd_22_com_ratio'], bar_width, label='Decompose+Zstd_22')
ax1.bar([i + 4*bar_width for i in index[::2]], df['comp_ratio_gzip'], bar_width, label='comp_ratio_gzip')
ax1.bar([i + 5*bar_width for i in index[::2]], df['max_Decom+gzip_com_ratio'], bar_width, label='Decom+gzip')

# Labels for bar chart
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Compression Ratios ')
ax1.set_xticks([i + 2*bar_width for i in index[::2]])
ax1.set_xticklabels(df['dataset_name'], rotation=90)
ax1.legend(loc='upper left')

# Line plot for entropy_all with space
ax2 = ax1.twinx()
ax2.plot(index[::2], df['entropy_all'], color='r', marker='o', label='Entropy of dataset')
ax2.plot(index[::2], df['sum_entropy_b3_zstd'], color='b', marker='o', label='Decomposs+zstd+Entropy')
ax2.plot(index[::2], df['sum_entropy_b3_gzip'], color='g', marker='o', label='Decomposs+gzip+Entropy')
ax2.set_ylabel('Entropy')
ax2.legend(loc='upper right')

plt.title('Compression Ratios and Entropies for Different Datasets')
plt.tight_layout()
plt.savefig("zstd_gzip_entropy.png")

plt.show()