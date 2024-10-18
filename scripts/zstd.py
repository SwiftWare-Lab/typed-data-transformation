import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
file_path = "/home/jamalids/Documents/compression-part3/final/final/L_E_agg_P.csv"  # Change this to your actual path
df = pd.read_csv(file_path)
gmean_com_ratio_zstd = np.power(np.prod(df['comp_ratio_zstd_default']), 1/len(df['comp_ratio_zstd_default']))
gmean_decom = np.power(np.prod(df['max_Decom+zstd_com_ratio']), 1/len(df['max_Decom+zstd_com_ratio']))

print(gmean_com_ratio_zstd)
print(gmean_decom)

# Adjusting the plot to add space between the bars of different datasets
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart data
bar_width = 0.20
index = range(len(df) * 2)  # Adding space by expanding the range

# Defining custom colors for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

# Plotting bars with custom colors
ax1.bar(index[::2], df['comp_ratio_zstd_default'], bar_width, label='comp_ratio_Zstd Default', color=colors[0])
ax1.bar([i + bar_width for i in index[::2]], df['comp_ratio_l22'], bar_width, label='comp_ratio_Zstd_22', color=colors[1])
ax1.bar([i + 2*bar_width for i in index[::2]], df['comp_ratio_gzip'], bar_width, label='comp_ratio_gzip', color=colors[2])
ax1.bar([i + 3*bar_width for i in index[::2]], df['max_Decom+zstd_com_ratio'], bar_width, label='SpDecomp+comp_ratio_Zstd', color=colors[3])
ax1.bar([i + 4*bar_width for i in index[::2]], df['max_Decom+zstd_22_com_ratio'], bar_width, label='SpDecomp+comp_ratio_22', color=colors[4])
ax1.bar([i + 5*bar_width for i in index[::2]], df['max_Decom+gzip_com_ratio'], bar_width, label='SpDecomp+comp_ratio_gzip', color=colors[5])

#ax1.bar([i + 6*bar_width for i in index[::2]], df['max_com_ratio'], bar_width, label='Decomp+our method')
#ax1.bar([i + 7*bar_width for i in index[::2]], df['Non_uniform_1x4'], bar_width, label='Non_uniform_1x4')
ax1.set_yscale('log')
# Labels for bar chart
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Compression Ratios ')
ax1.set_xticks([i + 2*bar_width for i in index[::2]])
ax1.set_xticklabels(df['dataset_name'], rotation=90)
# Move the bar chart legend outside the plot (to the right)
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Compression Methods")



# Line plot for entropy_all with space
ax2 = ax1.twinx()
ax2.plot(index[::2], df['entropy_float'], color='r', marker='o', label='Entropy of dataset')
ax2.plot(index[::2], df['entropy_remainig'], color='b', marker='o', label='entropy_remainig')
#ax2.plot(index[::2], df['sum_entropy_b3_gzip'], color='g', marker='o', label='Decomposs+gzip+Entropy')
#ax2.plot(index[::2], df['sum_entropy_sh_b3'], color='purple', marker='o', label='Decomposs+zstd+Entropy_Multi-Component')
#ax2.plot(index[::2], df['sum_entropy_b3_sh_gzip'], color='yellow', marker='o', label='Decomposs+gzip+Entropy_Multi-Component')
ax2.set_ylabel('Entropy')
ax2.legend(loc='upper right')
# Set the y-axis limits, starting from 1
ax2.set_ylim(1, ax2.get_ylim()[1])
# Move the line plot legend outside the plot (below the first legend)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.2), title="Entropies")
#plt.title('Compression Ratios and Entropies for Datasets that have High Entropy after Removing Repetitive Consecutive Values', loc='left', fontsize=11)



plt.title('Compression Ratios and Entropies for Datasets that has Low_entropy Datasets')
plt.tight_layout()
plt.savefig("Low.png")

plt.show()