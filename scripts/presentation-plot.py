import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
#file_path = '/home/jamalids/Documents/compression-part3/results_V2_present/result2M/Decom+zstd+gzip.csv'  # Change this to your actual path
file_path="/home/jamalids/Documents/compression-part3/big-data-compression/modeling/hst/Decom+zstd+gzip.csv"
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
ax1.bar([i + 2*bar_width for i in index[::2]], df['zstd_22_com_ratio_b1'], bar_width, label='zstd_22_com_ratio_b1 ')
ax1.bar([i + 3*bar_width for i in index[::2]], df['zstd_22_com_ratio_b2'], bar_width, label='zstd_22_com_ratio_b2')
ax1.bar([i + 4*bar_width for i in index[::2]], df['zstd_com_ratio_b1'], bar_width, label='zstd_com_ratio_b1')
ax1.bar([i + 5*bar_width for i in index[::2]], df['zstd_com_ratio_b2'], bar_width, label='zstd_com_ratio_b2')
#ax1.bar([i + 6*bar_width for i in index[::2]], df['comp_ratio_gzip'], bar_width, label='comp_ratio_gzip')
#ax1.bar([i + 7*bar_width for i in index[::2]], df['max_Decom+gzip_com_ratio'], bar_width, label='Decom+gzip')
#ax1.set_yscale('log')
# Labels for bar chart
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Compression Ratios ')
ax1.set_xticks([i + 2*bar_width for i in index[::2]])
ax1.set_xticklabels(df['dataset_name'], rotation=90)
# Move the bar chart legend outside the plot (to the right)
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Compression Methods")



# Line plot for entropy_all with space
ax2 = ax1.twinx()
ax2.plot(index[::2], df['entropy_float'], color='black', marker='o', label='Entropy of dataset')
ax2.plot(index[::2], df['entropy_remainig'], color='red', marker='o', label='Entropy Remaining')

# Leading Entropy Pair
ax2.plot(index[::2], df['b1_leading_entropy'], color='darkblue', marker='o', label='b1_leading_entropy')
ax2.plot(index[::2], df['b2_leading_entropy'], color='lightblue', marker='o', label='b2_leading_entropy')

# Content Entropy Pair
ax2.plot(index[::2], df['b1_content_entropy'], color='darkgreen', marker='o', label='b1_content_entropy')
ax2.plot(index[::2], df['b2_content_entropy'], color='lightgreen', marker='o', label='b2_content_entropy')

# Tailing Entropy Pair
ax2.plot(index[::2], df['b1_tailing_entropy'], color='purple', marker='o', label='b1_tailing_entropy')
ax2.plot(index[::2], df['b2_tailing_entropy'], color='mediumorchid', marker='o', label='b2_tailing_entropy')

# Adding a legend for clarity
ax2.legend(loc='upper left')
ax2.set_ylabel('Entropy')
ax2.legend(loc='upper right')
# Set the y-axis limits, starting from 1
ax2.set_ylim(1, ax2.get_ylim()[1])
# Move the line plot legend outside the plot (below the first legend)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.2), title="Entropies")

plt.title('Compression Ratios and Entropies for different component sizes in some high-entropy datasets ')
plt.tight_layout()
plt.savefig("zstd_entropy_archive.png")

plt.show()

# Prepare a new DataFrame to store the calculated values displayed in the plots
plot_data = pd.DataFrame({
    'dataset_name': df['dataset_name'],
    'comp_ratio_zstd_default': df['comp_ratio_zstd_default'],
    'comp_ratio_l22': df['comp_ratio_l22'],
    'zstd_22_com_ratio_b1': df['zstd_22_com_ratio_b1'],
    'zstd_22_com_ratio_b2': df['zstd_22_com_ratio_b2'],
    'zstd_com_ratio_b1': df['zstd_com_ratio_b1'],
    'zstd_com_ratio_b2': df['zstd_com_ratio_b2'],
    'entropy_float': df['entropy_float'],
    'entropy_remainig': df['entropy_remainig'],
    'b1_leading_entropy': df['b1_leading_entropy'],
    'b2_leading_entropy': df['b2_leading_entropy'],
    'b1_content_entropy': df['b1_content_entropy'],
    'b2_content_entropy': df['b2_content_entropy'],
    'b1_tailing_entropy': df['b1_tailing_entropy'],
    'b2_tailing_entropy': df['b2_tailing_entropy'],
    'gmean_com_ratio_zstd': gmean_com_ratio_zstd,
    'gmean_decom': gmean_decom
})
plot_data.to_csv("plots_data.csv")