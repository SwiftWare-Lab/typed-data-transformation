import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/jamalids/Documents/compression-part3/big-data-compression/modeling/results/combined_data.csv'
df= pd.read_csv(file_path)
# Define the columns to plot
columns_to_plot = ['max_com_ratio', 't-max_com_ratio', 'Non_uniform_1x4', 'comp_ratio_l22', 'comp_ratio_zstd_default']

# Grouping the data by 'dataset_name' and selecting the relevant columns
grouped_data_full = df.groupby('dataset')[columns_to_plot].mean()

# Create the plot with all datasets
fig, ax = plt.subplots(figsize=(14, 8))

# Plot a bar chart for each column
grouped_data_full.plot(kind='bar', ax=ax)

# Set plot title and labels
ax.set_title('Comparison of Compression Ratios for All Datasets')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Compression Ratio')
ax.legend(title='Metrics')
# Add descriptions
description = (
    "t-max_com_ratio: Decomposition compression ratio with dictionary.\n"
    "max_com_ratio: Decomposition compression ratio without dictionary.\n"
    "Non_uniform_1x4: Compression ratio with a non-uniform method.\n"
    "comp_ratio_l22: Compression ratio with level-22 setting.\n"
    "comp_ratio_zstd_default: Compression ratio using Zstd with default settings.\n"
    "S: Indicates similarity leading and trailing.\n"
    "0: Indicates zero leading and trailing.\n"
    "b1: Boundary leading .\n"
    "b2: Boundary trailing .\n"
)

# Add the description as a text box
plt.gcf().text(1.02, 0.5, description, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Show the plot
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('tunnig.png', bbox_inches='tight')
plt.show()