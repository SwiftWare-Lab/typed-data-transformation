import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
file_path = "/home/jamalids/Documents/compression-part3/big-data-compression/modeling/hst/Decom+zstd+gzip.csv"
df = pd.read_csv(file_path)

# Rename the column from 'comp_ratio_l22' to 'com_ratio_22'
df = df.rename(columns={'comp_ratio_l22': 'zstd_com_ratio_22'})
df = df.rename(columns={'zstd_22_com_ratio_b1': 'Decompose+zstd_22_com_ratio'})
df = df.rename(columns={'zstd_com_ratio_b1': 'Decompose+zstd_com_ratio'})

# Convert specific columns to numeric, forcing errors to NaN (in case there are non-numeric values)
columns_to_convert = ['comp_ratio_zstd_default', 'zstd_com_ratio_22', 'Decompose+zstd_22_com_ratio', 'Decompose+zstd_com_ratio']
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Adjusting the plot to make it narrower and improve readability
fig, ax1 = plt.subplots(figsize=(12, 6))  # Changed to a narrower figure size

# Bar chart data
bar_width = 0.2  # Adjusted bar width for better spacing
index = np.arange(len(columns_to_convert))  # Creating a numerical index for the compression methods

# Plotting the bars with space between datasets
for i, dataset in enumerate(df['dataset_name']):
    ax1.bar(index + i * bar_width, df.loc[i, columns_to_convert], bar_width, label=dataset)

# Labels for the bar chart
ax1.set_xlabel('Compression Methods')
ax1.set_ylabel('Compression Ratios')
ax1.set_title('Compression Ratios Comparison Across Methods')
ax1.set_xticks(index + (len(df) - 1) * bar_width / 2)  # Center the labels under the grouped bars
ax1.set_xticklabels(columns_to_convert, rotation=45, ha='right')  # Rotate labels for better readability

# Move the bar chart legend outside the plot (to the right)
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Datasets")

# Adjust layout to prevent clipping of tick labels and legend
plt.tight_layout()

# Save the plot as an image file
plt.savefig('compression_ratios_comparison_by_method.png', bbox_inches='tight')

# Display the plot
plt.show()
