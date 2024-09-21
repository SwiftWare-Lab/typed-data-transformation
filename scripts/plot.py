import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
file_path = '/home/jamalids/Documents/compression-part3/big-data-compression/modeling/Decom+zstd.csv'  # Change this to your actual path
df = pd.read_csv(file_path)

# Create a dictionary that maps old column names to new, more meaningful names
rename_dict = {
    't_com_ratio_b1': 'com_ratio_b1_dict',
    't_com_ratio_b2': 'com_ratio_b2_dict',
    't_com_ratio_b3': 'com_ratio_b3_dict',
    'R_com_ratio_b1': 'RLE_com_ratio_b1',
    'R_com_ratio_b2': 'RLE_com_ratio_b2',
    'R_com_ratio_b3': 'RLE_com_ratio_b3',
    'comp_ratio_l22': 'Zstd_comp_ratio_22',
    't-max_com_ratio': 'max_comp_ratio_dict'
}

# Rename the columns in the DataFrame
df = df.rename(columns=rename_dict)

# First bar chart columns
bar_columns_1 = [
    'com_ratio_b1', 'com_ratio_b1_dict', 'com_ratio_b2', 'com_ratio_b2_dict', 'com_ratio_b3', 'com_ratio_b3_dict',
    'RLE_com_ratio_b1', 'RLE_com_ratio_b2', 'RLE_com_ratio_b3', 'zstd_22_com_ratio_b1', 'zstd_22_com_ratio_b2', 'zstd_22_com_ratio_b3',
    'zstd_com_ratio_b1', 'zstd_com_ratio_b2', 'zstd_com_ratio_b3', 'comp_ratio_zstd_default', 'Zstd_comp_ratio_22', 'Non_uniform_1x4'
]

# Create a new column that holds the maximum value of R_com_ratio_b1, R_com_ratio_b2, and R_com_ratio_b3 for each row
df['max_Decom_RLE_com_ratio'] = df[['RLE_com_ratio_b1','RLE_com_ratio_b2', 'RLE_com_ratio_b3']].max(axis=1)

# Second bar chart columns
bar_columns_2 = [
    'max_com_ratio', 'max_comp_ratio_dict', 'max_Decom+zstd_22_com_ratio', 'max_Decom+zstd_com_ratio',
    'comp_ratio_zstd_default', 'Zstd_comp_ratio_22', 'Non_uniform_1x4', 'max_Decom_RLE_com_ratio' # Use the newly created column
]

# Entropy columns
entropy_columns = [
    'entropy_all', 'b1_tailing_entropy', 'b2_tailing_entropy', 'b3_tailing_entropy',
    'b1_content_entropy', 'b2_content_entropy', 'b3_content_entropy',
    'b1_leading_entropy', 'b2_leading_entropy', 'b3_leading_entropy'
]

# Create figure and axes for three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))  # Three plots stacked vertically

# Add space between datasets by modifying the index
gap_size = 1  # Adjust this value to increase or decrease the space between datasets
index = np.arange(0, len(df) * (1 + gap_size), 1 + gap_size)

# Plot 1: Compression ratios (First group)
bar_width = 0.2

# Plot the bar chart for each compression ratio column (First group)
for i, col in enumerate(bar_columns_1):
    ax1.bar(index + i*bar_width, df[col], bar_width, label=col)

# Customize the x-axis and labels for the first bar chart
ax1.set_xlabel('Datasets')
ax1.set_ylabel('Compression Ratios ')
ax1.set_title('Compression Ratios ')
ax1.set_xticks(index + bar_width * (len(bar_columns_1) // 2))
ax1.set_xticklabels(df['dataset_name'], rotation=90)  # Assuming 'dataset_name' is the identifier

# Add legend for the first bar chart
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Plot 2: Compression ratios (Second group)
# Plot the bar chart for each compression ratio column (Second group)
for i, col in enumerate(bar_columns_2):
    ax2.bar(index + i*bar_width, df[col], bar_width, label=col)

# Customize the x-axis and labels for the second bar chart
ax2.set_xlabel(' Datasets')
ax2.set_ylabel('Compression Ratios (aggregation)')
ax2.set_title('Compression Ratios (aggregation)')
ax2.set_xticks(index + bar_width * (len(bar_columns_2) // 2))
ax2.set_xticklabels(df['dataset_name'], rotation=90)

# Add legend for the second bar chart
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Plot 3: Entropy values as bar charts
# Plot the bar chart for each entropy column
for i, col in enumerate(entropy_columns):
    ax3.bar(index + i*bar_width, df[col], bar_width, label=col)

# Customize the x-axis and labels for the entropy chart
ax3.set_xlabel(' Datasets')
ax3.set_ylabel('Entropy Values')
ax3.set_title('Entropy Values for Different Components')
ax3.set_xticks(index + bar_width * (len(entropy_columns) // 2))
ax3.set_xticklabels(df['dataset_name'], rotation=90)

# Add legend for the entropy plot
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Tight layout for better spacing
plt.tight_layout()
plt.savefig('decom+zstd_with_spacing.png')
# Show the plots
plt.show()
