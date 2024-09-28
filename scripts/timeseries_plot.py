import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/home/jamalids/Documents/compression-part3/big-data-compression/modeling/citytemp_f32.tsv"
data = pd.read_csv(file_path)

# Calculate the averages for specific columns
grouped_data = data.groupby('dataset_name').agg({
    'entropy_all': 'mean',
    'entropy_float_all': 'mean',
    't-max_com_ratio': 'mean',
    'max_Decom+zstd_22_com_ratio': 'mean',
    'max_Decom+zstd_com_ratio': 'mean',
    'max_Decom+gzip_com_ratio': 'mean',
    'comp_ratio_zstd_default': 'mean',
    'comp_ratio_l22': 'mean',
    'comp_ratio_gzip': 'mean',
    'Non_uniform_1x4': 'mean'
}).reset_index()

# Plot the data
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot for compression ratios with custom legend names
bar_columns = ['t-max_com_ratio', 'max_Decom+zstd_22_com_ratio', 'max_Decom+zstd_com_ratio', 'max_Decom+gzip_com_ratio',
               'comp_ratio_zstd_default', 'comp_ratio_l22', 'comp_ratio_gzip']
custom_labels = ['Mean Our method Ratio with dict', 'Mean Decom+Zstd22 Ratio', 'Mean Decom+Zstd Ratio',
                 'Mean Decom+Gzip Ratio', 'Mean Zstd Default Ratio', 'Mean Zstd L22 Ratio', 'Mean Gzip Ratio']

# Create x positions for the bars with additional space between datasets
bar_width = 0.1  # Adjust width of each bar
dataset_count = len(grouped_data)
space_between_datasets = 0.3  # Adjust this to control the space between datasets
positions = [i * (len(bar_columns) * bar_width + space_between_datasets) for i in range(dataset_count)]

# Plot the bars for each compression metric
for i, (column, label) in enumerate(zip(bar_columns, custom_labels)):
    ax1.bar([pos + i * bar_width for pos in positions], grouped_data[column], width=bar_width, label=label)

# Set x-axis labels and ticks
ax1.set_xlabel('Dataset Name')
ax1.set_ylabel('Compression Ratios')
ax1.set_xticks([pos + (len(bar_columns) / 2) * bar_width for pos in positions])
ax1.set_xticklabels(grouped_data['dataset_name'], rotation=90)
ax1.legend(loc='upper left')

# Line plot for average entropies with custom legend names
ax2 = ax1.twinx()
ax2.plot([pos + (len(bar_columns) / 2) * bar_width for pos in positions], grouped_data['entropy_all'], color='r', marker='o', label='Average Entropy of time series')
ax2.plot([pos + (len(bar_columns) / 2) * bar_width for pos in positions], grouped_data['entropy_float_all'], color='g', marker='s', label='Entropy of all data')
ax2.set_ylabel('Average Entropies')
ax2.legend(loc='upper right')

# Show the plot
plt.title('Compression Ratios and Entropies by Dataset')
plt.tight_layout()
plt.savefig('timeseries_plot40.png')
plt.show()
###################################################################
# Re-define necessary variables
bar_columns = ['t-max_com_ratio', 'max_Decom+zstd_22_com_ratio', 'max_Decom+zstd_com_ratio', 'max_Decom+gzip_com_ratio',
               'comp_ratio_zstd_default', 'comp_ratio_l22', 'comp_ratio_gzip', 'Non_uniform_1x4']

bar_width = 0.1  # Adjust width of each bar
block_idx_count = len(data['block_idx'].unique())
positions = list(range(len(data)))

# Create the plot with log scale for the y-axis
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot the bars for each compression metric with log scale on y-axis
for i, column in enumerate(bar_columns):
    ax1.bar([p + i * bar_width for p in positions], data[column], width=bar_width, label=column)

# Set log scale for y-axis to handle small size values
ax1.set_yscale('log')

# Set x-axis labels and ticks
ax1.set_xlabel('Block_idx and Dataset Name')
ax1.set_ylabel('Compression Ratios and Non_uniform_1x4 (Log Scale)')
ax1.set_xticks([p + (len(bar_columns) / 2) * bar_width for p in positions])
ax1.set_xticklabels([f"{row['block_idx']}_{row['dataset_name']}" for _, row in data.iterrows()], rotation=90)
ax1.legend(loc='upper left')

# Line plot for entropy values on the secondary axis
ax2 = ax1.twinx()
ax2.plot([p + (len(bar_columns) / 2) * bar_width for p in positions], data['entropy_all'], color='r', marker='o', label='Entropy each block')
ax2.plot([p + (len(bar_columns) / 2) * bar_width for p in positions], data['entropy_float_all'], color='g', marker='s', label='Entropy_All')
ax2.set_ylabel('Entropy Values')
ax2.legend(loc='upper right')

# Show the plot
plt.title('Compression Ratios, Non-uniform, and Entropy Values by Block Index and Dataset (Log Scale)')
plt.tight_layout()
plt.savefig('timeseries_plot_with_log_entropy_block_fixed.png')
plt.show()
