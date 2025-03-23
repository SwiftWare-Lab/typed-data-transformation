import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Create a proper boxplot showing distribution of decomposed zstd compression ratios
import seaborn as sns
# Use the correct file name from the uploaded files
combine_df = pd.read_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/exhusive/combine_all.csv')


# Identify relevant column names
std_ratio_col = "standard zstd compression ratio"
decomposed_ratio_col = "decomposed zstd compression ratio"

# Find the correct dataset name column
dataset_col = None
for col in ['dataset name', 'Dataset', 'dataset']:
    if col in combine_df.columns:
        dataset_col = col
        break

if dataset_col is None:
    raise Exception("No dataset column found in combine.csv")

# Group by dataset and compute standard, min, max
box_data = []
dataset_labels = []

for dataset, group in combine_df.groupby(dataset_col):
    std_val = group[std_ratio_col].iloc[0] if pd.notnull(group[std_ratio_col].iloc[0]) else None
    min_val = group[decomposed_ratio_col].min(skipna=True)
    max_val = group[decomposed_ratio_col].max(skipna=True)

    if pd.notnull(std_val) and pd.notnull(min_val) and pd.notnull(max_val):
        box_data.append([min_val, std_val, max_val])
        dataset_labels.append(dataset)

# Create a boxplot-style visualization
fig, ax = plt.subplots(figsize=(14, 8))

for i, (min_val, std_val, max_val) in enumerate(box_data):
    ax.plot([i, i], [min_val, max_val], color='gray', linewidth=2)
    ax.scatter(i, std_val, color='green', label='Standard Zstd Ratio' if i == 0 else "", zorder=5)
    ax.scatter(i, min_val, color='red', label='Min Decomposed Ratio' if i == 0 else "", zorder=5)
    ax.scatter(i, max_val, color='blue', label='Max Decomposed Ratio' if i == 0 else "", zorder=5)

ax.set_xticks(range(len(dataset_labels)))
ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Compression Ratio')
ax.set_title('Standard Zstd vs Min/Max Decomposed Zstd Compression Ratios')
ax.legend()

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/motivefig.png')
#########################################


# Prepare data: drop rows with NaNs in the decomposed ratio
boxplot_df = combine_df.dropna(subset=[decomposed_ratio_col])

# Set up the figure
plt.figure(figsize=(14, 8))

# Create a seaborn boxplot for each dataset
sns.boxplot(data=boxplot_df, x=dataset_col, y=decomposed_ratio_col)

# Overlay standard zstd compression ratio as green 'x' markers
std_ratios = combine_df.groupby(dataset_col)[std_ratio_col].first()

# Match order of datasets as used in boxplot
ordered_datasets = boxplot_df[dataset_col].unique().tolist()
std_values = [std_ratios[ds] for ds in ordered_datasets]

plt.scatter(
    x=range(len(ordered_datasets)),
    y=std_values,
    color='green',
    marker='x',
    s=100,
    label='Standard Zstd Ratio'
)

plt.xlabel('Dataset')
plt.ylabel('Compression Ratio')
plt.title('Boxplot of Decomposed Zstd Compression Ratios with Standard Zstd Overlay')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/motivefig2.png')
