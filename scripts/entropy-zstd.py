import matplotlib.pyplot as plt
import pandas as pd
import os

# File path
file_path = '/home/jamalids/Documents/WE/64-High-Entropy/combined_64H_data.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Ensure the dataset contains a 'dataset' column for grouping
if 'dataset' not in df.columns:
    raise ValueError("The dataframe must contain a 'dataset' column to group by datasets.")

# Get unique datasets and determine the number of subplots
unique_datasets = df['dataset'].unique()
num_datasets = len(unique_datasets)

# Set the number of rows and columns for subplots
nrows, ncols = 3,2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True)
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Create handles for legends
handles_compression = []
labels_compression = []
handles_entropy = []
labels_entropy = []

# Iterate over datasets and plot each on a separate subplot
for i, (dataset, ax) in enumerate(zip(unique_datasets, axes)):
    data = df[df['dataset'] == dataset]
    df_full = data[data["RunType"] == "Full"]
    df_parallel = data[data["RunType"] == "Parallel"]

    # Plot bar for Full and Parallel Compression Ratios
    bar_width = 0.3
    x = range(len(df_full["ComponentSizes"]))
    bars_full = ax.bar(x, df_full["CompressionRatio"], width=bar_width, label="Compression Ratio (Zstd)", color='cyan')
    bars_parallel = ax.bar(
        [p + bar_width for p in x], df_parallel["CompressionRatio"], width=bar_width, label="Compression Ratio (Decompose+Zstd)", color='red'
    )
    ax.set_ylabel("Compression Ratio", fontsize=10)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(df_full["ComponentSizes"], rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Create a secondary y-axis for entropy lines
    ax2 = ax.twinx()
    line_full = ax2.plot(
        x, df_full["entropy_full_byte"], color='green', marker='o', linestyle='--', label="Normal Entropy"
    )
    line_decomposed = ax2.plot(
        x, df_parallel["entropy_decompose_byte"], color='black', marker='x', linestyle='--', label="Weighted Entropy"
    )
    ax2.set_ylabel("Entropy", fontsize=10)
    ax2.tick_params(axis='y', labelsize=9)

    # Add title for each subplot
    ax.set_title(f"Dataset: {dataset}", fontsize=12)

    # Collect handles and labels for legend
    if i == 0:  # Collect legend from the first subplot only
        handles_compression, labels_compression = ax.get_legend_handles_labels()
        handles_entropy, labels_entropy = ax2.get_legend_handles_labels()


plt.savefig("/home/jamalids/Documents/WE/64-High-Entropy/entropy-zstd.png")
plt.show()
