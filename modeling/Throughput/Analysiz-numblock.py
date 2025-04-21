import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
directories = ['/home/jamalids/Documents/results1']
dataframes = []

for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                # Read CSV file using semicolon as delimiter.
                df = pd.read_csv(file_path, sep=';')
                df.fillna(0, inplace=True)
                # Ensure a "DatasetName" column exists.
                if 'DatasetName' not in df.columns and 'dataset' in df.columns:
                    df.rename(columns={'dataset': 'DatasetName'}, inplace=True)
                if 'DatasetName' not in df.columns:
                    df['DatasetName'] = os.path.basename(file_path).replace('.csv', '')
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if not dataframes:
    print("No CSV files were processed. Please check the directories.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.fillna(0, inplace=True)
if 'Index' in combined_df.columns:
    combined_df.drop(columns=['Index'], inplace=True)

print("Columns in combined_df:", combined_df.columns.tolist())

all_data_output_path = '/home/jamalids/Documents/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')

# ====================================
# 2. Median Aggregation
# ====================================
required_cols = [
    'DatasetName', 'Threads', 'RunType', 'BlockSize', 'ConfigString',
    'TotalTimeCompressed', 'TotalTimeDecompressed',
    'CompressionThroughput', 'DecompressionThroughput', 'CompressionRatio','Num-Block'
]
missing_cols = [col for col in required_cols if col not in combined_df.columns]
if missing_cols:
    print("Missing columns in combined_df:", missing_cols)
    exit()

group_columns = ['DatasetName', 'Threads', 'RunType', 'BlockSize', 'ConfigString']
median_df = combined_df.groupby(group_columns, as_index=False).median(numeric_only=True)

median_output_path = '/home/jamalids/Documents/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')
##################compression Throughput import matplotlib.pyplot as plt

# Filter the data for the desired RunType
filtered_df = median_df[median_df["RunType"] == "Chunked_Decompose_Parallel"].copy()

# Ensure correct data types
filtered_df["Threads"] = filtered_df["Threads"].astype(int)
filtered_df["Num-Block"] = filtered_df["Num-Block"].astype(int)

# Create formatted x-axis labels
filtered_df["x_label"] = filtered_df.apply(
    lambda row: f"{int(row['Num-Block'])}-{int(row['BlockSize']) // 1024}KB", axis=1
)

# Unique thread counts
unique_threads = sorted(filtered_df["Threads"].unique())

# Global y-limits for consistency
comp_min = filtered_df["CompressionThroughput"].min()
comp_max = filtered_df["CompressionThroughput"].max()
decomp_min = filtered_df["DecompressionThroughput"].min()
decomp_max = filtered_df["DecompressionThroughput"].max()

# Create subplots: 2 rows per thread (Compression + Decompression)
fig, axes = plt.subplots(nrows=2 * len(unique_threads), ncols=1,
                         figsize=(12, 5 * len(unique_threads)), sharex=True)

# Ensure axes are iterable
if len(unique_threads) == 1:
    axes = [axes]

# Plot each pair of subplots for each thread count
for idx, thread in enumerate(unique_threads):
    subset = filtered_df[filtered_df["Threads"] == thread]

    # Compression Throughput subplot
    ax_comp = axes[2 * idx]
    ax_comp.plot(subset["x_label"], subset["CompressionThroughput"], color='tab:blue', marker='o')
    ax_comp.set_ylabel("CompressionThroughput", color='tab:blue')
    ax_comp.set_ylim(comp_min, comp_max)
    ax_comp.set_title(f"Compression Throughput (Threads = {thread})")
    ax_comp.grid(True)

    # Decompression Throughput subplot
    ax_decomp = axes[2 * idx + 1]
    ax_decomp.plot(subset["x_label"], subset["DecompressionThroughput"], color='tab:green', marker='s')
    ax_decomp.set_ylabel("DecompressionThroughput", color='tab:green')
    ax_decomp.set_ylim(decomp_min, decomp_max)
    ax_decomp.set_title(f"Decompression Throughput (Threads = {thread})")
    ax_decomp.grid(True)

# Label the shared x-axis
axes[-1].set_xlabel("Number of block - BlockSize")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/comp_decomp_throughput_plot.png")


##########################################
# Determine the global min/max for both y-axes
throughput_min = filtered_df["CompressionThroughput"].min()
throughput_max = filtered_df["CompressionThroughput"].max()
decomp_min = filtered_df["DecompressionThroughput"].min()
decomp_max = filtered_df["DecompressionThroughput"].max()
ratio_min = filtered_df["CompressionRatio"].min()
ratio_max = filtered_df["CompressionRatio"].max()

# Create subplots: 2 rows per thread (Compression + Decompression)
fig, axes = plt.subplots(nrows=2 * len(unique_threads), ncols=1, figsize=(12, 5 * len(unique_threads)), sharex=True)

# Make sure axes is iterable
if len(unique_threads) == 1:
    axes = [axes]

# Plot each pair of subplots per thread count
for idx, thread in enumerate(unique_threads):
    subset = filtered_df[filtered_df["Threads"] == thread]

    # --- Compression Throughput subplot ---
    ax = axes[2 * idx]
    ax.plot(subset["x_label"], subset["CompressionThroughput"], color='tab:blue', marker='o', label='CompressionThroughput')
    ax.set_ylabel("CompressionThroughput", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_ylim(throughput_min, throughput_max)
    ax.set_title(f"Compression Throughput with {thread} Threads")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(subset["x_label"], subset["CompressionRatio"], color='tab:red', marker='s', linestyle='--', label='CompressionRatio')
    ax2.set_ylabel("CompressionRatio", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(ratio_min, ratio_max)

    # --- Decompression Throughput subplot ---
    ax_decomp = axes[2 * idx + 1]
    ax_decomp.plot(subset["x_label"], subset["DecompressionThroughput"], color='tab:green', marker='^', label='DecompressionThroughput')
    ax_decomp.set_ylabel("DecompressionThroughput", color='tab:green')
    ax_decomp.tick_params(axis='y', labelcolor='tab:green')
    ax_decomp.set_ylim(decomp_min, decomp_max)
    ax_decomp.set_title(f"Decompression Throughput with {thread} Threads")
    ax_decomp.grid(True)

    ax2_decomp = ax_decomp.twinx()
    ax2_decomp.plot(subset["x_label"], subset["CompressionRatio"], color='tab:red', marker='s', linestyle='--')
    ax2_decomp.set_ylabel("CompressionRatio", color='tab:red')
    ax2_decomp.tick_params(axis='y', labelcolor='tab:red')
    ax2_decomp.set_ylim(ratio_min, ratio_max)

# Shared x-axis label
axes[-1].set_xlabel("BlockNumber - BlockSize")
plt.tight_layout()
plt.savefig(("/home/jamalids/Documents/comp_decomp_throughput_ratio.png"))




































