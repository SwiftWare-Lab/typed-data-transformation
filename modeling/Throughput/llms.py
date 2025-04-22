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
    'CompressionThroughput', 'DecompressionThroughput', 'CompressionRatio'
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
########################################

#######################3
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = '/home/jamalids/Documents/combined_median_rows.csv'
df = pd.read_csv(file_path)

# Filter for Chunked_Decompose_Parallel and Threads == 16
chunked_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]


# Compute geometric mean by BlockSize
gmean_chunked = chunked_df.groupby('BlockSize').agg({
    'CompressionRatio': gmean,
    'CompressionThroughput': gmean,
    'DecompressionThroughput': gmean
}).reset_index()

# Convert BlockSize to string for clear x-axis labels
gmean_chunked['BlockSize'] = gmean_chunked['BlockSize'].astype(str)

# Compute Full Baseline: apply gmean to each column directly (instead of deprecated .agg)
full_df = df[(df['RunType'] == 'Full') & (df['Threads'] == 16)]
gmean_full = full_df[['CompressionRatio', 'CompressionThroughput', 'DecompressionThroughput']].apply(gmean)

# Extract scalar values using .item()
baseline_cr = gmean_full['CompressionRatio'].item()
baseline_ct = gmean_full['CompressionThroughput'].item()
baseline_dct = gmean_full['DecompressionThroughput'].item()

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

# Compression Ratio plot
axes[0].plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionRatio'], marker='o', color='blue', label='GMean CompressionRatio')
axes[0].axhline(baseline_cr, color='red', linestyle='--', label='Full Baseline CompressionRatio')
axes[0].set_ylabel('GMean CompressionRatio')
axes[0].set_title('GMean Compression Ratio vs BlockSize')
axes[0].legend()
axes[0].grid(True)

# Throughput plot
axes[1].plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionThroughput'], marker='o', color='green', label='GMean TDT CompressionThroughput')
axes[1].plot(gmean_chunked['BlockSize'], gmean_chunked['DecompressionThroughput'], marker='^', color='purple', label='GMean TDT DecompressionThroughput')
axes[1].axhline(baseline_ct, color='red', linestyle='--', label='GMean Standard CompressionThroughput')
axes[1].axhline(baseline_dct, color='black', linestyle='--', label='GMean Standard DecompressionThroughput')
axes[1].set_ylabel('GMean Throughput')
axes[1].set_xlabel('BlockSize')
axes[1].set_title('GMean Compression & Decompression Throughput vs BlockSize using Zstd')
axes[1].legend()
axes[1].grid(True)

# Rotate x-axis labels
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Save and show
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/Throughput.png')

########################LLma////////////////////////////////

import pandas as pd
import matplotlib.pyplot as plt

# === Load data for Chunked_Decompose_Parallel ===
file_path = '/home/jamalids/Documents/combined_median_rows.csv'
df = pd.read_csv(file_path)

# Ensure necessary columns exist
if 'Compressedsize' not in df.columns or 'TotalValues' not in df.columns:
    raise ValueError("CSV must include 'Compressedsize' and 'TotalValues' columns.")

# Filter Chunked_Decompose_Parallel
chunked_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]
chunked_df['BlockSize'] = chunked_df['BlockSize'].astype(str)

# Group by BlockSize
chunked_cr2_df = chunked_df.groupby('BlockSize').agg({
    'Compressedsize': 'sum',
    'TotalValues': 'sum'
}).reset_index()
chunked_cr2_df['CompressionRatio2'] =  chunked_cr2_df['TotalValues']/ chunked_cr2_df['Compressedsize']

# === Compute full baseline CompressionRatio2 ===
full_df = df[(df['RunType'] == 'Full') & (df['Threads'] == 16)]
total_compressed = full_df['Compressedsize'].sum()
total_original = full_df['TotalValues'].sum()
baseline_cr2 =  total_original/total_compressed
print(f"Full CompressionRatio2 (baseline) = {baseline_cr2:.4f}")

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(chunked_cr2_df['BlockSize'], chunked_cr2_df['CompressionRatio2'], marker='o', label='Chunked_Decompose_Parallel')
plt.axhline(y=baseline_cr2, color='red', linestyle='--', label=f'Full Baseline ({baseline_cr2:.4f})')

plt.xlabel('BlockSize')
plt.ylabel('CompressionRatio2 (Total Original/Compressed)')
plt.title('CompressionRatio2 vs BlockSize\nAll Datasets Combined')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save
plt.savefig('/home/jamalids/Documents/CompressionRatio2_AllDatasets_with_FullBaseline.png')

