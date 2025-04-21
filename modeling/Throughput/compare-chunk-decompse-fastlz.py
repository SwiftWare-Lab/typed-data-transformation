import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
directories = ['/home/jamalids/Documents/results-fastlz']
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
##################

selected_pairs = []

# Group by DatasetName, Threads, and BlockSize
for (dataset, threads, blockSize), group in median_df.groupby(['DatasetName', 'Threads', 'BlockSize']):

    # --- 1) Select the best Decompose_Chunk_Parallel row, if present ---
    chunked_rows = group[group['RunType'] == 'Decompose_Block_Parallel']
    if not chunked_rows.empty:
        sorted_chunked = chunked_rows.sort_values(
            by=['CompressionRatio', 'CompressionThroughput'],
            ascending=False
        )
        selected_pairs.append(sorted_chunked.iloc[0])

    # --- 2) Select the best Decompose_Chunk_Parallel row, if present ---
    decompose_chunk_rows = group[group['RunType'] == 'Decompose_Chunk_Parallel']
    if not decompose_chunk_rows.empty:
        sorted_decompose_chunk = decompose_chunk_rows.sort_values(
            by=['CompressionRatio', 'CompressionThroughput'],
            ascending=False
        )
        selected_pairs.append(sorted_decompose_chunk.iloc[0])

    # --- 3) Select the Full row, if present ---
    full_rows = group[group['RunType'] == 'Full']
    if not full_rows.empty:
        selected_pairs.append(full_rows.iloc[0])

# Create DataFrame
selected_df = pd.DataFrame(selected_pairs)

# Round compression ratio
selected_df['CompressionRatio'] = selected_df['CompressionRatio'].round(3)

# Save to CSV
output_path = '/home/jamalids/Documents/selected_pairs.csv'
selected_df.to_csv(output_path, index=False)
print(f"✅ Done. CSV with selected pairs saved to: {output_path}")
##########################################################################
final_A = []
chunked_pairs = []
decomposed_pairs = []
full_rows_added = set()

# Filter only desired RunTypes
chunked_df = selected_df[selected_df['RunType'].isin(['Decompose_Block_Parallel', 'Decompose_Chunk_Parallel'])]
L2_value = 24 * 1024 * 1024
chunked_df = chunked_df[chunked_df['BlockSize'] == L2_value].copy()

if not chunked_df.empty:
    for run_type in ['Decompose_Chunk_Parallel', 'Decompose_Block_Parallel']:
        best_df = chunked_df[chunked_df['RunType'] == run_type]
        best_rows = best_df.loc[best_df.groupby('DatasetName')['CompressionThroughput'].idxmax()]

        for idx, row in best_rows.iterrows():
            dataset = row['DatasetName']
            threads = row['Threads']

            # Add best chunked/decomposed row
            final_A.append(row)

            # Add Full row only once per dataset
            if dataset not in full_rows_added:
                full_row = selected_df[
                    (selected_df['DatasetName'] == dataset) &
                    (selected_df['Threads'] == threads) &
                    (selected_df['RunType'] == 'Full')
                ]
                if not full_row.empty:
                    full_row = full_row.iloc[0]
                    final_A.append(full_row)
                    full_rows_added.add(dataset)

                # Also collect for per-runType comparisons
                if run_type == 'Decompose_Chunk_Parallel':
                    chunked_pairs.extend([row, full_row])
                else:
                    decomposed_pairs.extend([row, full_row])

# Sort final output by DatasetName
final_df_A = pd.DataFrame(final_A)
final_df_A = final_df_A.sort_values(by='DatasetName')

# Save files
final_df_A.to_csv('/home/jamalids/Documents/max_compression_throughput_pairs.csv', index=False)
pd.DataFrame(chunked_pairs).to_csv('/home/jamalids/Documents/max_Chunked_Decompose_Parallel.csv', index=False)
pd.DataFrame(decomposed_pairs).to_csv('/home/jamalids/Documents/max_Decompose_Chunk_Parallel.csv', index=False)

print("✅ Clean and sorted CSVs saved:")
print("→ /home/jamalids/Documents/max_compression_throughput_pairs.csv")
print("→ /home/jamalids/Documents/max_Chunked_Decompose_Parallel.csv")
print("→ /home/jamalids/Documents/max_Decompose_Chunk_Parallel.csv")
###################################plot###############################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV
file_path = "/home/jamalids/Documents/max_compression_throughput_pairs.csv"
df = pd.read_csv(file_path)

# Set style for the plots
sns.set(style="whitegrid")

# Replace RunType values as requested
# df['RunType'] = df['RunType'].replace({
#     'Full': 'standard',
#     'Decompose_Chunk_Parallel': 'first decompose then chunk',
#     'Decompose_Block_Parallel':'first chunk then decompose',
# })
# Replace RunType values as requested
df['RunType'] = df['RunType'].replace({
    'Full': 'standard',
    'Decompose_Block_Parallel': 'Chunk-decompose ',
    'Decompose_Chunk_Parallel': 'first decompose then chunk'
})
# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)

# Sort by DatasetName for consistent x-axis
df = df.sort_values(by="DatasetName")

# First subplot: CompressionThroughput
sns.barplot(
    data=df,
    x='DatasetName',
    y='CompressionThroughput',
    hue='RunType',
    ax=ax1
)
ax1.set_title("Compression Throughput by Dataset and RunType")
ax1.set_ylabel("Compression Throughput ")
ax1.set_xlabel("")
ax1.tick_params(axis='x', rotation=90)

# Second subplot: DecompressionThroughput
sns.barplot(
    data=df,
    x='DatasetName',
    y='DecompressionThroughput',
    hue='RunType',
    ax=ax2
)
ax2.set_title("Decompression Throughput by Dataset and RunType")
ax2.set_ylabel("Decompression Throughput (MB/s)")
ax2.set_xlabel("Dataset Name")
ax2.tick_params(axis='x', rotation=90)

# Adjust layout and show
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/throughput.png")
