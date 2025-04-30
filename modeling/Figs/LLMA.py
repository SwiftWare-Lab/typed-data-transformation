import os

import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
directories = ["/mnt/c/Users/jamalids/Downloads/figs/results/zlib-llama"]
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

all_data_output_path = '/mnt/c/Users/jamalids/Downloads/figs/combined_all_data.csv'
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

median_output_path = '/mnt/c/Users/jamalids/Downloads/figs/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')
########################################

########################LLma////////////////////////////////

import pandas as pd
import matplotlib.pyplot as plt

# === Load data for Chunked_Decompose_Parallel ===
file_path = '/mnt/c/Users/jamalids/Downloads/figs/combined_median_rows.csv'
from scipy.stats import gmean


# Drop NA and fill zeros
df.fillna(0, inplace=True)

# Step 1: Group by required keys
grouped = df.groupby([ 'Threads', 'BlockSize', 'ConfigString', 'RunType'], as_index=False)

# Step 2: Compute aggregation
aggregated = grouped.agg({
    'Compressedsize': 'sum',
    'TotalValues': 'sum',
    'TotalTimeCompressed': 'median',
    'TotalTimeDecompressed': 'median',
    'CompressionThroughput': lambda x: gmean(x[x > 0]) if any(x > 0) else 0,
    'DecompressionThroughput': lambda x: gmean(x[x > 0]) if any(x > 0) else 0,
    'Num-Block': 'sum',
})
aggregated.reset_index(inplace=True)
#aggregated.rename(columns={"index": "Index"}, inplace=True)
aggregated['DatasetName'] = 'LLama'

# Step 3: Compute CompressionRatio (TotalValues / Compressedsize)
aggregated['CompressionRatio'] = aggregated['TotalValues'] / aggregated['Compressedsize']

# Step 4: Reorder and rename
aggregated.reset_index(inplace=True)
aggregated.rename(columns={"index": "Index"}, inplace=True)

# Final column order
final_cols = [
    'Index', 'DatasetName', 'Threads', 'BlockSize', 'ConfigString', 'RunType',
    'CompressionRatio', 'TotalTimeCompressed', 'TotalTimeDecompressed',
    'CompressionThroughput', 'DecompressionThroughput',
    'TotalValues'
]
aggregated = aggregated[final_cols]

# Step 5: Save to CSV with semicolon separator
aggregated.to_csv("/mnt/c/Users/jamalids/Downloads/figs/llama.csv", sep=';', index=False)

print("✅ Final CSV saved to final_median_throughput_ratio.csv")
