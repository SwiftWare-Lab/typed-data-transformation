import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import gmean

# ======================
# VLDB-Style Global Plot Settings
# ======================
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ======================
# 1. Combine CSVs
# ======================
directories = ['/mnt/c/Users/jamalids/Downloads/figs/results/results-zstd/all']
dataframes = []

for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                df = pd.read_csv(file_path, sep=';')
                df.fillna(0, inplace=True)
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

all_data_output_path = '/mnt/c/Users/jamalids/Downloads/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')

# ======================
# 2. Median Aggregation
# ======================
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

median_output_path = '/mnt/c/Users/jamalids/Downloads/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')

# ======================
# 3. Load Median Data
# ======================
file_path = '/mnt/c/Users/jamalids/Downloads/combined_median_rows.csv'
df = pd.read_csv(file_path)

# ======================
# 4. Plots: GMean by DatasetName
# ======================
filtered_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]

gmean_df = filtered_df.groupby(['DatasetName', 'BlockSize']).agg({
    'CompressionRatio': gmean,
    'CompressionThroughput': gmean,
    'DecompressionThroughput': gmean
}).reset_index()

gmean_df_sorted = gmean_df.sort_values(by='DatasetName')
gmean_df_sorted['BlockSize'] = gmean_df_sorted['BlockSize'].astype(str)

# --- CompressionRatio Plot ---
plt.figure(figsize=(6.2, 2.5))
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='CompressionRatio', marker='o', color='skyblue')
plt.ylabel('GMean Compression Ratio')
plt.xlabel('Dataset Name')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig('/mnt/c/Users/jamalids/Downloads/Cot-CompressionRatio-vldb.pdf', bbox_inches='tight')
plt.close()

# --- Compression and Decompression Throughput Plot ---
plt.figure(figsize=(6.2, 2.5))
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='CompressionThroughput', marker='o', color='green', label='Compression Throughput')
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='DecompressionThroughput', marker='^', color='purple', label='Decompression Throughput')
plt.ylabel('GMean Throughput')
plt.xlabel('Dataset Name')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/jamalids/Downloads/Cot-Throughput-vldb.pdf', bbox_inches='tight')
plt.close()

# ======================
# 5. Plots: GMean by BlockSize
# ======================
chunked_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]

gmean_chunked = chunked_df.groupby('BlockSize').agg({
    'CompressionRatio': gmean,
    'CompressionThroughput': gmean,
    'DecompressionThroughput': gmean
}).reset_index()

gmean_chunked['BlockSize'] = gmean_chunked['BlockSize'].astype(str)

# --- CompressionRatio vs BlockSize ---
plt.figure(figsize=(6.2, 2.5))
plt.plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionRatio'], marker='o', color='blue', label='Compression Ratio')
plt.ylabel('GMean CR')
plt.xlabel('Block Size')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/jamalids/Downloads/CompressionRatio-BlockSize.pdf', bbox_inches='tight')
plt.close()

# --- Throughput vs BlockSize ---
plt.figure(figsize=(6.2, 2.5))
plt.plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionThroughput'], marker='o', color='green', label='Compression Throughput')
plt.plot(gmean_chunked['BlockSize'], gmean_chunked['DecompressionThroughput'], marker='^', color='purple', label='Decompression Throughput')
plt.ylabel('GMean Throughput')
plt.xlabel('Block Size')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/jamalids/Downloads/Throughput-BlockSize.pdf', bbox_inches='tight')
plt.close()

# ======================
# Done ✅
# ======================
print("All plots saved separately in VLDB style (PDF).")
