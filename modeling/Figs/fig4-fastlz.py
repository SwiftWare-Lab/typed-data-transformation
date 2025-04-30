import os

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load your processed data
directories = ['/mnt/c/Users/jamalids/Downloads/figs/results/fastlz']
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

all_data_output_path = '/mnt/c/Users/jamalids/Downloads/figs/results/combined_all_data.csv'
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

median_output_path = '/mnt/c/Users/jamalids/Downloads/figs/results/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')
##################

selected_pairs = []

for (dataset, threads, blockSize), group in median_df.groupby(['DatasetName', 'Threads', 'BlockSize']):
    # --- 1) Try to pick Decompose_Chunk_Parallel (if present) ---
    decompose_rows = group[group['RunType'] == 'Decompose_Chunk_Parallel']
    if not decompose_rows.empty:
        sorted_decompose = decompose_rows.sort_values(
            by=['CompressionRatio', 'CompressionThroughput'],
            ascending=False
        )
        chosen_decompose = sorted_decompose.iloc[0]
        selected_pairs.append(chosen_decompose)

    # --- 2) Also pick the Full row (if present) ---
    full_rows = group[group['RunType'] == 'Full']
    if not full_rows.empty:
        # For "Full" you might or might not want to sort.
        # Here we just pick the first (or best) row:
        corresponding_full = full_rows.iloc[0]
        selected_pairs.append(corresponding_full)

selected_df = pd.DataFrame(selected_pairs)
#To round the CompressionRatio column to three decimal places
selected_df['CompressionRatio'] = selected_df['CompressionRatio'].round(3)
selected_df.to_csv('/mnt/c/Users/jamalids/Downloads/figs/results/selected_pairs.csv', index=False)
print("Done. CSV with selected pairs saved.")
final_A = []
chunked_df = selected_df[selected_df['RunType'] == 'Decompose_Chunk_Parallel']
L2_value =1* 1024 *1024,
chunked_df = chunked_df [chunked_df ['BlockSize'] == L2_value].copy()

if not chunked_df.empty:
    best_chunked = chunked_df.loc[chunked_df.groupby('DatasetName')['CompressionThroughput'].idxmax()]
    for idx, row in best_chunked.iterrows():
        dataset = row['DatasetName']
        threads = row['Threads']
        final_A.append(row)
        full_row = selected_df[(selected_df['DatasetName'] == dataset) &
                               (selected_df['Threads'] == threads) &
                               (selected_df['RunType'] == 'Full')]
        if not full_row.empty:
            final_A.append(full_row.iloc[0])
final_df_A = pd.DataFrame(final_A)
final_output_path_A = '/mnt/c/Users/jamalids/Downloads/figs/results/max_compression_throughput_pairs.csv'
final_df_A.to_csv(final_output_path_A, index=False)
print(f'CSV with pairs having maximum CompressionThroughput saved to {final_output_path_A}')
###############################################################
df_pairs = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/max_compression_throughput_pairs.csv")
df_entropy = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")

# === Merge
final_df_A = pd.merge(df_pairs, df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='left')

pivot_cr = final_df_A.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
pivot_cr = pivot_cr.dropna(subset=['Full', 'Decompose_Chunk_Parallel'])

pivot_cr = pivot_cr.merge(df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='inner')
#pivot_cr = pivot_cr.sort_values(by=['Domain', 'Entropy']).reset_index(drop=True)
pivot_cr["DatasetID_SortKey"] = pivot_cr["DatasetID"].str.extract(r'D(\d+)').astype(int)
pivot_cr = pivot_cr.sort_values(by="DatasetID_SortKey")


# === Calculate ratio
ratio_series = pivot_cr['Decompose_Chunk_Parallel'] / pivot_cr['Full']

# === PLOT (VLDB Style!)
fig, ax = plt.subplots(figsize=(6.8, 3))  # Same as XOR/RLE/Huffman plots
x = np.arange(len(ratio_series))
bars = ax.bar(
    x, ratio_series,
    width=0.8,
    color="#9467bd",     # <<< Deep Red for FastLZ
    edgecolor="black",
    linewidth=0.3
)


#ax.set_xlabel("Dataset ID", labelpad=7)
ax.set_ylabel("TDT / Standard CR(FastLZ)", labelpad=6)

ax.set_xticks(x)
ax.set_xticklabels(pivot_cr['DatasetID'], rotation=45, ha='right')

# Add value labels above bars
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     if height < 10:
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}",
#                 ha='center', va='bottom', fontsize=7)
# Add horizontal red line at y=1
ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.5)

# === Save
plot_path = "/mnt/c/Users/jamalids/Downloads/figs/foundation-fastlz-matched.pdf"
fig.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(plot_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ FastLZ Plot saved to: {plot_path}")
