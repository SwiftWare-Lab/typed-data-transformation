import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
#directories = ['/mnt/c/Users/jamalids/Downloads/figs/results/results-zstd/all']
directories = ['/mnt/c/Users/jamalids/Downloads/figs/results/results-lz4']
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
    # --- 1) Try to pick Chunked_Decompose_Parallel (if present) ---
    decompose_rows = group[group['RunType'] == 'Chunked_Decompose_Parallel']
    if not decompose_rows.empty:
        sorted_decompose = decompose_rows.sort_values(
            by=['CompressionRatio', 'CompressionThroughput'],
            ascending=False
        )
        chosen_decompose = sorted_decompose.iloc[0]
        selected_pairs.append(chosen_decompose)

    # --- 2) Also pick the Full row (if present) ---
    #full_rows = group[group['RunType'] == 'Full']
    full_rows = group[(group['RunType'] == 'Chunked_parallel') & (group['BlockSize'] == 31457280)]

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
chunked_df = selected_df[selected_df['RunType'] == 'Chunked_Decompose_Parallel']
#L2_value =1024 *1024,
L2_value =786432 #lz4
chunked_df = chunked_df [chunked_df ['BlockSize'] == L2_value].copy()

if not chunked_df.empty:
    best_chunked = chunked_df.loc[chunked_df.groupby('DatasetName')['CompressionThroughput'].idxmax()]
    for idx, row in best_chunked.iterrows():
        dataset = row['DatasetName']
        threads = row['Threads']
        final_A.append(row)
        full_row = selected_df[(selected_df['DatasetName'] == dataset) &
                               (selected_df['Threads'] == threads) &
                               (selected_df['RunType'] == 'Chunked_parallel')]
                              # (selected_df['RunType'] == 'Full')]
        if not full_row.empty:
            final_A.append(full_row.iloc[0])
final_df_A = pd.DataFrame(final_A)
final_output_path_A = '/mnt/c/Users/jamalids/Downloads/figs/results/max_compression_throughput_pairs.csv'
final_df_A.to_csv(final_output_path_A, index=False)
print(f'CSV with pairs having maximum CompressionThroughput saved to {final_output_path_A}')
############################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load max_compression_throughput_pairs.csv and entropy results ===
df_pairs = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/max_compression_throughput_pairs.csv")
df_entropy = pd.read_csv(("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv"))

# === 2. Merge them by DatasetName ===
# df = pd.merge(df_pairs, df_entropy, on='DatasetName', how='left')
# final_df_A = df.copy()
# pivot_cr = final_df_A.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
# pivot_cr = pivot_cr.dropna(subset=['Full', 'Chunked_Decompose_Parallel'])
#
# # Extract metadata for sorting and labeling
# meta_info = final_df_A.drop_duplicates(subset=['DatasetName'])[['DatasetName', 'Entropy', 'Domain']]
# pivot_cr = pivot_cr.merge(meta_info, on='DatasetName', how='left')
#
# # ✅ Sort by Domain then Entropy
# pivot_cr = pivot_cr.sort_values(by=['Domain', 'Entropy'])
#
# # ✅ Add short Dataset IDs like D1, D2, ...
#
# pivot_cr['DatasetID'] = ['D' + str(i + 1) for i in range(len(pivot_cr))]
final_df_A = pd.merge(df_pairs, df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='left')

pivot_cr = final_df_A.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
#pivot_cr = pivot_cr.dropna(subset=['Full', 'Chunked_Decompose_Parallel'])
pivot_cr = pivot_cr.dropna(subset=['Chunked_parallel', 'Chunked_Decompose_Parallel'])



pivot_cr = pivot_cr.merge(df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='inner')
#pivot_cr = pivot_cr.sort_values(by=['Domain', 'Entropy']).reset_index(drop=True)
pivot_cr["DatasetID_SortKey"] = pivot_cr["DatasetID"].str.extract(r'D(\d+)').astype(int)
pivot_cr = pivot_cr.sort_values(by="DatasetID_SortKey")


# Compute compression ratio (TDT / Full)
#ratio_series = pivot_cr['Chunked_Decompose_Parallel'] / pivot_cr['Full']
ratio_series = pivot_cr['Chunked_Decompose_Parallel'] / pivot_cr['Chunked_parallel']


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font configuration
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

# === Prepare Data ===
# df_pairs and df_entropy are already merged and prepared
# You have 'pivot_cr' and 'ratio_series' ready

sorted_df = pivot_cr.copy()
#sorted_df["Ratio"] = sorted_df["Chunked_Decompose_Parallel"] / sorted_df["Full"]
sorted_df["Ratio"] = sorted_df["Chunked_Decompose_Parallel"] / sorted_df["Chunked_parallel"]


# === Step 3: Plot ===
fig, ax = plt.subplots(figsize=(6.8, 3))

# Set your bar color for Zstd plot
#bar_color = "#1f77b4"  # Example color, you can change

bar_color = "#d62728"
bars = ax.bar(sorted_df["DatasetID"], sorted_df["Ratio"],
              width=0.8, color=bar_color, edgecolor="black", linewidth=0.3)

# Labels
#ax.set_xlabel(r"Dataset ID", labelpad=6)
ax.set_ylabel(r"TDT / Standard CR (ZLIB)", labelpad=6)

# X ticks
if len(sorted_df) > 40:
    step = 5
    ticks = sorted_df["DatasetID"][::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45)
else:
    ax.set_xticks(sorted_df["DatasetID"])
    ax.set_xticklabels(sorted_df["DatasetID"], rotation=45)

# Value labels
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     if height < 10:
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03, f"{height:.2f}",
#                 ha="center", va="bottom", fontsize=6)

# Remove top and right spines
ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.5)

# === Save ===
output_base = "/mnt/c/Users/jamalids/Downloads/figs/results/lz"
fig.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{output_base}.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"✅ Plot and dataset ID mapping saved successfully!")
