import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
directories = ['/home/jamalids/Documents/results-zstd/all']
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
    full_rows = group[group['RunType'] == 'Full']
    if not full_rows.empty:
        # For "Full" you might or might not want to sort.
        # Here we just pick the first (or best) row:
        corresponding_full = full_rows.iloc[0]
        selected_pairs.append(corresponding_full)

selected_df = pd.DataFrame(selected_pairs)
#To round the CompressionRatio column to three decimal places
selected_df['CompressionRatio'] = selected_df['CompressionRatio'].round(3)
selected_df.to_csv('/home/jamalids/Documents/selected_pairs.csv', index=False)
print("Done. CSV with selected pairs saved.")
final_A = []
chunked_df = selected_df[selected_df['RunType'] == 'Chunked_Decompose_Parallel']
L2_value =24* 1024 *1024,
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
final_output_path_A = '/home/jamalids/Documents/max_compression_throughput_pairs.csv'
final_df_A.to_csv(final_output_path_A, index=False)
print(f'CSV with pairs having maximum CompressionThroughput saved to {final_output_path_A}')
############################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load max_compression_throughput_pairs.csv and entropy results ===
df_pairs = pd.read_csv("/home/jamalids/Documents/max_compression_throughput_pairs.csv")
df_entropy = pd.read_csv("/home/jamalids/Documents/float_level_entropy_results.csv")

# === 2. Merge them by DatasetName ===
merged = pd.merge(df_pairs, df_entropy, on='DatasetName', how='left')

# === 3. Sort by Entropy ===
merged_sorted = merged.sort_values(by='Entropy').reset_index(drop=True)

# === 4. Reuse this sorted DataFrame for the next plots ===
final_df_A = merged_sorted  # Now all your plots will be entropy-sorted
final_output_path_A = '/home/jamalids/Documents/max_compression_throughput_pairs.csv'
final_df_A.to_csv(final_output_path_A, index=False)
# ====================================
# Mapping run types to new labels:
# "Full" -> "Standard Compression"
# "Chunked_Decompose_Parallel" -> "TDT Compressions"
# ====================================
run_type_mapping = {
    "Full": "Standard Compression",
    "Chunked_Decompose_Parallel": "TDT Compressions"
}



##################################
# ------------------------------
# 1. Define domain mapping
# ------------------------------
domain_mapping = {
    "msg_bt": "HPC", "num_brain": "HPC", "num_control": "HPC", "rsim": "HPC",
    "astro_mhd": "HPC", "astro_pt": "HPC", "miranda3d": "HPC", "turbulence": "HPC", "wave": "HPC", "hurricane": "HPC",
    "citytemp": "TS", "ts_gas": "TS", "phone_gyro": "TS", "wesad_chest": "TS", "jane_street": "TS",
    "nyc_taxi2015": "TS", "gas_price": "TS", "solar_wind": "TS",
    "acs_wht": "OBS", "hdr_night": "OBS", "hdr_palermo": "OBS", "hst_wfc3_uvis": "OBS",
    "hst_wfc3_ir": "OBS", "spitzer_irac": "OBS", "g24_78_usb": "OBS", "jw_mirimage": "OBS",
    "tpcH_order": "DB", "tpcxbb_store": "DB", "tpcxbb_web": "DB", "tpch_lineitem": "DB",
    "tpcds_catalog": "DB", "tpcds_store": "DB", "tpch_order": "DB", "tpcds_web": "DB"
}

# ------------------------------
# 2. Clean trailing suffix from DatasetName and map Domain
# ------------------------------
def remove_trailing_pattern(s):
    return re.sub(r'(_?[fF]\d*)$', '', s)

df = pd.read_csv("/home/jamalids/Documents/max_compression_throughput_pairs.csv")
df['NormalizedDatasetName'] = df['DatasetName'].apply(remove_trailing_pattern)
df['Domain'] = df['NormalizedDatasetName'].map(domain_mapping)

# ------------------------------
# 3. Pivot and prepare plot data
# ------------------------------
final_df_A = df.copy()
pivot_cr = final_df_A.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
pivot_cr = pivot_cr.dropna(subset=['Full', 'Chunked_Decompose_Parallel'])

# Extract metadata for sorting and labeling
meta_info = final_df_A.drop_duplicates(subset=['DatasetName'])[['DatasetName', 'Entropy', 'Domain']]
pivot_cr = pivot_cr.merge(meta_info, on='DatasetName', how='left')

# ✅ Sort by Domain then Entropy
pivot_cr = pivot_cr.sort_values(by=['Domain', 'Entropy'])

# ✅ Add short Dataset IDs like D1, D2, ...
pivot_cr['DatasetID'] = ['D' + str(i + 1) for i in range(len(pivot_cr))]

# Compute compression ratio (TDT / Full)
ratio_series = pivot_cr['Chunked_Decompose_Parallel'] / pivot_cr['Full']

# ------------------------------
# 4. Plot
# ------------------------------
plt.figure(figsize=(12, 6))
x = np.arange(len(ratio_series))
plt.bar(x, ratio_series, width=0.5)
plt.xticks(x, pivot_cr['DatasetID'], rotation=45, ha='right')
plt.xlabel("Dataset ID (sorted by Domain → Entropy)")
plt.ylabel("Compressed Size Ratio (TDT / Standard)")
plt.title("Ratio of Compressed Sizes (zstd), Sorted by Domain and Entropy")

# Add value labels
for i, v in enumerate(ratio_series):
    plt.text(x[i], v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plot_path = "/home/jamalids/Documents/plot_compressed_size_ratio_sorted_by_domain_entropy.png"
plt.savefig(plot_path)
plt.close()
print(f"✅ Plot saved to: {plot_path}")

# ------------------------------
# 5. Save Dataset ID Mapping
# ------------------------------
id_map_path = "/home/jamalids/Documents/dataset_id_mapping.csv"
pivot_cr[['DatasetID', 'DatasetName', 'Domain', 'Entropy']].to_csv(id_map_path, index=False)
print(f"✅ Dataset ID mapping saved to: {id_map_path}")
