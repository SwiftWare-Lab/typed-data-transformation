import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ====================================
# 1. Read CSV files, fill NaNs, and combine
# ====================================
directories = ['/home/jamalids/Documents/results1/64-selection']
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

# ====================================
# 3. Select pairs per (DatasetName, Threads)
# ====================================
selected_pairs = []
for (dataset, threads), group in median_df.groupby(['DatasetName', 'Threads']):
    decompose_rows = group[group['RunType'] == 'Chunked_Decompose_Parallel']
    if not decompose_rows.empty:
        sorted_decompose = decompose_rows.sort_values(
            by=['CompressionRatio', 'CompressionThroughput'], ascending=False
        )
        chosen_decompose = sorted_decompose.iloc[0]
        selected_pairs.append(chosen_decompose)
        full_rows = group[group['RunType'] == 'Full']
        if not full_rows.empty:
            corresponding_full = full_rows.iloc[0]
            selected_pairs.append(corresponding_full)

selected_df = pd.DataFrame(selected_pairs)
selected_output_path = '/home/jamalids/Documents/selected_pairs.csv'
selected_df.to_csv(selected_output_path, index=False)
print(f'CSV with selected pairs saved to {selected_output_path}')

# ====================================
# 4. Create two final selections.
# 4A. Final Selection A: For each DatasetName, choose the pair with maximum CompressionThroughput
#     (using the "Chunked_Decompose_Parallel" row).
# 4B. Final Selection B: For each DatasetName, choose the pair with maximum DecompressionThroughput
#     (using the "Full" row).
# ====================================
final_A = []
chunked_df = selected_df[selected_df['RunType'] == 'Chunked_Decompose_Parallel']
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

final_B = []
full_rows = selected_df[selected_df['RunType'] == 'Full']
if not full_rows.empty:
    best_full = full_rows.loc[full_rows.groupby('DatasetName')['DecompressionThroughput'].idxmax()]
    for idx, row in best_full.iterrows():
        dataset = row['DatasetName']
        threads = row['Threads']
        final_B.append(row)
        chunked_row = selected_df[(selected_df['DatasetName'] == dataset) &
                                  (selected_df['Threads'] == threads) &
                                  (selected_df['RunType'] == 'Chunked_Decompose_Parallel')]
        if not chunked_row.empty:
            final_B.append(chunked_row.iloc[0])
final_df_B = pd.DataFrame(final_B)
final_output_path_B = '/home/jamalids/Documents/max_decompression_throughput_pairs.csv'
final_df_B.to_csv(final_output_path_B, index=False)
print(f'CSV with pairs having maximum DecompressionThroughput saved to {final_output_path_B}')

# ====================================
# Mapping run types to new labels:
# "Full" -> "Standard Compression"
# "Chunked_Decompose_Parallel" -> "TDT Compressions"
# ====================================
run_type_mapping = {
    "Full": "Standard Compression",
    "Chunked_Decompose_Parallel": "TDT Compressions"
}

# ====================================
# 5. Plotting: Create grouped bar plots for final selections
# ====================================
def create_and_save_plot(final_df, title_suffix, output_filename):
    # Pivot the DataFrame so that the index is DatasetName and columns are RunType.
    pivot_log_ratio = final_df.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
    pivot_log_ratio = np.log10(pivot_log_ratio)
    pivot_comp = final_df.pivot(index='DatasetName', columns='RunType', values='CompressionThroughput')
    pivot_decomp = final_df.pivot(index='DatasetName', columns='RunType', values='DecompressionThroughput')
    # Get Threads from the "Full" row.
    threads_pivot = final_df.pivot(index='DatasetName', columns='RunType', values='Threads')
    if 'Full' in threads_pivot.columns:
        threads_per_dataset = threads_pivot['Full']
    else:
        threads_per_dataset = threads_pivot.iloc[:, 0]

    # Replace run type column names using the mapping.
    pivot_log_ratio.columns = [run_type_mapping.get(col, col) for col in pivot_log_ratio.columns]
    pivot_comp.columns = [run_type_mapping.get(col, col) for col in pivot_comp.columns]
    pivot_decomp.columns = [run_type_mapping.get(col, col) for col in pivot_decomp.columns]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)
    x = np.arange(len(pivot_log_ratio.index))
    width = 0.35

    # Subplot 1: Log10(Compression Ratio)
    ax = axes[0]
    for i, col in enumerate(pivot_log_ratio.columns):
        offset = (i - len(pivot_log_ratio.columns) / 2) * width + width / 2
        ax.bar(x + offset, pivot_log_ratio[col].values, width=width, label=col)
    ax.set_title("Log10(Compression Ratio) Comparison " + title_suffix)
    ax.set_ylabel("Log10(Compression Ratio)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_log_ratio.index, rotation=45, ha='right')
    ax.legend()
    for j, dataset in enumerate(pivot_log_ratio.index):
        for i, col in enumerate(pivot_log_ratio.columns):
            thread_str = str(int(threads_per_dataset.loc[dataset]))
            ax.text(x[j] + (i - len(pivot_log_ratio.columns) / 2) * width + width / 2,
                    pivot_log_ratio.loc[dataset, col] + 0.01,
                    thread_str, ha='center', va='bottom', fontsize=10, color='black')

    # Subplot 2: Compression Throughput
    ax = axes[1]
    for i, col in enumerate(pivot_comp.columns):
        offset = (i - len(pivot_comp.columns) / 2) * width + width / 2
        ax.bar(x + offset, pivot_comp[col].values, width=width, label=col)
    ax.set_title("Compression Throughput Comparison " + title_suffix)
    ax.set_ylabel("Compression Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_comp.index, rotation=45, ha='right')
    ax.legend()
    for j, dataset in enumerate(pivot_comp.index):
        for i, col in enumerate(pivot_comp.columns):
            thread_str = str(int(threads_per_dataset.loc[dataset]))
            ax.text(x[j] + (i - len(pivot_comp.columns) / 2) * width + width / 2,
                    pivot_comp.loc[dataset, col] + 0.01,
                    thread_str, ha='center', va='bottom', fontsize=10, color='black')

    # Subplot 3: Decompression Throughput
    ax = axes[2]
    for i, col in enumerate(pivot_decomp.columns):
        offset = (i - len(pivot_decomp.columns) / 2) * width + width / 2
        ax.bar(x + offset, pivot_decomp[col].values, width=width, label=col)
    ax.set_title("Decompression Throughput Comparison " + title_suffix)
    ax.set_ylabel("Decompression Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_decomp.index, rotation=45, ha='right')
    ax.legend()
    for j, dataset in enumerate(pivot_decomp.index):
        for i, col in enumerate(pivot_decomp.columns):
            thread_str = str(int(threads_per_dataset.loc[dataset]))
            ax.text(x[j] + (i - len(pivot_decomp.columns) / 2) * width + width / 2,
                    pivot_decomp.loc[dataset, col] + 0.01,
                    thread_str, ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f'Plot saved to {output_filename}')

# Create and save the two grouped bar plots.
create_and_save_plot(final_df_A, "(Max CompressionThroughput Pair)",
                     '/home/jamalids/Documents/plot_max_compression_throughput1.png')
create_and_save_plot(final_df_B, "(Max DecompressionThroughput Pair)",
                     '/home/jamalids/Documents/plot_max_decompression_throughput1.png')

# ====================================
# 6. Additional Plots
# 6A. Geometric Mean Comparison Plot
# ====================================
def safe_gmean(series):
    # Filter to keep only positive values to avoid issues with log(0) or negative numbers.
    positive_vals = series[series > 0]
    if len(positive_vals) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive_vals)))

# Filter for the two RunTypes of interest.
gmean_data = median_df[median_df['RunType'].isin(['Chunked_Decompose_Parallel', 'Full'])]
gmean_df = gmean_data.groupby('RunType')['CompressionRatio'].apply(safe_gmean).reset_index()
gmean_df.columns = ['RunType', 'GeometricMeanCompressionRatio']

# Replace run type names using the mapping.
gmean_df['RunType'] = gmean_df['RunType'].map(run_type_mapping)

# Create a bar plot for the geometric means.
plt.figure(figsize=(8, 6))
bars = plt.bar(gmean_df['RunType'], gmean_df['GeometricMeanCompressionRatio'])
plt.title("Geometric Mean of CompressionRatio by RunType")
plt.xlabel("RunType")
plt.ylabel("Geometric Mean of CompressionRatio")
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
gmean_output_path = '/home/jamalids/Documents/plot_geometric_mean_compression_ratio.png'
plt.savefig(gmean_output_path)
plt.close()
print(f'Geometric Mean plot saved to {gmean_output_path}')

# ====================================
# 6B. Dual-axis Plot: TotalValues vs. BlockSize
# ====================================
# This plot uses the best "Chunked_Decompose_Parallel" rows (i.e. TDT Compressions)
# and shows the relation between TotalValues (primary axis) and BlockSize (secondary axis).

chunked_df = median_df[median_df['RunType'] == 'Chunked_Decompose_Parallel']
best_chunked_df = chunked_df.loc[chunked_df.groupby('DatasetName')['CompressionRatio'].idxmax()].copy()

if 'TotalValues' not in best_chunked_df.columns:
    raise ValueError("The column 'TotalValues' does not exist in the data. Please check your combined CSV.")

x = np.arange(len(best_chunked_df))
dataset_labels = best_chunked_df['DatasetName']

fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(x, best_chunked_df['TotalValues'], color='C0', alpha=0.7, label='TotalValues')
ax1.set_xlabel("DatasetName")
ax1.set_ylabel("TotalValues", color='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.set_xticks(x)
ax1.set_xticklabels(dataset_labels, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(x, best_chunked_df['BlockSize'], color='C1', marker='o', linestyle='-', linewidth=2, label='BlockSize')
ax2.set_ylabel("BlockSize", color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

plt.title("Relation between TotalValues and BlockSize\nfor Best TDT Compressions per Dataset")
plt.tight_layout()
output_path = '/home/jamalids/Documents/plot_totalvalues_vs_blocksize.png'
plt.savefig(output_path)
plt.close()
print(f'Dual-axis plot saved to {output_path}')
################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Filter for "Chunked_Decompose_Parallel" rows.
chunked_df = median_df[median_df['RunType'] == 'Chunked_Decompose_Parallel']
# Further filter to only include rows with allowed BlockSize values.
allowed_block_sizes = [655360, 1024000, 102400000]
chunked_df = chunked_df[chunked_df['BlockSize'].isin(allowed_block_sizes)]

# For each DatasetName, select the row with the maximum CompressionRatio.
best_chunked_df = chunked_df.loc[chunked_df.groupby('DatasetName')['CompressionRatio'].idxmax()][
    ['DatasetName', 'CompressionRatio', 'BlockSize']
]

# Create numeric x positions.
x = np.arange(len(best_chunked_df))

# Create the primary axis for CompressionRatio.
fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(x, best_chunked_df['CompressionRatio'], color='C0', alpha=0.7, label='Best Compression Ratio')
ax1.set_xlabel("DatasetName")
ax1.set_ylabel("Best Compression Ratio", color='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax1.set_xticks(x)
ax1.set_xticklabels(best_chunked_df['DatasetName'], rotation=45, ha='right')

# Create a secondary axis for BlockSize.
ax2 = ax1.twinx()
ax2.plot(x, best_chunked_df['BlockSize'], color='C1', marker='o', linestyle='-', linewidth=2, label='BlockSize')
ax2.set_ylabel("BlockSize", color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

# Set the secondary y-axis to logarithmic scale so the ticks are evenly spaced.
ax2.set_yscale('log')
# Define custom tick positions.
custom_ticks = [655360, 1024000, 102400000]
ax2.set_yticks(custom_ticks)
# Set the tick labels as strings.
ax2.set_yticklabels([str(tick) for tick in custom_ticks])

# Optionally combine legends from both axes.
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Best Compression Ratio and BlockSize per Dataset\n(TDT)")
plt.tight_layout()
best_comp_ratio_dual_output_path = '/home/jamalids/Documents/plot_best_compression_ratio_dual_axis.png'
plt.savefig(best_comp_ratio_dual_output_path)
plt.close()
print(f'Dual-axis plot saved to {best_comp_ratio_dual_output_path}')
#####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Filter for TDT runs.
tdt_df = median_df[median_df['RunType'] == 'Chunked_Decompose_Parallel']

# Further filter for the two BlockSize values of interest.
blocksize_low = 655360
blocksize_high = 102400000

df_low = tdt_df[tdt_df['BlockSize'] == blocksize_low]
df_high = tdt_df[tdt_df['BlockSize'] == blocksize_high]

# Merge the two subsets on DatasetName.
# We'll add suffixes _low and _high for the two block sizes.
merged_df = pd.merge(df_low, df_high, on="DatasetName", suffixes=('_low', '_high'))

# Compute differences for each metric (high minus low)
merged_df['Diff_CompressionRatio'] = merged_df['CompressionRatio_high'] - merged_df['CompressionRatio_low']
merged_df['Diff_CompressionThroughput'] = merged_df['CompressionThroughput_high'] - merged_df['CompressionThroughput_low']
merged_df['Diff_DecompressionThroughput'] = merged_df['DecompressionThroughput_high'] - merged_df['DecompressionThroughput_low']

# Sort by DatasetName (or any order you prefer)
merged_df.sort_values("DatasetName", inplace=True)

# Prepare x-axis (dataset names) and y-axis values for each difference.
datasets = merged_df["DatasetName"].tolist()
x = np.arange(len(datasets))

diff_comp_ratio = merged_df["Diff_CompressionRatio"].values
diff_comp_throughput = merged_df["Diff_CompressionThroughput"].values
diff_decomp_throughput = merged_df["Diff_DecompressionThroughput"].values

# Plot the differences with 3 separate lines.
plt.figure(figsize=(12, 6))
plt.plot(x, diff_comp_ratio, marker='o', linestyle='-', label="Diff Compression Ratio")
plt.plot(x, diff_comp_throughput, marker='s', linestyle='-', label="Diff Compression Throughput")
plt.plot(x, diff_decomp_throughput, marker='^', linestyle='-', label="Diff Decompression Throughput")

plt.xlabel("DatasetName")
plt.ylabel("Difference (High BlockSize - Low BlockSize)")
plt.title("Differences in Metrics for BlockSize 102400000 vs 655360 (TDT)")
plt.xticks(x, datasets, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

output_path = '/home/jamalids/Documents/plot_metric_differences.png'
plt.savefig(output_path)
plt.close()
print(f'Difference plot saved to {output_path}')
##################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_gmean(series):
    # Compute geometric mean of positive values only.
    positive_vals = series[series > 0]
    if len(positive_vals) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive_vals)))

# Filter for TDT runs (i.e. RunType == "Chunked_Decompose_Parallel")
tdt_df = median_df[median_df['RunType'] == 'Chunked_Decompose_Parallel']

# Further filter to only include rows with BlockSize equal to 655360 or 102400000.
tdt_df = tdt_df[tdt_df['BlockSize'].isin([655360, 102400000])]

# Group by BlockSize and compute the geometric mean of CompressionRatio for each group.
geom_df = tdt_df.groupby('BlockSize')['CompressionRatio'].apply(safe_gmean).reset_index()

# For plotting, sort the data by BlockSize.
geom_df.sort_values('BlockSize', inplace=True)

# Create a bar plot for the geometric means.
x = np.arange(len(geom_df))  # two groups
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, geom_df['CompressionRatio'], width, color='C0')

ax.set_xlabel("BlockSize")
ax.set_ylabel("Geometric Mean Compression Ratio")
ax.set_title("Geometric Mean Compression Ratio for TDT\n(655360 vs. 102400000)")
ax.set_xticks(x)
ax.set_xticklabels(geom_df['BlockSize'].astype(str))
plt.tight_layout()

output_path = '/home/jamalids/Documents/plot_geom_mean_comp_ratio.png'
plt.savefig(output_path)
plt.close()
print(f"Geometric mean plot saved to {output_path}")
