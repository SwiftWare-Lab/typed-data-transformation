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
#directories = ['/home/jamalids/Documents/results1']
directories = ['/home/jamalids/Documents/results-zstd']
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
# selected_pairs = []
# for (dataset, threads,BlockSize), group in median_df.groupby(['DatasetName', 'Threads','BlockSize']):
#     decompose_rows = group[group['RunType'] == 'Decompose_Chunk_Parallel']
#     if not decompose_rows.empty:
#         sorted_decompose = decompose_rows.sort_values(
#             by=['CompressionRatio', 'CompressionThroughput'], ascending=False
#         )
#         chosen_decompose = sorted_decompose.iloc[0]
#         selected_pairs.append(chosen_decompose)
#         full_rows = group[group['RunType'] == 'Full']
#         if not full_rows.empty:
#             corresponding_full = full_rows.iloc[0]
#             selected_pairs.append(corresponding_full)
#
# selected_df = pd.DataFrame(selected_pairs)
# selected_output_path = '/home/jamalids/Documents/selected_pairs.csv'
# selected_df.to_csv(selected_output_path, index=False)
# print(f'CSV with selected pairs saved to {selected_output_path}')

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
selected_df.to_csv('/home/jamalids/Documents/selected_pairs.csv', index=False)
print("Done. CSV with selected pairs saved.")

# ====================================
# 4. Create two final selections.
# 4A. Final Selection A: For each DatasetName, choose the pair with maximum CompressionThroughput
#     (using the "Decompose_Chunk_Parallel" row).
# 4B. Final Selection B: For each DatasetName, choose the pair with maximum DecompressionThroughput
#     (using the "Full" row).
# ====================================
final_A = []
chunked_df = selected_df[selected_df['RunType'] == 'Decompose_Chunk_Parallel']
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
full_rows = selected_df[selected_df['RunType'] == 'Decompose_Chunk_Parallel']
if not full_rows.empty:
    best_full = full_rows.loc[full_rows.groupby('DatasetName')['DecompressionThroughput'].idxmax()]
    for idx, row in best_full.iterrows():
        dataset = row['DatasetName']
        threads = row['Threads']
        final_B.append(row)
        chunked_row = selected_df[(selected_df['DatasetName'] == dataset) &
                                  (selected_df['Threads'] == threads) &
                                  (selected_df['RunType'] == 'Full')]
        if not chunked_row.empty:
            final_B.append(chunked_row.iloc[0])
final_df_B = pd.DataFrame(final_B)
final_output_path_B = '/home/jamalids/Documents/max_decompression_throughput_pairs.csv'
final_df_B.to_csv(final_output_path_B, index=False)
print(f'CSV with pairs having maximum DecompressionThroughput saved to {final_output_path_B}')

# ====================================
# Mapping run types to new labels:
# "Full" -> "Standard Compression"
# "Decompose_Chunk_Parallel" -> "TDT Compressions"
# ====================================
run_type_mapping = {
    "Full": "Standard Compression",
    "Decompose_Chunk_Parallel": "TDT Compressions"
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
gmean_data = median_df[median_df['RunType'].isin(['Decompose_Chunk_Parallel', 'Full'])]
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
# This plot uses the best "Decompose_Chunk_Parallel" rows (i.e. TDT Compressions)
# and shows the relation between TotalValues (primary axis) and BlockSize (secondary axis).

chunked_df = median_df[median_df['RunType'] == 'Decompose_Chunk_Parallel']
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
# ################################
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
#
# # Filter for "Decompose_Chunk_Parallel" rows.
# chunked_df = median_df[median_df['RunType'] == 'Decompose_Chunk_Parallel']
# # Further filter to only include rows with allowed BlockSize values.
# # allowed_block_sizes = [655360, 1024000, 102400000]
# # chunked_df = chunked_df[chunked_df['BlockSize'].isin(allowed_block_sizes)]
#
# # For each DatasetName, select the row with the maximum CompressionRatio.
# best_chunked_df = chunked_df.loc[chunked_df.groupby('DatasetName')['CompressionRatio'].idxmax()][
#     ['DatasetName', 'CompressionRatio', 'BlockSize']
# ]
#
# # Create numeric x positions.
# x = np.arange(len(best_chunked_df))
#
# # Create the primary axis for CompressionRatio.
# fig, ax1 = plt.subplots(figsize=(10, 6))
# bars = ax1.bar(x, best_chunked_df['CompressionRatio'], color='C0', alpha=0.7, label='Best Compression Ratio')
# ax1.set_xlabel("DatasetName")
# ax1.set_ylabel("Best Compression Ratio", color='C0')
# ax1.tick_params(axis='y', labelcolor='C0')
# ax1.set_xticks(x)
# ax1.set_xticklabels(best_chunked_df['DatasetName'], rotation=45, ha='right')
#
# # Create a secondary axis for BlockSize.
# ax2 = ax1.twinx()
# ax2.plot(x, best_chunked_df['BlockSize'], color='C1', marker='o', linestyle='-', linewidth=2, label='BlockSize')
# ax2.set_ylabel("BlockSize", color='C1')
# ax2.tick_params(axis='y', labelcolor='C1')
#
# # Set the secondary y-axis to logarithmic scale so the ticks are evenly spaced.
# ax2.set_yscale('log')
# # Define custom tick positions.
# custom_ticks = [ 100 *1024,
#
#  300 *1024,
#     400 *1024,
#     640* 1024,
#     768*1024,
#     1024*1024,
#     10*1024*1024,
#     24 *1024 *1024,
#     30 *1024 *1024,
#     40 *1024 *1024,
# ]
# ax2.set_yticks(custom_ticks)
# # Set the tick labels as strings.
# ax2.set_yticklabels([str(tick) for tick in custom_ticks])
#
# # Optionally combine legends from both axes.
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
#
# plt.title("Best Compression Ratio and BlockSize per Dataset\n(TDT)")
# plt.tight_layout()
# best_comp_ratio_dual_output_path = '/home/jamalids/Documents/plot_best_compression_ratio_dual_axis.png'
# plt.savefig(best_comp_ratio_dual_output_path)
# plt.close()
# print(f'Dual-axis plot saved to {best_comp_ratio_dual_output_path}')
import numpy as np
import matplotlib.pyplot as plt

# Define the special block sizes that determine cache levels.
special_blocks = [
    (640 * 1024, 'L1'),
    (24 * 1024 * 1024, 'L2'),
    (30 * 1024 * 1024, 'L3')
]

# Loop over each thread (assuming the column is named 'Thread').
for thread in sorted(chunked_df['Threads'].unique()):
    # Filter the DataFrame for the current thread.
    thread_df = chunked_df[chunked_df['Threads'] == thread]

    # For each DatasetName, select the row with the maximum CompressionRatio.
    best_thread_df = thread_df.loc[thread_df.groupby('DatasetName')['CompressionRatio'].idxmax()][
        ['DatasetName', 'CompressionRatio', 'BlockSize', 'CompressionThroughput']
    ]

    # Sort by BlockSize for consistent categorical ordering.
    best_thread_df = best_thread_df.sort_values('BlockSize').reset_index(drop=True)

    # Create a mapping for block sizes to categorical indices.
    unique_block_sizes = sorted(best_thread_df['BlockSize'].unique())
    blocksize_to_cat = {bs: i for i, bs in enumerate(unique_block_sizes)}

    # Map each BlockSize to its corresponding categorical index.
    cat_block_sizes = best_thread_df['BlockSize'].map(blocksize_to_cat)

    # Create x positions (one per dataset).
    x = np.arange(len(best_thread_df))

    # Create the primary axis for CompressionRatio (bar plot).
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x, best_thread_df['CompressionRatio'], color='C0', alpha=0.7,
                   label='Best Compression Ratio')
    ax1.set_xlabel("DatasetName")
    ax1.set_ylabel("Best Compression Ratio", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(best_thread_df['DatasetName'], rotation=45, ha='right')

    # Create a secondary axis for BlockSize (plotted as a categorical variable).
    ax2 = ax1.twinx()
    ax2.plot(x, cat_block_sizes, color='C1', marker='o', linestyle='-', linewidth=2,
             label='BlockSize')
    ax2.set_ylabel("BlockSize", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.set_yticks(range(len(unique_block_sizes)))
    ax2.set_yticklabels([str(bs) for bs in unique_block_sizes])

    # Add horizontal dashed lines to mark the special cache thresholds.
    for block_val, cache_label in special_blocks:
        if block_val in blocksize_to_cat:
            pos = blocksize_to_cat[block_val]
            ax2.axhline(y=pos, color='grey', linestyle='--', linewidth=1)
            # Annotate the cache level near the right edge.
            ax2.text(len(x) - 0.5, pos, cache_label, color='grey',
                     va='bottom', ha='right', fontsize=10)

    # Create a third axis for Compression Throughput.
    ax3 = ax1.twinx()
    # Offset the third axis to the right.
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    # Plot Compression Throughput as a line.
    ax3.plot(x, best_thread_df['CompressionThroughput'], color='C2', marker='s',
             linestyle='-', linewidth=2, label='Compression Throughput')
    ax3.set_ylabel("Compression Throughput", color='C2')
    ax3.tick_params(axis='y', labelcolor='C2')

    # Combine legends from all axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.title(f"Best Compression Ratio, BlockSize, and Compression Throughput per Dataset\n(TDT) - Thread {thread}")
    plt.tight_layout()

    # Save the figure with a thread-specific filename.
    output_path = f'/home/jamalids/Documents/plot_best_compression_ratio_dual_axis_thread_{thread}.png'
    plt.savefig(output_path)
    plt.close()
    print(f'Plot saved to {output_path}')

#####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Filter for TDT runs.
tdt_df = median_df[median_df['RunType'] == 'Decompose_Chunk_Parallel']

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
# Convert it to a DataFrame:
diff_comp_ratio_df = pd.DataFrame(diff_comp_ratio, columns=['diff_comp_ratio'])

# Now you can call .to_csv():
diff_comp_ratio_df.to_csv('/home/jamalids/Documents/diff_comp_ratio.csv', index=False)

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
############################
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Read the median CSV and filter for the desired RunType
# ---------------------------------------------------
median_csv_path = '/home/jamalids/Documents/combined_median_rows.csv'
median_df = pd.read_csv(median_csv_path)

# Only consider rows where RunType is 'Decompose_Chunk_Parallel'
median_df = median_df[median_df['RunType'] == 'Decompose_Chunk_Parallel']

# ---------------------------------------------------
# 2. Group by DatasetName and Threads and compute max and min for each metric
# ---------------------------------------------------
grouped = median_df.groupby(['DatasetName', 'Threads']).agg({
    'CompressionThroughput': ['max', 'min'],
    'DecompressionThroughput': ['max', 'min'],
    'CompressionRatio': ['max', 'min']
}).reset_index()

# Flatten the MultiIndex columns
grouped.columns = ['DatasetName', 'Threads',
                   'CT_max', 'CT_min',
                   'DT_max', 'DT_min',
                   'CR_max', 'CR_min']

# Compute the difference (max - min) for each metric
grouped['CT_diff'] = grouped['CT_max'] - grouped['CT_min']
grouped['DT_diff'] = grouped['DT_max'] - grouped['DT_min']
grouped['CR_diff'] = grouped['CR_max'] - grouped['CR_min']

# ---------------------------------------------------
# 3. Plot differences per dataset for each thread (1, 8, 16)
# ---------------------------------------------------
thread_values = [1, 8, 16]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

for idx, thread_val in enumerate(thread_values):
    ax = axes[idx]
    # Select groups for the current thread value
    df_thread = grouped[grouped['Threads'] == thread_val]

    # Sort by DatasetName for consistent ordering on the x-axis
    df_thread = df_thread.sort_values('DatasetName')
    x = df_thread['DatasetName']

    # Plot the differences for each metric
    ax.plot(x, df_thread['CT_diff'], marker='o', label='Compression Throughput Diff')
    ax.plot(x, df_thread['DT_diff'], marker='o', label='Decompression Throughput Diff')
    ax.plot(x, df_thread['CR_diff'], marker='o', label='Compression Ratio Diff')

    ax.set_title(f"Threads = {thread_val}")
    ax.set_xlabel('DatasetName')
    if idx == 0:
        ax.set_ylabel('Difference (max - min)')

    # Rotate x-axis labels for clarity
    ax.tick_params(axis='x', rotation=90)
    ax.legend(loc='best')

plt.tight_layout()

# Save the final plot to file
output_path = '/home/jamalids/Documents/selected_pairs_diff_by_dataset_thread.png'
plt.savefig(output_path)
plt.close()

print(f"Plot of metric differences for each dataset (grouped by thread) saved to {output_path}")
########################
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Read the median CSV (already created) and filter for RunType = 'Decompose_Chunk_Parallel'
# ---------------------------------------------------
median_csv_path = '/home/jamalids/Documents/combined_median_rows.csv'
df = pd.read_csv(median_csv_path)
df = df[df['RunType'] == 'Decompose_Chunk_Parallel']

# ---------------------------------------------------
# 2. Define the two block sizes (L1 and L2) and filter the data accordingly
# ---------------------------------------------------
L1_value = 640 * 1024  # L1 block size (e.g., 655360)
L2_value = 24 * 1024 * 1024  # L2 block size (e.g., 25165824)

# Keep only rows with the two specified block sizes
df_block = df[df['BlockSize'].isin([L1_value, L2_value])]

# ---------------------------------------------------
# 3. Pivot the data to have separate columns for L1 and L2 for each metric
# ---------------------------------------------------
# We create pivot tables for each metric using DatasetName and Threads as the index,
# and BlockSize as the columns. Since the CSV is already median-aggregated, we assume one value per group.
pivot_ct = df_block.pivot_table(index=['DatasetName', 'Threads'], columns='BlockSize', values='CompressionThroughput')
pivot_dt = df_block.pivot_table(index=['DatasetName', 'Threads'], columns='BlockSize', values='DecompressionThroughput')
pivot_cr = df_block.pivot_table(index=['DatasetName', 'Threads'], columns='BlockSize', values='CompressionRatio')

# ---------------------------------------------------
# 4. Compute the difference (L2 - L1) for each metric
# ---------------------------------------------------
# We use the .get() method with a fallback of NaN in case a value is missing.
pivot_ct['Diff'] = pivot_ct.get(L2_value, float('nan')) - pivot_ct.get(L1_value, float('nan'))
pivot_dt['Diff'] = pivot_dt.get(L2_value, float('nan')) - pivot_dt.get(L1_value, float('nan'))
pivot_cr['Diff'] = pivot_cr.get(L2_value, float('nan')) - pivot_cr.get(L1_value, float('nan'))

# Reset the index to merge these difference values
pivot_ct = pivot_ct.reset_index()[['DatasetName', 'Threads', 'Diff']].rename(columns={'Diff': 'CT_Diff'})
pivot_dt = pivot_dt.reset_index()[['DatasetName', 'Threads', 'Diff']].rename(columns={'Diff': 'DT_Diff'})
pivot_cr = pivot_cr.reset_index()[['DatasetName', 'Threads', 'Diff']].rename(columns={'Diff': 'CR_Diff'})

# Merge the difference DataFrames on DatasetName and Threads
merged_diff = pivot_ct.merge(pivot_dt, on=['DatasetName', 'Threads']).merge(pivot_cr, on=['DatasetName', 'Threads'])

# ---------------------------------------------------
# 5. Plot the differences for each thread in separate subplots
# ---------------------------------------------------
thread_values = [1, 8, 16]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=False)

for idx, thread_val in enumerate(thread_values):
    ax = axes[idx]
    # Select data for the current thread value and sort by DatasetName for clarity
    df_thread = merged_diff[merged_diff['Threads'] == thread_val].sort_values('DatasetName')

    # Use DatasetName as the x-axis labels
    x = df_thread['DatasetName']

    # Plot differences for each metric (L2 - L1)
    ax.plot(x, df_thread['CT_Diff'], marker='o', label='Compression Throughput Diff')
    ax.plot(x, df_thread['DT_Diff'], marker='o', label='Decompression Throughput Diff')
    ax.plot(x, df_thread['CR_Diff'], marker='o', label='Compression Ratio Diff')

    ax.set_title(f"Threads = {thread_val}")
    ax.set_xlabel("DatasetName")
    if idx == 0:
        ax.set_ylabel("Difference (L2 - L1)")

    ax.tick_params(axis='x', rotation=90)
    ax.legend(loc='best')

plt.tight_layout()
output_path = '/home/jamalids/Documents/block_size_diff_l.png'
plt.savefig(output_path)
plt.close()

print(f"Block size difference plot saved to {output_path}")
