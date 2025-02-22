import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt

# ====================================
# 1. Read CSV files, fill NaNs, and combine
# ====================================
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

# ====================================
# 3. Step 2: Select pairs per (DatasetName, Threads)
#
# For each (DatasetName, Threads) group:
#   (a) Select the "Chunked_Decompose_Parallel" row with the maximum CompressionRatio
#       (and if tied, with maximum CompressionThroughput).
#   (b) Then, from the same group, select a corresponding "Full" row.
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
# 4. Step 3: Create two final selections.
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
# 5. Plotting
# We will create two separate figures (one for each final selection) and save them.
#
# For each final selection DataFrame (final_df_A and final_df_B), we create a figure with 3 subplots:
#   Subplot 1: Bar plot of log10(CompressionRatio) by DatasetName (grouped by RunType).
#   Subplot 2: Bar plot of CompressionThroughput by DatasetName (grouped by RunType).
#   Subplot 3: Bar plot of DecompressionThroughput by DatasetName (grouped by RunType).
#
# Above each bar, annotate the number of Threads (from the "Full" row; assumed to be the same for both).
# ====================================

def create_and_save_plot(final_df, title_suffix, output_filename):
    # Pivot the DataFrame so that the index is DatasetName and columns are RunType.
    pivot_log_ratio = final_df.pivot(index='DatasetName', columns='RunType', values='CompressionRatio')
    pivot_log_ratio = np.log10(pivot_log_ratio)
    pivot_comp = final_df.pivot(index='DatasetName', columns='RunType', values='CompressionThroughput')
    pivot_decomp = final_df.pivot(index='DatasetName', columns='RunType', values='DecompressionThroughput')
    # Get Threads from the "Full" row.
    threads_pivot = final_df.pivot(index='DatasetName', columns='RunType', values='Threads')
    # If "Full" column exists, use it; otherwise, use any available column.
    if 'Full' in threads_pivot.columns:
        threads_per_dataset = threads_pivot['Full']
    else:
        threads_per_dataset = threads_pivot.iloc[:, 0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)
    x = np.arange(len(pivot_log_ratio.index))
    width = 0.35

    # Subplot 1: Log10(Compression Ratio)
    ax = axes[0]
    for i, col in enumerate(pivot_log_ratio.columns):
        offset = (i - len(pivot_log_ratio.columns) / 2) * width + width / 2
        bars = ax.bar(x + offset, pivot_log_ratio[col].values, width=width, label=col)
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
        bars = ax.bar(x + offset, pivot_comp[col].values, width=width, label=col)
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
        bars = ax.bar(x + offset, pivot_decomp[col].values, width=width, label=col)
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


# Create and save the two plots.
create_and_save_plot(final_df_A, "(Max CompressionThroughput Pair)",
                     '/home/jamalids/Documents/plot_max_compression_throughput1.png')
create_and_save_plot(final_df_B, "(Max DecompressionThroughput Pair)",
                     '/home/jamalids/Documents/plot_max_decompression_throughput1.png')
