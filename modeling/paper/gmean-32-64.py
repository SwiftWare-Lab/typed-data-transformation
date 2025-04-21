from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd

# Helper to compute geometric mean
def compute_gmean(x):
    return gmean(x) if (x > 0).all() else None
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Define file paths and corresponding compression tool names
files = {
    'lz4': '/home/jamalids/Documents/combine-com-through/lz4.csv',
    'snappy': '/home/jamalids/Documents/combine-com-through/snappy.csv',
    'zlib': '/home/jamalids/Documents/combine-com-through/zlib.csv',
    'zstd': '/home/jamalids/Documents/combine-com-through/zstd.csv',
    'bzip': '/home/jamalids/Documents/combine-com-through/bzip.csv',
    'Lempel-Ziv': '/home/jamalids/Documents/combine-com-through/fastlz.csv'
}

# Read and process each CSV
dfs = []
for comp_tool, path in files.items():
    df = pd.read_csv(path)
    # Add new column with the compression tool name
    df['compression_tool'] = comp_tool
    dfs.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Print the DataFrame columns to check available metric columns
print("Columns in the combined DataFrame:", combined_df.columns)

# Replace RunType values:
# "Chunked_Decompose_Parallel" or "Chunk-decompose_Parallel" become "TDT"
# "Full" becomes "standard"
combined_df['RunType'] = combined_df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel": "TDT",
    "Decompose_Chunk_Parallel": "TDT",
    "Full": "standard"
})

# Save the combined DataFrame to a CSV file
combined_csv_path = "/home/jamalids/Documents/combine-com-through/combine-all.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved to: {combined_csv_path}")
# Load data
df = pd.read_csv("/home/jamalids/Documents/combine-com-through/combine-all.csv")

# Filter for TDT and standard
df_filtered = df[df['RunType'].isin(['TDT', 'standard'])]

# Plot per precision (_f32 and _f64) – Original Code
for suffix in ['32', '64']:
    df_suffix = df_filtered[df_filtered['DatasetName'].str.endswith(suffix)]

    print(f"Processing suffix: {suffix}")
    if df_suffix.empty:
        print(f"No data found for suffix {suffix}")
        continue

    grouped_ratio = df_suffix.groupby(['compression_tool', 'RunType'])['CompressionRatio'] \
        .agg(lambda x: gmean(x) if (x > 0).all() else None).unstack()

    # Plot individual compression ratios
    plt.figure(figsize=(10, 6))
    grouped_ratio.plot(kind='bar')
    plt.xlabel("Compression Tool")
    plt.ylabel("Geometric Mean Compression Ratio")
    plt.title(f"Geometric Mean Compression Ratio for Datasets {suffix} bits")
    plt.xticks(rotation=0)
    plt.legend(title="RunType")
    plt.tight_layout()

    out_path = f"/home/jamalids/Documents/combine-com-through/gmean_ratio{suffix}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved geometric mean plot for {suffix} datasets to {out_path}")

# === New Plot: Improvement Bar Plot (for single/double precision) ===
# Map suffix to label
suffix_label_map = {'32': 'single', '64': 'double'}

# Compute geometric mean improvement per precision and compression tool
improvement_data = []
for suffix in ['32', '64']:
    df_suffix = df_filtered[df_filtered['DatasetName'].str.endswith(suffix)]

    if df_suffix.empty:
        continue

    grouped = df_suffix.groupby(['compression_tool', 'RunType'])['CompressionRatio'] \
        .agg(lambda x: gmean(x) if (x > 0).all() else None).unstack()

    if {'TDT', 'standard'}.issubset(grouped.columns):
        grouped['Improvement (%)'] = (grouped['TDT'] - grouped['standard']) / grouped['standard'] * 100
        grouped['Precision'] = suffix_label_map[suffix]
        improvement_data.append(grouped[['Improvement (%)', 'Precision']].reset_index())

# Combine results
if improvement_data:
    improvement_df = pd.concat(improvement_data, ignore_index=True)

    # Plot
    plt.figure(figsize=(12, 6))
    import seaborn as sns
    sns.barplot(
        data=improvement_df,
        x='Precision',
        y='Improvement (%)',
        hue='compression_tool'
    )

   # plt.title("TDT Compression Ratio Improvement across Precisions")
    plt.ylabel("Improvement (%) in Geometric Mean Compression Ratio")
    plt.xlabel("Precision")
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend(title="Compression Tool", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out_path = "/home/jamalids/Documents/improvement_barplot_precision.png"
    plt.savefig(out_path)
    plt.savefig("/home/jamalids/Documents/gmean-precisions.pdf", format='pdf')
    plt.close()
    print(f"Saved TDT improvement bar plot across precisions to {out_path}")
