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

# === 1. Load max_compression_throughput_pairs.csv and entropy results ===
df_pairs = combined_df
df_entropy = pd.read_csv("/home/jamalids/Documents/datasetname/dataset_id_mapping.csv")

# === 2. Merge them by DatasetName ===
df = pd.merge(df_pairs, df_entropy, on='DatasetName', how='left')
df.to_csv(("/home/jamalids/Documents/nwrge.csv"))
######
############################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean

# === Load data ===
# Read combined dataframe created earlier
df = pd.read_csv("/home/jamalids/Documents/nwrge.csv")

# === Compute geometric mean of CompressionRatio grouped by RunType, Domain, and compression_tool ===
gmean_df = df.groupby(['Domain', 'compression_tool', 'RunType'])['CompressionRatio'].agg(gmean).reset_index()
gmean_df.rename(columns={'CompressionRatio': 'GeometricMeanCR'}, inplace=True)

# === Prepare color palette with dark/light variations per tool ===
base_colors = {
    'zstd': '#1f77b4',
    'gzip': '#ff7f0e',
    'bzip': '#2ca02c',
    'lz4': '#d62728',
    'Lempel-Ziv': '#9467bd',
    'blosc': '#8c564b',
    'fpzip': '#e377c2',
    'sz': '#7f7f7f',
    'zfp': '#bcbd22',
    'zlib': '#17becf',
    'snappy': '#6a3d9a'
}

# Create mapping for dark (TDT) and light (standard) variants
color_palette = {}
for tool, color in base_colors.items():
    color_palette[(tool, 'TDT')] = color
    color_palette[(tool, 'standard')] = sns.light_palette(color, n_colors=2, input="hex")[1]

# Create a string label combining tool and RunType for hue
gmean_df['Tool_RunType'] = gmean_df['compression_tool'] + ' (' + gmean_df['RunType'] + ')'
# Let's explicitly generate and assign light/dark colors for each tool's TDT and standard variants

# Generate light variants of the base colors
final_palette = {}
for tool, base_color in base_colors.items():
    final_palette[f'{tool} (TDT)'] = base_color
    light_variant = sns.light_palette(base_color, n_colors=6, input="hex")[3]
    final_palette[f'{tool} (standard)'] = light_variant

# Plot using the updated final_palette with clearer light/dark contrast
plt.figure(figsize=(16, 6))
sns.barplot(
    data=gmean_df,
    x='Domain',
    y='GeometricMeanCR',
    hue='Tool_RunType',
    palette=final_palette,
    dodge=True
)

#plt.title("Geometric Mean Compression Ratio by Domain and Compression Tool (TDT vs Standard)")
plt.ylabel("Geometric Mean Compression Ratio")
plt.xlabel("Application")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Compression Tool ", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

plt.savefig('/home/jamalids/Documents/gmean-each.png')
plt.savefig("/home/jamalids/Documents/gmean-application.pdf", format='pdf')
#########################
# Calculate % improvement in geometric mean compression ratio for each tool within each domain
# (standard - TDT) / standard → higher positive values mean better improvement by TDT
pivot_df = gmean_df.pivot_table(index=['Domain', 'compression_tool'], columns='RunType', values='GeometricMeanCR').reset_index()

# Drop rows where either TDT or standard is missing
pivot_df = pivot_df.dropna(subset=['TDT', 'standard'])

# Calculate percentage improvement
pivot_df['Improvement (%)'] = (pivot_df['TDT'] - pivot_df['standard']) / pivot_df['standard'] * 100

# Plotting
plt.figure(figsize=(16, 6))
sns.barplot(
    data=pivot_df,
    x='Domain',
    y='Improvement (%)',
    hue='compression_tool',
    dodge=True
)

#plt.title("Geometric Mean Compression Ratio Improvement by Domain (TDT vs Standard)")
plt.ylabel("Improvement (%) in Compression Ratio(TDT vs Standard)")
plt.xlabel("Application")
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.legend(title="Compression Tool", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/home/jamalids/Documents/gmeanImp.png')