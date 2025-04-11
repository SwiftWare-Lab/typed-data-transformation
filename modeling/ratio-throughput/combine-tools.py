import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Define file paths and corresponding compression tool names
files = {
    'lz4': '/home/jamalids/Documents/combine-com-through/lz4.csv',
    'snappy': '/home/jamalids/Documents/combine-com-through/snappy.csv',
    'zlib': '/home/jamalids/Documents/combine-com-through/zlib.csv',
    'zstd': '/home/jamalids/Documents/combine-com-through/zstd.csv'
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
    "Full": "standard"
})

# Save the combined DataFrame to a CSV file
combined_csv_path = "/home/jamalids/Documents/combine-com-through/combine-all.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved to: {combined_csv_path}")

###############################
# Create Bar Plots (already in your code)
###############################

# Filter the DataFrame to include only rows with RunType "TDT" or "standard"
df_filtered = combined_df[combined_df['RunType'].isin(['TDT', 'standard'])]

# 1. Bar plot for geometric mean CompressionRatio
grouped_ratio = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionRatio'] \
    .agg(lambda x: gmean(x) if (x > 0).all() else None).unstack()
ax_ratio = grouped_ratio.plot(kind='bar', figsize=(10, 6))
ax_ratio.set_xlabel("Compression Tool")
ax_ratio.set_ylabel("Geometric Mean Compression Ratio")
ax_ratio.set_title("Geometric Mean Compression Ratio by Compression Tool and RunType")
plt.xticks(rotation=0)
plt.legend(title="RunType")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/combine.png")
plt.close()


# Function to compute geometric mean if all values are positive
def compute_gmean(x):
    return gmean(x) if (x > 0).all() else None


# 2. Bar plot for geometric mean CompressionThroughput and DecompressionThroughput
grouped_comp = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionThroughput'] \
    .agg(compute_gmean).unstack()
grouped_decomp = df_filtered.groupby(['compression_tool', 'RunType'])['DecompressionThroughput'] \
    .agg(compute_gmean).unstack()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

grouped_comp.plot(kind='bar', ax=axes[0])
axes[0].set_xlabel("Compression Tool")
axes[0].set_ylabel("Geometric Mean Compression Throughput")
axes[0].set_title("Compression Throughput by Compression Tool and RunType")
axes[0].legend(title="RunType")
axes[0].tick_params(axis='x', rotation=0)

grouped_decomp.plot(kind='bar', ax=axes[1])
axes[1].set_xlabel("Compression Tool")
axes[1].set_ylabel("Geometric Mean Decompression Throughput")
axes[1].set_title("Decompression Throughput by Compression Tool and RunType")
axes[1].legend(title="RunType")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/throughput.png")
plt.close()


###############################
# 3. Boxplots for Throughput Metrics
###############################

# Define a function to create and save a boxplot for a given metric
def create_boxplot(metric, ylabel, title, save_path):
    # Get the unique compression tools (sorted for consistent order)
    tools = sorted(df_filtered['compression_tool'].unique())

    # Prepare lists to hold data for each RunType for every tool
    data_tdt = []
    data_std = []
    for tool in tools:
        data_tdt.append(
            df_filtered[(df_filtered['compression_tool'] == tool) & (df_filtered['RunType'] == 'TDT')][metric].dropna())
        data_std.append(df_filtered[(df_filtered['compression_tool'] == tool) & (df_filtered['RunType'] == 'standard')][
                            metric].dropna())

    # Define positions for boxplots: for each tool, we create two positions (one for each RunType)
    positions_tdt = [i * 3 + 1 for i in range(len(tools))]
    positions_std = [i * 3 + 2 for i in range(len(tools))]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplots for TDT and standard
    bp_tdt = ax.boxplot(data_tdt, positions=positions_tdt, widths=0.6, patch_artist=True, showfliers=True)
    bp_std = ax.boxplot(data_std, positions=positions_std, widths=0.6, patch_artist=True, showfliers=True)

    # Set custom colors for clarity
    for box in bp_tdt['boxes']:
        box.set_facecolor('lightblue')
    for box in bp_std['boxes']:
        box.set_facecolor('lightgreen')

    # Set x-axis ticks in the middle of the two boxes per tool
    ax.set_xticks([i * 3 + 1.5 for i in range(len(tools))])
    ax.set_xticklabels(tools)
    ax.set_xlabel("Compression Tool")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Create custom legend
    import matplotlib.patches as mpatches
    patch_tdt = mpatches.Patch(color='lightblue', label='TDT')
    patch_std = mpatches.Patch(color='lightgreen', label='standard')
    ax.legend(handles=[patch_tdt, patch_std])

    plt.tight_layout()
    plt.savefig(save_path)
   #plt.show()


# Create and save boxplot for CompressionThroughput
create_boxplot(
    metric="CompressionThroughput",
    ylabel="Compression Throughput",
    title="Boxplot of Compression Throughput by Tool and RunType",
    save_path="/home/jamalids/Documents/combine-com-through/throughput_boxplot.png"
)

# Create and save boxplot for DecompressionThroughput
create_boxplot(
    metric="DecompressionThroughput",
    ylabel="Decompression Throughput",
    title="Boxplot of Decompression Throughput by Tool and RunType",
    save_path="/home/jamalids/Documents/combine-com-through/decompression_boxplot.png"
)
##############improvement
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Read the combined CSV file
df = pd.read_csv("/home/jamalids/Documents/combine-com-through/combine-all.csv")

# Filter for RunType "TDT" and "standard" only
df_filtered = df[df['RunType'].isin(['TDT', 'standard'])]

# Helper: Compute geometric mean (works if all values > 0)
def compute_gmean(x):
    return gmean(x) if (x > 0).all() else None

########################################
# 1. Line Plot for Geometric Mean CompressionRatio
########################################

# Group by compression_tool and RunType and compute geometric mean of CompressionRatio
grouped_ratio = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionRatio']\
                .agg(lambda x: gmean(x) if (x > 0).all() else None).unstack()

plt.figure(figsize=(8,6))
# Using a line plot with markers; each column (RunType) becomes a separate line.
grouped_ratio.plot(kind='line', marker='o', linewidth=2)
plt.xlabel("Compression Tool")
plt.ylabel("Geometric Mean Compression Ratio")
plt.title("Geometric Mean Compression Ratio by Compression Tool and RunType")
plt.xticks(rotation=0)
plt.legend(title="RunType")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/geomean_compression_ratio_line.png")


########################################
# 2. Line Plot for Percent Improvement in CompressionThroughput
########################################

# Group by compression_tool and RunType for CompressionThroughput
grouped_comp = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionThroughput']\
                .agg(compute_gmean).unstack()

# Calculate percent improvement: ((TDT - standard) / standard) * 100
improvement_comp = (grouped_comp['TDT'] - grouped_comp['standard']) / grouped_comp['standard'] * 100

plt.figure(figsize=(8,6))
# Plot percent improvement as a line with markers; x-axis is the compression tools
plt.plot(improvement_comp.index, improvement_comp.values, marker='o', linestyle='-', linewidth=2, color='skyblue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("Compression Tool")
plt.ylabel("Percent Improvement (%)")
plt.title("Percent Improvement in Compression Throughput (TDT vs Standard)")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/improvement_compression_throughput_line.png")


########################################
# 3. Line Plot for Percent Improvement in DecompressionThroughput
########################################

# Group by compression_tool and RunType for DecompressionThroughput
grouped_decomp = df_filtered.groupby(['compression_tool', 'RunType'])['DecompressionThroughput']\
                  .agg(compute_gmean).unstack()

# Calculate percent improvement
improvement_decomp = (grouped_decomp['TDT'] - grouped_decomp['standard']) / grouped_decomp['standard'] * 100

plt.figure(figsize=(8,6))
plt.plot(improvement_decomp.index, improvement_decomp.values, marker='o', linestyle='-', linewidth=2, color='lightgreen')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("Compression Tool")
plt.ylabel("Percent Improvement (%)")
plt.title("Percent Improvement in Decompression Throughput (TDT vs Standard)")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/improvement_decompression_throughput_line.png")
##########################################3
# Group by compression_tool and RunType, computing the geometric mean of CompressionRatio
grouped_ratio = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionRatio']\
                .agg(lambda x: gmean(x) if (x > 0).all() else None).unstack()

# Calculate percent improvement: ((TDT - standard) / standard) * 100
improvement_ratio = (grouped_ratio['TDT'] - grouped_ratio['standard']) / grouped_ratio['standard'] * 100

# Create a line plot for percent improvement in CompressionRatio
plt.figure(figsize=(8,6))
plt.plot(improvement_ratio.index, improvement_ratio.values, marker='o', linestyle='-', linewidth=2, color='purple')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("Compression Tool")
plt.ylabel("Percent Improvement (%)")
plt.title("Percent Improvement in Compression Ratio (TDT vs Standard)")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/combine-com-through/improvement_compression_ratio_line.png")

#######################################
# --- after you compute grouped_ratio = df_filtered.groupby(...) ... .unstack() ---

# Save the geometric‐mean CompressionRatio for both RunTypes into one CSV
# grouped_ratio is a DataFrame with index=compression_tool and columns=['standard','TDT']
gmean_csv_path = "/home/jamalids/Documents/combine-com-through/gmean_compression_ratio.csv"
grouped_ratio.to_csv(gmean_csv_path, index=True)
print(f"Geometric‐mean compression ratios saved to: {gmean_csv_path}")

# If you also want separate CSVs for TDT and standard:
gmean_tdt = grouped_ratio['TDT'].reset_index().rename(columns={'TDT':'gmean_TDT'})
gmean_std = grouped_ratio['standard'].reset_index().rename(columns={'standard':'gmean_standard'})
split = pd.merge(gmean_tdt, gmean_std, on='compression_tool')
split_csv_path = "/home/jamalids/Documents/combine-com-through/gmean_compression_ratio_split.csv"
split.to_csv(split_csv_path, index=False)
print(f"Split geometric‐mean ratios saved to: {split_csv_path}")

#import pandas as pd
from scipy.stats import gmean

# --- after your grouped_ratio = ... .unstack() ---

# Drop any missing values (just in case) and compute the overall geometric mean across tools
std_vals = grouped_ratio['standard'].dropna().values
tdt_vals = grouped_ratio['TDT'].dropna().values

overall_std = gmean(std_vals) if len(std_vals) else None
overall_tdt = gmean(tdt_vals) if len(tdt_vals) else None

# Build a small DataFrame with the two numbers
df_overall = pd.DataFrame({
    'RunType': ['standard', 'TDT'],
    'Overall_GeometricMean_CompressionRatio': [overall_std, overall_tdt]
})

# Save to CSV
overall_csv = "/home/jamalids/Documents/combine-com-through/overall_gmean_compression_ratio.csv"
df_overall.to_csv(overall_csv, index=False)
print(f"Overall geometric‐mean compression ratios saved to: {overall_csv}")
print(df_overall)

#################
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# --- assume grouped_ratio, grouped_comp, grouped_decomp exist here ---

# 1. Extract the arrays of geometric‐mean values for each RunType
std_ratio_vals = grouped_ratio.loc[:, 'standard'].dropna().values
tdt_ratio_vals = grouped_ratio.loc[:, 'TDT'].dropna().values

std_comp_thr_vals = grouped_comp.loc[:, 'standard'].dropna().values
tdt_comp_thr_vals = grouped_comp.loc[:, 'TDT'].dropna().values

std_decomp_thr_vals = grouped_decomp.loc[:, 'standard'].dropna().values
tdt_decomp_thr_vals = grouped_decomp.loc[:, 'TDT'].dropna().values

# 2. Compute overall geometric means across tools
overall_std_ratio = gmean(std_ratio_vals)      if len(std_ratio_vals)      else None
overall_tdt_ratio = gmean(tdt_ratio_vals)      if len(tdt_ratio_vals)      else None
overall_std_comp_thr = gmean(std_comp_thr_vals) if len(std_comp_thr_vals) else None
overall_tdt_comp_thr = gmean(tdt_comp_thr_vals) if len(tdt_comp_thr_vals) else None
overall_std_decomp_thr = gmean(std_decomp_thr_vals) if len(std_decomp_thr_vals) else None
overall_tdt_decomp_thr = gmean(tdt_decomp_thr_vals) if len(tdt_decomp_thr_vals) else None

# 3. Build a DataFrame and save to CSV
df_overall = pd.DataFrame({
    'RunType': ['standard', 'TDT'],
    'Overall_Gmean_CompressionRatio': [overall_std_ratio, overall_tdt_ratio],
    'Overall_Gmean_CompressionThroughput': [overall_std_comp_thr, overall_tdt_comp_thr],
    'Overall_Gmean_DecompressionThroughput': [overall_std_decomp_thr, overall_tdt_decomp_thr]
})

out_csv = "/home/jamalids/Documents/combine-com-through/overall_metrics.csv"
df_overall.to_csv(out_csv, index=False)
print(f"Overall metrics saved to: {out_csv}")

# 4. Plot each metric as a simple two‑bar chart
metrics = [
    ('Overall_Gmean_CompressionRatio',      'Geometric Mean Compression Ratio',      'ratio'),
    ('Overall_Gmean_CompressionThroughput','Geometric Mean Compression Throughput','comp_throughput'),
    ('Overall_Gmean_DecompressionThroughput','Geometric Mean Decompression Throughput','decomp_throughput')
]
df_overall.to_csv("/home/jamalids/Documents/combine-com-through/overall.csv")
for col, ylabel, fname in metrics:
    plt.figure(figsize=(6,4))
    plt.bar(df_overall['RunType'], df_overall[col], color=['gray','skyblue'])
    plt.xlabel("RunType")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: standard vs TDT")
    plt.tight_layout()
    out_plot = f"/home/jamalids/Documents/combine-com-through/overall_{fname}.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved plot: {out_plot}")
