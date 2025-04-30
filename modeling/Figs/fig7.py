# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
#
#
# # Define file paths and corresponding compression tool names
# files = {
#     'lz4': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
#     'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
#     'zlib': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
#     'zstd': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
#     'bzip': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
#     'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv',
#     'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
# }
#
# # Read and process each CSV
# dfs = []
# for comp_tool, path in files.items():
#     df = pd.read_csv(path)
#     # Add new column with the compression tool name
#     df['compression_tool'] = comp_tool
#     dfs.append(df)
#
# # Combine all dataframes into one
# combined_df = pd.concat(dfs, ignore_index=True)
#
# # Print the DataFrame columns to check available metric columns
# print("Columns in the combined DataFrame:", combined_df.columns)
#
# # Replace RunType values:
# # "Chunked_Decompose_Parallel" or "Chunk-decompose_Parallel" become "TDT"
# # "Full" becomes "standard"
# combined_df['RunType'] = combined_df['RunType'].replace({
#     "Chunked_Decompose_Parallel": "TDT",
#     "Chunk-decompose_Parallel": "TDT",
#     "Decompose_Chunk_Parallel": "TDT",
#     "Component":"TDT",
#     "Whole":"standard",
#     "Full": "standard",
#     "Chunked_parallel":"standard"
# })
#
# # Save the combined DataFrame to a CSV file
# # combined_csv_path = "/mnt/c/Users/jamalids/Downloads/figs/combine-com-through/combine-all.csv/combine.csv"
# # combined_df.to_csv(combined_csv_path, index=False)
# # print(f"Combined CSV saved to: {combined_csv_path}")
#
# # === 1. Load max_compression_throughput_pairs.csv and entropy results ===
# df_pairs = combined_df
# df_entropy =  pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
#
# # === 2. Merge them by DatasetName ===
# df = pd.merge(df_pairs, df_entropy, on='DatasetName', how='inner')
# df.to_csv(("/mnt/c/Users/jamalids/Downloads/figs/nwrge.csv"))
# ######
# ############################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from scipy.stats import gmean
# import matplotlib as mpl
#
# # 🌟 VLDB Font Setup
# mpl.rcParams.update({
#     "text.usetex": False,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
#     "font.size": 11,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })
#
#
# # === Load data ===
# # Read merged dataframe
# df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/nwrge.csv")
#
# # === Compute geometric mean of CompressionRatio grouped by RunType, Domain, and compression_tool ===
# gmean_df = df.groupby(['Domain', 'compression_tool', 'RunType'])['CompressionRatio'].agg(gmean).reset_index()
# gmean_df.rename(columns={'CompressionRatio': 'GeometricMeanCR'}, inplace=True)
#
# # === Color palette for tools ===
# base_colors = {
#     'zstd': '#1f77b4',
#     'gzip': '#ff7f0e',
#
#     'lz4': '#d62728',
#     'FastLZ': '#9467bd',
#     'huffman': '#8c564b',
#     'fpzip': '#e377c2',
#     'xor': '#7f7f7f',
#     'nvCOMP': '#bcbd22',
#     'bzip': '#2ca02c',
#     'zlib': '#17becf',
#     'snappy': '#6a3d9a'
# }
#
# # Generate light variant for "standard"
# final_palette = {}
# for tool, base_color in base_colors.items():
#     final_palette[f'{tool} (TDT)'] = base_color
#     light_variant = sns.light_palette(base_color, n_colors=6, input="hex")[3]
#     final_palette[f'{tool} (standard)'] = light_variant
#
# # Prepare string label for hue
# gmean_df['Tool_RunType'] = gmean_df['compression_tool'] + ' (' + gmean_df['RunType'] + ')'
# gmean_df.to_csv("/mnt/c/Users/jamalids/Downloads/figs/final_palette.csv")
# # === First plot: Geometric Mean Compression Ratio ===
# fig, ax = plt.subplots(figsize=(6.8, 3))  # VLDB preferred figure size
#
# sns.barplot(
#     data=gmean_df,
#     x='Domain',
#     y='GeometricMeanCR',
#     hue='Tool_RunType',
#     palette=final_palette,
#     dodge=True,
#     ax=ax
# )
#
# ax.set_ylabel("Geometric Mean CR", labelpad=4)
# ax.set_xlabel("Application", labelpad=4)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#
# # Clean up spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # Legend outside
# ax.legend(title="Compression Tool", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
#
# fig.tight_layout(pad=0.5)
#
# # Save
# fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-each.png', bbox_inches="tight", dpi=300)
# fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-application.pdf', bbox_inches="tight", dpi=300)
# plt.close(fig)
#
# print("✅ Saved gmean-each plots.")
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gmean
import matplotlib as mpl

# ── VLDB Font Setup ────────────────────────────────────────────────
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

# === Load data ===
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/nwrge.csv")

# === Compute geometric mean of CompressionRatio grouped by RunType, Domain, and compression_tool ===
gmean_df = (
    df.groupby(['Domain', 'compression_tool', 'RunType'])['CompressionRatio']
    .agg(gmean)
    .reset_index()
    .rename(columns={'CompressionRatio': 'GeometricMeanCR'})
)

# === Color palette for tools ===
base_colors = {
    'zstd':   '#1f77b4',
    'snappy': '#6a3d9a',
    'lz4':    '#d62728',
    'zlib':   '#17becf',
    'bzip':   '#2ca02c',
    'nvCOMP': '#bcbd22',
}

# Create final_palette
final_palette = {}
for tool, base_color in base_colors.items():
    final_palette[f'TDT+{tool}'] = base_color
    final_palette[f'{tool}'] = sns.light_palette(base_color, n_colors=6, input="hex")[3]

# Modify Tool_RunType column as requested
def format_tool_label(row):
    return f'TDT+{row["compression_tool"]}' if row['RunType'] == 'TDT' else row["compression_tool"]

gmean_df['Tool_RunType'] = gmean_df.apply(format_tool_label, axis=1)

# Define order for hue (TDT+tool, tool)
tool_order = ['zstd', 'snappy', 'lz4', 'zlib', 'bzip', 'nvCOMP']
hue_order = [f'TDT+{t}' for t in tool_order] + [t for t in tool_order]

# === PLOT ===
fig, ax = plt.subplots(figsize=(6.8, 3))  # VLDB preferred figure size

sns.barplot(
    data=gmean_df,
    x='Domain',
    y='GeometricMeanCR',
    hue='Tool_RunType',
    hue_order=hue_order,
    palette=final_palette,
    dodge=True,
    ax=ax
)

ax.set_ylabel("Geometric Mean CR", labelpad=4)
ax.set_xlabel("Application", labelpad=4)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(title="Compression Tool", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

fig.tight_layout(pad=0.5)

# Save
fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-each.png', dpi=300, bbox_inches="tight")
fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-application.pdf', dpi=300, bbox_inches="tight")
plt.close(fig)

print("✅ Legend uses TDT+tool and tool only, ordered as requested.")
