import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from scipy.stats import gmean
import numpy as np

# ── VLDB font setup ────────────────────────────────────────────────
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

# ---------------------------------------------------------------
# 1.  READ + MERGE  (unchanged)
# ---------------------------------------------------------------
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
    #'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv',
    'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
}

dfs = []
for tool, path in files.items():
    df = pd.read_csv(path)
    df['compression_tool'] = tool
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
# ------------------------------------------------------------------
# Exclude specific precision-suffixed datasets
# ------------------------------------------------------------------
bad_datasets = [
    "ts_gas_f32",
    "phone_gyro_f64",
    "nyc_taxi2015_f64",
    "jane_street_f64",
    "gas_price_f64",
    "tpch_lineitem_f32",
]

combined_df = combined_df[~combined_df["DatasetName"].isin(bad_datasets)].copy()
# ------------------------------------------------------------------

combined_df['RunType'] = combined_df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})

entropy_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
df = pd.merge(combined_df, entropy_df, on='DatasetName', how='inner')
combined_df.to_csv('/mnt/c/Users/jamalids/Downloads/combined_df.csv')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import gmean

# ── VLDB font setup ────────────────────────────────────────────────
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

# ── Tool-specific colors ───────────────────────────────────────────
tool_colors = {
    'zstd':   '#1f77b4',
    'gzip':   '#ff7f0e',
    'lz4':    '#d62728',
    'FastLZ': '#9467bd',
    'huffman':'#8c564b',
    'fpzip':  '#e377c2',
    'xor':    '#7f7f7f',
    'nvCOMP': '#bcbd22',
    'bzip':   '#2ca02c',
    'zlib':   '#17becf',
    'snappy': '#6a3d9a'
}

# ── Read and preprocess ────────────────────────────────────────────
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")

df['RunType'] = df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})

# ───────────────────────────────────────────────────────────────────
# 📊 1. Plot: CRI (TDT vs Standard)
# ───────────────────────────────────────────────────────────────────
df_cri = df[df['RunType'].isin(['TDT', 'standard'])]
pivoted = df_cri.pivot_table(index=['DatasetName', 'compression_tool'], columns='RunType', values='CompressionRatio').dropna()
pivoted['CRI'] = (pivoted['TDT'] - pivoted['standard']) / pivoted['standard'] * 100
cri_per_tool = pivoted.reset_index().groupby('compression_tool')['CRI'].apply(list)

fig, ax = plt.subplots(figsize=(6.8, 3))
for i, (tool, values) in enumerate(cri_per_tool.items(), start=1):
    ax.boxplot([values], positions=[i], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=tool_colors.get(tool, 'lightgray'), color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(markerfacecolor='gray', marker='o', markersize=3,
                               linestyle='none', markeredgecolor='black'))

ax.set_xticks(range(1, len(cri_per_tool) + 1))
ax.set_xticklabels(cri_per_tool.index, rotation=45, ha='right')
ax.set_ylabel("CRI (%)", labelpad=6)
#ax.set_xlabel("Compression Tool", labelpad=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.grid(axis='y')
fig.tight_layout(pad=0.5)

fig.savefig("/mnt/c/Users/jamalids/Downloads/CRI_boxplot.pdf", format='pdf', bbox_inches='tight')
fig.savefig("/mnt/c/Users/jamalids/Downloads/CRI_boxplot.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("✅ CRI boxplot (TDT vs Standard) saved.")


# ───────────────────────────────────────────────────────────────────
# 📊 2. Plot: CR (TDT only) with log scale
# ───────────────────────────────────────────────────────────────────

# ── Read data ──
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
df['RunType'] = df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import matplotlib as mpl
import seaborn as sns

# ── VLDB font setup ──
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

# === Load and clean data ===
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
df['RunType'] = df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})
df = df[df['RunType'].isin(['TDT', 'standard'])]
# Re-importing necessary libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import matplotlib as mpl
import seaborn as sns

# ── VLDB font setup ──
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

# Tool-specific colors
tool_colors = {
    'zstd':   '#1f77b4',
    'gzip':   '#ff7f0e',
    'lz4':    '#d62728',
    'FastLZ': '#9467bd',
    'huffman':'#8c564b',
    'fpzip':  '#e377c2',
    'xor':    '#7f7f7f',
    'nvCOMP': '#bcbd22',
    'bzip':   '#2ca02c',
    'zlib':   '#17becf',
    'snappy': '#6a3d9a'
}

# Define final_palette with standard and TDT variants
final_palette = {}
for tool, col in tool_colors.items():
    final_palette[f'{tool} (standard)'] = sns.light_palette(col, n_colors=6, input="hex")[3]
    final_palette[f'{tool} (TDT)'] = col

# Load the combined_df CSV again after reset
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
df = df[df['RunType'].isin(['TDT', 'standard'])]

# Group by (tool, RunType)
grouped_cr = df.groupby(['compression_tool', 'RunType'])['CompressionRatio'].apply(list)

tool_order = ['zstd', 'snappy', 'lz4','zlib', 'bzip', 'nvCOMP']

offset_between_tools = 1.2
offset_within_tool = 0.45

positions = []
data = []
colors = []
labels = []

current_pos = 1
for tool in tool_order:
    std_key = (tool, 'standard')
    tdt_key = (tool, 'TDT')

    if std_key in grouped_cr:
        positions.append(current_pos)
        data.append(grouped_cr[std_key])
        colors.append(final_palette.get(f"{tool} (standard)", 'lightgray'))
        labels.append(tool)

    if tdt_key in grouped_cr:
        positions.append(current_pos + offset_within_tool)
        data.append(grouped_cr[tdt_key])
        colors.append(final_palette.get(f"{tool} (TDT)", 'lightgray'))
        labels.append(f"TDT+{tool}")

    current_pos += offset_between_tools

# Create boxplot
fig, ax = plt.subplots(figsize=(6.8, 3))

for pos, vals, color in zip(positions, data, colors):
    ax.boxplot([vals], positions=[pos], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor=color, color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(markerfacecolor='gray', marker='o', markersize=3,
                               linestyle='none', markeredgecolor='black'))

# Set x-axis
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha='right')

# Axis labels and style
ax.set_ylabel("Compression Ratio", labelpad=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 4)  # crop y-axis range for better visibility
#ax.grid(axis='y')

fig.tight_layout(pad=0.5)

# Save
fig.savefig("/mnt/c/Users/jamalids/Downloads/CRI_boxplot_TDT_standard.pdf", format='pdf', bbox_inches='tight')
fig.savefig("/mnt/c/Users/jamalids/Downloads/CRI_boxplot_TDT_standard.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("✅ Paired standard & TDT boxplot saved with adjusted spacing and y-axis range.")
