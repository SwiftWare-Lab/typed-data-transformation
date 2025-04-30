import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
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
# 1.  READ + MERGE
# ---------------------------------------------------------------
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
    'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv',
    'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
   # 'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv'
}

dfs = []
for tool, path in files.items():
    df = pd.read_csv(path)
    df['compression_tool'] = tool
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df['RunType'] = combined_df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})

combined_df.to_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
from scipy.stats import gmean

# VLDB formatting
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

# Load combined dataset (ensure the file path is correct)
csv_path = Path("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
df = pd.read_csv(csv_path)

# Only include TDT and standard
df_filtered = df[df["RunType"].isin(["TDT", "standard"])]

# Define the final palette
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

final_palette = {}
for tool, col in tool_colors.items():
    final_palette[f"{tool} (TDT)"] = col
    final_palette[f"{tool} (standard)"] = sns.light_palette(col, n_colors=6, input="hex")[3]

# Grouping throughput data
grouped_tp = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionThroughput'].apply(list)
grouped_dtp = df_filtered.groupby(['compression_tool', 'RunType'])['DecompressionThroughput'].apply(list)

def plot_log_throughput(grouped_data, metric_name, output_path):
    tool_order = ['zstd', 'snappy', 'lz4', 'zlib', 'bzip', 'nvCOMP']  # Fixed order
    offset_within_tool = 0.5
    offset_between_tools = 1.3

    data = []
    positions = []
    colors = []
    labels = []

    current_pos = 1
    for tool in tool_order:
        std_key = (tool, 'standard')
        tdt_key = (tool, 'TDT')

        if std_key in grouped_data:
            positions.append(current_pos)
            data.append(grouped_data[std_key])
            colors.append(final_palette.get(f"{tool} (standard)", 'lightgray'))
            labels.append(tool)

        if tdt_key in grouped_data:
            positions.append(current_pos + offset_within_tool)
            data.append(grouped_data[tdt_key])
            colors.append(final_palette.get(f"{tool} (TDT)", 'lightgray'))
            labels.append(f"TDT+{tool}")

        current_pos += offset_between_tools

    fig, ax = plt.subplots(figsize=(6.8, 3))
    for i, (pos, d, c) in enumerate(zip(positions, data, colors)):
        ax.boxplot([d], positions=[pos], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=c, color='black'),
                   medianprops=dict(color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   flierprops=dict(markerfacecolor='gray', marker='o', markersize=3,
                                   linestyle='none', markeredgecolor='black'))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f"log({metric_name})", labelpad=6)
    ax.set_yscale("log")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path.with_suffix(".pdf"), format='pdf', bbox_inches='tight')
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
# Plot compression throughput
plot_log_throughput(grouped_tp, "CT", Path("/mnt/c/Users/jamalids/Downloads/compression_throughput_log"))

# Plot decompression throughput
plot_log_throughput(grouped_dtp, "DT", Path("/mnt/c/Users/jamalids/Downloads/decompression_throughput_log"))

"✅ Log-scale compression and decompression throughput plots saved."
