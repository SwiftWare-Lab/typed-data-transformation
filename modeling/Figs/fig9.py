import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
import matplotlib as mpl

# ────────── VLDB font / figure defaults ──────────
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

# ────────── 1.  Gather all CSVs ──────────
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
    'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv',
    'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
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
combined_df['DatasetName'] = combined_df['DatasetName'].replace({
    "LLama": "LLama_f16"
})

# ────────── 2.  Merge with metadata ──────────
entropy_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping1.csv")
df = pd.merge(combined_df, entropy_df, on='DatasetName', how='inner')

# Keep only TDT & standard rows
df = df[df['RunType'].isin(['TDT', 'standard'])]

# ────────── 3.  Geometric-mean CR per Domain × Tool × RunType ──────────
gmean_df = (
    df.groupby(['Domain', 'compression_tool', 'RunType'])['CompressionRatio']
      .agg(lambda x: gmean(x) if (x > 0).all() else None)
      .reset_index()
      .rename(columns={'CompressionRatio': 'GeometricMeanCR'})
)

# ────────── 4.  Colour palette and **hue order** ──────────
base_colors = {
    'lz4':    '#d62728',
    'snappy': '#6a3d9a',
    'zlib':   '#17becf',
    'zstd':   '#1f77b4',
    'FastLZ': '#9467bd',
    'bzip':   '#2ca02c',
    'nvCOMP': '#bcbd22',
}
final_palette = {}
for tool, col in base_colors.items():
    final_palette[f'{tool} (TDT)']      = col
    final_palette[f'{tool} (standard)'] = sns.light_palette(col, n_colors=6, input="hex")[3]

tool_order = ['lz4', 'snappy', 'zlib', 'zstd', 'FastLZ', 'bzip', 'nvCOMP']    # nvCOMP last
hue_order  = [f'{t} ({rt})' for t in tool_order for rt in ['TDT', 'standard']]  # adjacent pairs

gmean_df['Tool_RunType'] = (
    gmean_df['compression_tool'] + ' (' + gmean_df['RunType'] + ')'
)

# ────────── 6.  Precision-wise Improvement Plot  (unchanged logic) ──────────
suffix_label_map = {'16': 'half', '32': 'single', '64': 'double'}
improvement_data = []

for suffix in ['32', '64', '16']:
    df_suffix = df[df['DatasetName'].str.endswith(suffix)]
    if df_suffix.empty:
        continue
    grouped = (
        df_suffix.groupby(['compression_tool', 'RunType'])['CompressionRatio']
                 .agg(lambda x: gmean(x) if (x > 0).all() else None)
                 .unstack()
    )
    if {'TDT', 'standard'}.issubset(grouped.columns):
        grouped['Improvement (%)'] = (
            (grouped['TDT'] - grouped['standard']) /
            grouped['standard'] * 100
        )
        grouped['Precision'] = suffix_label_map[suffix]
        improvement_data.append(grouped[['Improvement (%)', 'Precision']]
                                .reset_index())

if improvement_data:
    improvement_df = pd.concat(improvement_data, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    sns.barplot(data=improvement_df, x='Precision', y='Improvement (%)',
                hue='compression_tool', hue_order=tool_order,
                palette=[base_colors[t] for t in tool_order], ax=ax)
    ax.set_ylabel("Improvement(%)in GMean CR")
    ax.set_xlabel("Precision")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(title="Compression Tool", bbox_to_anchor=(1.02,1),
              loc='upper left', frameon=False)
    fig.tight_layout(pad=0.5)
    fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-precisions.pdf',
                dpi=300, bbox_inches='tight')
    fig.savefig('/mnt/c/Users/jamalids/Downloads/figs/gmean-precisions.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

print("✅ All plots saved with nvCOMP bars adjacent (TDT+standard) and VLDB formatting.")
