# import pandas as pd
# from scipy.stats import gmean
# import matplotlib.pyplot as plt
#
# # ── Load combined zstd data ─────────────────────────────────────────
# combined = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
# combined['compression_tool'] = 'zstd'
#
# # Map RunType to standard vs TDT
# combined['RunType'] = combined['RunType'].replace({
#     "Chunked_Decompose_Parallel": "TDT",
#     "Chunk-decompose_Parallel":  "TDT",
#     "Decompose_Chunk_Parallel":  "TDT",
#     "Component":                 "TDT",
#     "Whole":                     "standard",
#     "Full":                      "standard",
#     "Chunked_parallel":          "standard"
# })
#
# # Keep only std & TDT
# combined = combined[combined['RunType'].isin(['standard', 'TDT'])]
#
# # ── Load fpzip data ─────────────────────────────────────────────────
# fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
# fpzip = fpzip.rename(columns={'Dataset': 'DatasetName', 'FPZIP_Ratio': 'CompressionRatio'})
# fpzip['compression_tool'] = 'fpzip'
# fpzip['RunType'] = 'standard'
#
# # ── Restrict to common DatasetName ──────────────────────────────────
# common = set(combined['DatasetName']).intersection(fpzip['DatasetName'])
# combined = combined[combined['DatasetName'].isin(common)]
# fpzip   = fpzip[fpzip['DatasetName'].isin(common)]
#
# # ── Compute geometric means ────────────────────────────────────────
# g_fpzip    = gmean(fpzip['CompressionRatio'])
# g_zstd_std = gmean(combined[combined['RunType']=='standard']['CompressionRatio'])
# g_zstd_tdt = gmean(combined[combined['RunType']=='TDT']['CompressionRatio'])
#
# # ── Plot bar chart ─────────────────────────────────────────────────
# labels = ['fpzip', 'zstd', 'TDT+zstd']
# values = [g_fpzip, g_zstd_std, g_zstd_tdt]
#
# fig, ax = plt.subplots(figsize=(6.8, 3))
# ax.bar(labels, values, width=0.6)
# ax.set_xlabel("Compression tool")
# ax.set_ylabel("Geometric mean compression ratio")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('/home/jamalids/Documents/fpzip-compare.png')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean, sem

# ── 1) Load & preprocess zstd data ─────────────────────────────────
combined = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
combined['compression_tool'] = 'zstd'
combined['RunType'] = combined['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Whole":                     "standard",
    "Full":                      "standard",
    "Chunked_parallel":          "standard"
})
combined = combined[combined['RunType'].isin(['standard', 'TDT'])]

# ── 2) Load & preprocess fpzip data ────────────────────────────────
fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
fpzip = fpzip.rename(columns={
    'Dataset':      'DatasetName',
    'FPZIP_Ratio':  'CompressionRatio'
})
fpzip['compression_tool'] = 'fpzip'
fpzip['RunType']         = 'standard'

# ── 3) Keep only the common datasets ───────────────────────────────
common = set(combined['DatasetName']).intersection(fpzip['DatasetName'])
combined = combined[combined['DatasetName'].isin(common)]
fpzip    = fpzip[fpzip['DatasetName'].isin(common)]

# ── 4) Build a single DataFrame for plotting ───────────────────────
df_plot = pd.concat([
    combined[['DatasetName','compression_tool','RunType','CompressionRatio']],
    fpzip[['DatasetName','compression_tool','RunType','CompressionRatio']]
], ignore_index=True)
# Compute means & errors
groups = [('fpzip','standard'), ('zstd','standard'), ('zstd','TDT')]
labels = ['fpzip', 'zstd', 'TDT+zstd']
means, errs = [], []
for tool, run in groups:
    vals = df_plot[
        (df_plot['compression_tool']==tool) &
        (df_plot['RunType']==run)
    ]['CompressionRatio'].dropna()
    means.append(gmean(vals))
    errs.append(sem(vals))

# Build your color list as before
base_colors = {
    'zstd':   '#1f77b4',
    'snappy': '#6a3d9a',
    'lz4':    '#d62728',
    'zlib':   '#17becf',
    'bzip':   '#2ca02c',
    'nvCOMP': '#bcbd22',
}
final_palette = {}
for t, c in base_colors.items():
    final_palette[f'TDT+{t}'] = c
    final_palette[f'{t}']     = sns.light_palette(c, n_colors=6, input="hex")[3]
fpzip_color = "#e377c2"
colors = [fpzip_color,
          final_palette['zstd'],
          final_palette['TDT+zstd']]

# Plot
sns.set_style("ticks")            # lighter style, no big grid
fig, ax = plt.subplots(figsize=(5, 3))
y_pos = range(len(labels))
bars = ax.barh(y_pos, means, xerr=errs, height=0.6, color=colors, capsize=4)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("Geometric mean compression ratio", fontsize=13)
ax.set_xlim(0, max(means) * 1.2)
sns.despine(left=True, bottom=False)

# --- OPTION A: text *outside* with white bbox
for bar, m in zip(bars, means):
    ax.text(
        bar.get_width() + max(means)*0.02,
        bar.get_y() + bar.get_height()/2,
        f"{m:.2f}",
        va="center", ha="left", fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, pad=1, edgecolor="none")
    )

# --- OPTION B: text *inside* in white
# for bar, m in zip(bars, means):
#     width = bar.get_width()
#     ax.text(
#         width * 0.98,
#         bar.get_y() + bar.get_height()/2,
#         f"{m:.2f}",
#         va="center", ha="right", fontsize=11, color="white", fontweight="bold"
#     )

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/fpzip-compare.png', dpi=300)
