import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gmean

# Apply VLDB-style fonts and plot settings
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ======================
# Load combined CSV
# ======================
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/result-throughput/combine-all.csv")

# ======================
# 1. Geometric Mean Plot
# ======================
grouped = df.groupby(['compression_tool', 'RunType'])
gmean_df = grouped.agg({
    'CompressionThroughput': lambda x: gmean(x[x > 0]),
    'DecompressionThroughput': lambda x: gmean(x[x > 0])
}).reset_index()

gmean_df['RunType'] = gmean_df['RunType'].replace({
    'Full': 'standard',
    'Chunked_Decompose_Parallel': 'first chunk then decompose',
    'Decompose_Chunk_Parallel': 'first decompose then chunk',
    'full': 'standard'
})

pivot_df = gmean_df.pivot(index='compression_tool', columns='RunType', values=['CompressionThroughput', 'DecompressionThroughput'])

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)

bar_width = 0.25
x = np.arange(len(pivot_df.index))

# Compression Throughput
for i, runtype in enumerate(pivot_df['CompressionThroughput'].columns):
    ax1.bar(x + i * bar_width, pivot_df['CompressionThroughput'][runtype], width=bar_width, label=runtype)

ax1.set_ylabel("Throughput")
ax1.set_title("Geometric Mean Compression Throughput")
#ax1.grid(True)

# Decompression Throughput
for i, runtype in enumerate(pivot_df['DecompressionThroughput'].columns):
    ax2.bar(x + i * bar_width, pivot_df['DecompressionThroughput'][runtype], width=bar_width, label=runtype)

ax2.set_ylabel("Throughput")
ax2.set_xlabel("Compression Tool")
ax2.set_title("Geometric Mean Decompression Throughput")
#ax2.grid(True)

ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(pivot_df.index, rotation=45, ha='right')

ax1.legend(ncol=1)
ax2.legend(ncol=1)

plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/Throughput-geomean.pdf", bbox_inches='tight')
plt.close()


# ======================
# 2. Average Plot
# ======================
mean_df = grouped.agg({
    'CompressionThroughput': lambda x: x[x > 0].mean(),
    'DecompressionThroughput': lambda x: x[x > 0].mean()
}).reset_index()

pivot_df = mean_df.pivot(index='compression_tool', columns='RunType', values=['CompressionThroughput', 'DecompressionThroughput'])

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)

# Compression Throughput
for i, runtype in enumerate(pivot_df['CompressionThroughput'].columns):
    ax1.bar(x + i * bar_width, pivot_df['CompressionThroughput'][runtype], width=bar_width, label=runtype)

ax1.set_ylabel("Throughput")
ax1.set_title("Average Compression Throughput")
ax1.grid(True)

# Decompression Throughput
for i, runtype in enumerate(pivot_df['DecompressionThroughput'].columns):
    ax2.bar(x + i * bar_width, pivot_df['DecompressionThroughput'][runtype], width=bar_width, label=runtype)

ax2.set_ylabel("Throughput")
ax2.set_xlabel("Compression Tool")
ax2.set_title("Average Decompression Throughput")
ax2.grid(True)

ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(pivot_df.index, rotation=45, ha='right')

ax1.legend(ncol=1)
ax2.legend(ncol=1)

plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/Throughput-average.pdf", bbox_inches='tight')
plt.close()
