import pandas as pd
from scipy.stats import gmean

# Load all CSVs
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
    'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
}

dfs = []
for tool, path in files.items():
    df = pd.read_csv(path)
    df['compression_tool'] = tool
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Normalize RunType and DatasetName
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

# Merge with metadata
entropy_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping1.csv")
df = pd.merge(combined_df, entropy_df, on='DatasetName', how='inner')

# Keep only TDT & standard rows
df = df[df['RunType'].isin(['TDT', 'standard'])]

# Map suffix to precision name
suffix_label_map = {'16': 'half', '32': 'single', '64': 'double'}
improvement_results = []

# Loop through precisions
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
        grouped['GMean_CRI'] = grouped['TDT'] / grouped['standard']
        for tool, row in grouped.iterrows():
            improvement_results.append({
                "Precision": suffix_label_map[suffix],
                "CompressionTool": tool,
                "GMean_CRI": row['GMean_CRI']
            })

# Display results
results_df = pd.DataFrame(improvement_results)
print(results_df.to_string(index=False))
###############################################################
import pandas as pd
from scipy.stats import gmean

# ────────── 1. Load CSVs ──────────
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
    'nvCOMP': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/nvcomp.csv'
}

dfs = []
for tool, path in files.items():
    df = pd.read_csv(path)
    df['compression_tool'] = tool
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# ────────── 2. Normalize RunType and DatasetName ──────────
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

# ────────── 3. Merge with Metadata ──────────
entropy_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping1.csv")
df = pd.merge(combined_df, entropy_df, on='DatasetName', how='inner')

# Keep only TDT & standard
df = df[df['RunType'].isin(['TDT', 'standard'])]

# ────────── 4. Compute Geometric Mean CRI by Precision ──────────
suffix_label_map = {'16': 'half', '32': 'single', '64': 'double'}
cri_results = []

for suffix in ['32', '64', '16']:
    df_suffix = df[df['DatasetName'].str.endswith(suffix)]
    if df_suffix.empty:
        continue

    grouped = (
        df_suffix.groupby(['DatasetName', 'compression_tool', 'RunType'])['CompressionRatio']
        .first().unstack().dropna()
    )

    if {'TDT', 'standard'}.issubset(grouped.columns):
        grouped['CRI'] = grouped['TDT'] / grouped['standard']
        gm_cri = gmean(grouped['CRI'])
        cri_results.append({
            "Precision": suffix_label_map[suffix],
            "GeometricMeanCRI": round(gm_cri, 4)
        })

# ────────── 5. Print Results ──────────
print("✅ Geometric Mean CRI (TDT / standard) by Precision:")
for r in cri_results:
    print(f"• {r['Precision'].capitalize()}-precision: {r['GeometricMeanCRI']}")
