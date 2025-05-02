import pandas as pd
from scipy.stats import gmean

# ---------------------------------------------------------------
# 1.  READ + MERGE
# ---------------------------------------------------------------
files = {
    'lz4':    '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/lz4.csv',
    'snappy': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/snappy.csv',
    'zlib':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zlib.csv',
    'zstd':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv',
    'bzip':   '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/bzip.csv',
   # 'FastLZ': '/mnt/c/Users/jamalids/Downloads/figs/results/fig7/fastlz.csv',
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
# Load combined dataset
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")

# Filter for TDT and standard run types
df_filtered = df[df["RunType"].isin(["TDT", "standard"])]

# Group by compression_tool and RunType for compression and decompression throughput
grouped_ct = df_filtered.groupby(['compression_tool', 'RunType'])['CompressionThroughput'].mean().unstack()
grouped_dt = df_filtered.groupby(['compression_tool', 'RunType'])['DecompressionThroughput'].mean().unstack()

# Calculate CRI (TDT / standard) for compression and decompression throughput
grouped_ct['CRI_CT'] = grouped_ct['TDT'] / grouped_ct['standard']
grouped_dt['CRI_DT'] = grouped_dt['TDT'] / grouped_dt['standard']

# Merge both CRI results
cri_df = pd.merge(
    grouped_ct[['CRI_CT']],
    grouped_dt[['CRI_DT']],
    left_index=True,
    right_index=True
).reset_index()

# Print results
print("✅ Compression and Decompression CRI per tool:\n")
print(cri_df.to_string(index=False))
#####################################################################
from scipy.stats import gmean

# Step 1: Filter only TDT and standard
df_filtered = combined_df[combined_df["RunType"].isin(["TDT", "standard"])]

# Step 2: Compute geometric mean throughput
gmean_tp = (
    df_filtered.groupby(["compression_tool", "RunType"])[["CompressionThroughput", "DecompressionThroughput"]]
    .agg(lambda x: gmean(x[x > 0]))  # exclude zero values to avoid math domain errors
    .reset_index()
)

# Step 3: Pivot to make CRI computation easier
pivoted = gmean_tp.pivot(index="compression_tool", columns="RunType", values=["CompressionThroughput", "DecompressionThroughput"])

# Step 4: Compute CRI (TDT / standard)
pivoted["CRI_CT"] = pivoted["CompressionThroughput"]["TDT"] / pivoted["CompressionThroughput"]["standard"]
pivoted["CRI_DT"] = pivoted["DecompressionThroughput"]["TDT"] / pivoted["DecompressionThroughput"]["standard"]

# Step 5: Flatten column index and reset
pivoted.columns.name = None
pivoted = pivoted.reset_index()

# Step 6: Print result
print("✅ Geometric Mean Throughput + CRI per Tool:")
print(pivoted[["compression_tool", "CompressionThroughput", "DecompressionThroughput", "CRI_CT", "CRI_DT"]].to_string(index=False))
min_cri_ct = pivoted["CRI_CT"].min()
max_cri_ct = pivoted["CRI_CT"].max()
min_cri_dt = pivoted["CRI_DT"].min()
max_cri_dt = pivoted["CRI_DT"].max()

print(f"\n✅ Compression Throughput CRI range: {min_cri_ct:.2f}× – {max_cri_ct:.2f}×")
print(f"✅ Decompression Throughput CRI range: {min_cri_dt:.2f}× – {max_cri_dt:.2f}×")
########################
# Compute global gmean compression & decompression throughput for TDT
tdt_df = df_filtered[df_filtered["RunType"] == "TDT"]

gmean_compression = gmean(tdt_df["CompressionThroughput"] )
gmean_decompression = gmean(tdt_df["DecompressionThroughput"])

print(f"\n✅ Global GMean Compression Throughput (TDT):    {gmean_compression:.2f}")
print(f"✅ Global GMean Decompression Throughput (TDT):  {gmean_decompression:.2f}")
# Compute GMean compression & decompression throughput per tool for TDT only
tdt_only = df_filtered[df_filtered["RunType"] == "TDT"]

gmean_per_tool = (
    tdt_only.groupby("compression_tool")[["CompressionThroughput", "DecompressionThroughput"]]
    .agg(lambda x: gmean(x[x > 0]))  # Avoid log(0)
    .reset_index()
)

print("\n✅ GMean Compression and Decompression Throughput for TDT per Tool:")
print(gmean_per_tool.to_string(index=False))

####################################################################
import pandas as pd
from scipy.stats import gmean

# Load and normalize data
combined_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combined_df.csv")
bad_datasets = [
    "ts_gas_f32",
    "phone_gyro_f64",
    "nyc_taxi2015_f64",
    "jane_street_f64",
    "gas_price_f64",
    "tpch_lineitem_f32",
]

combined_df  = combined_df [~combined_df ["DatasetName"].isin(bad_datasets)].copy()
combined_df['RunType'] = combined_df['RunType'].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel": "TDT",
    "Decompose_Chunk_Parallel": "TDT",
    "Component": "TDT",
    "Whole": "standard",
    "Full": "standard",
    "Chunked_parallel": "standard"
})

df_filtered = combined_df[combined_df["RunType"].isin(["TDT", "standard"])].copy()

# ─────────────────────────────────────────────
# Step 1: Compute gmean per tool and runtype
# ─────────────────────────────────────────────
gmean_tp = (
    df_filtered.groupby(["compression_tool", "RunType"])[["CompressionThroughput", "DecompressionThroughput"]]
    .agg(lambda x: gmean(x[x > 0]))  # Avoid math domain errors
    .reset_index()
)

# Pivot to compare
pivoted = gmean_tp.pivot(index="compression_tool", columns="RunType", values=["CompressionThroughput", "DecompressionThroughput"])

# Calculate CRI (TDT / standard)
pivoted["CRI_CT"] = pivoted["CompressionThroughput"]["TDT"] / pivoted["CompressionThroughput"]["standard"]
pivoted["CRI_DT"] = pivoted["DecompressionThroughput"]["TDT"] / pivoted["DecompressionThroughput"]["standard"]

# Flatten columns
pivoted.columns.name = None
pivoted = pivoted.reset_index()

# Print CRI range
min_cri_ct = pivoted["CRI_CT"].min()
max_cri_ct = pivoted["CRI_CT"].max()
min_cri_dt = pivoted["CRI_DT"].min()
max_cri_dt = pivoted["CRI_DT"].max()

print("✅ Geometric Mean Throughput (TDT vs Standard) + CRI:")
print(pivoted[["compression_tool", "CompressionThroughput", "DecompressionThroughput", "CRI_CT", "CRI_DT"]].to_string(index=False))
print(f"\n✅ Compression Throughput CRI range: {min_cri_ct:.2f}× – {max_cri_ct:.2f}×")
print(f"✅ Decompression Throughput CRI range: {min_cri_dt:.2f}× – {max_cri_dt:.2f}×")

# ─────────────────────────────────────────────
# Step 2: Global GMean for TDT only
# ─────────────────────────────────────────────
tdt_df = df_filtered[df_filtered["RunType"] == "TDT"]
gmean_compression = gmean(tdt_df["CompressionThroughput"][tdt_df["CompressionThroughput"] > 0])
gmean_decompression = gmean(tdt_df["DecompressionThroughput"][tdt_df["DecompressionThroughput"] > 0])

print(f"\n✅ Global GMean Compression Throughput (TDT):    {gmean_compression:.2f}")
print(f"✅ Global GMean Decompression Throughput (TDT):  {gmean_decompression:.2f}")

# ─────────────────────────────────────────────
# Step 3: Per-tool GMean for TDT only
# ─────────────────────────────────────────────
gmean_per_tool = (
    tdt_df.groupby("compression_tool")[["CompressionThroughput", "DecompressionThroughput"]]
    .agg(lambda x: gmean(x[x > 0]))
    .reset_index()
)

print("\n✅ GMean Compression and Decompression Throughput for TDT per Tool1111:")
print(gmean_per_tool.to_string(index=False))
# ─────────────────────────────────────────────
# Step 4: Global GMean Compression Ratio (TDT vs Standard)
# ─────────────────────────────────────────────
gmean_cr = (
    df_filtered.groupby("RunType")[["CompressionRatio"]]
    .agg(lambda x: gmean(x[x > 0]))
    .reset_index()
)

# Calculate CRI (Compression Ratio Improvement in %)
tdt_cr = gmean_cr[gmean_cr["RunType"] == "TDT"]["CompressionRatio"].values[0]
std_cr = gmean_cr[gmean_cr["RunType"] == "standard"]["CompressionRatio"].values[0]
cri_cr = (tdt_cr - std_cr) / std_cr

# Print
print("\n✅ Global GMean Compression Ratio (TDT vs Standard):")
print(gmean_cr.to_string(index=False))
print(f"\n✅ Compression Ratio Improvement (CRI): {cri_cr * 100:.2f}%")
