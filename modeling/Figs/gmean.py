import pandas as pd
from scipy.stats import gmean

# === Load uploaded CSV ===
file_path = "/mnt/c/Users/jamalids/Downloads/figs/results/fig7/zstd.csv"
df = pd.read_csv(file_path)
bad_datasets = [
    "ts_gas_f32",
    "phone_gyro_f64",
    "nyc_taxi2015_f64",
    "jane_street_f64",
    "gas_price_f64",
    "tpch_lineitem_f32",
]

df = df[~df["DatasetName"].isin(bad_datasets)].copy()
# === Normalize RunType values ===
df["RunType"] = df["RunType"].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
   # "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Full":                      "standard",
    "Whole":                     "standard",
    "Chunked_parallel":          "standard"
})

# === Filter for TDT and standard only ===
df_filtered = df[df["RunType"].isin(["TDT", "standard"])]

# === Compute GMean Throughput (Compression & Decompression) ===
gmean_tp = (
    df_filtered.groupby("RunType")[["CompressionThroughput", "DecompressionThroughput"]]
    .agg(lambda x: gmean(x[x > 0]))  # avoid 0 for log
    .reset_index()
)

# === Compute GMean Compression Ratio ===
gmean_cr = (
    df_filtered.groupby("RunType")[["CompressionRatio"]]
    .agg(lambda x: gmean(x[x > 0]))
    .reset_index()
)

# === Print Results ===
print("🔍 Geometric Mean Throughput:")
print(gmean_tp.to_string(index=False))

print("\n🔍 Geometric Mean Compression Ratio:")
print(gmean_cr.to_string(index=False))
#############################################################
# === Normalize RunType ===
df["RunType"] = df["RunType"].replace({
    "Chunked_Decompose_Parallel": "TDT",
    "Chunk-decompose_Parallel":  "TDT",
    "Decompose_Chunk_Parallel":  "TDT",
    "Component":                 "TDT",
    "Full":                      "standard",
    "Whole":                     "standard",
    "Chunked_parallel":          "standard"
})

# === Filter for TDT and standard ===
df_filtered = df[df["RunType"].isin(["TDT", "standard"])]

# === Compute GMeans ===
gmean_stats = (
    df_filtered.groupby("RunType")[["CompressionThroughput", "DecompressionThroughput", "CompressionRatio"]]
    .agg(lambda x: gmean(x[x > 0]))
    .reset_index()
)

# === Extract for CRI calculation ===
tdt_row = gmean_stats[gmean_stats["RunType"] == "TDT"].iloc[0]
std_row = gmean_stats[gmean_stats["RunType"] == "standard"].iloc[0]

cri_ct = tdt_row["CompressionThroughput"] / std_row["CompressionThroughput"]
cri_dt = tdt_row["DecompressionThroughput"] / std_row["DecompressionThroughput"]
cri_cr = (tdt_row["CompressionRatio"]-std_row["CompressionRatio"] )/ std_row["CompressionRatio"]

# === Print Results ===
print("🔍 Geometric Mean Results:")
print(gmean_stats.to_string(index=False))

print(f"\n✅ CRI (TDT / Standard):")
print(f"Compression Ratio     : {cri_cr:.2f}×")
print(f"Compression Throughput: {cri_ct:.2f}×")
print(f"Decompression Throughput: {cri_dt:.2f}×")