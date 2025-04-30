# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load CSV file (update path as needed)
# csv_file = "/mnt/c/Users/jamalids/Downloads/figs/results/joined_compression_stats1.csv"   # or use the full path if necessary
# df = pd.read_csv(csv_file)
#
# # (Optional) Print the columns to check
# print("Columns in CSV:", df.columns.tolist())
#
# # Rename the columns so that the legend displays the desired names.
# # Here we assume:
# #   - "raw_zstd_ratio" will be plotted as "TDT_zstd"
# #   - "bitshuffle_zstd_ratio" as "bitshuffle_zstd"
# #   - "raw_zstd_byteshuffle_ratio" as "byteshuffle_zstd"
# df = df.rename(columns={
#
#     "bitshuffle_zstd_ratio": "bitshuffle_zstd",
#     "blosc_zstd_byteshuffle_ratio": "byteshuffle_zstd",
#     "raw_zstd_ratio_y":"standard_zstd"
#
# })
#
# # Optionally, sort by dataset name to maintain a consistent ordering on the x-axis.
# df = df.sort_values("DatasetName")
#
# # Create the plot.
# plt.figure(figsize=(12, 6))
# plt.plot(df["DatasetName"], df["TDT_zstd"], marker='o', label="TDT_zstd")
# plt.plot(df["DatasetName"], df["bitshuffle_zstd"], marker='s', label="bitshuffle_zstd")
# #plt.plot(df["DatasetName"], df["byteshuffle_zstd"], marker='^', label="byteshuffle_zstd")
# # --- Added lines for raw_zstd_ratio and standard_zstd ---
# #plt.plot(df["DatasetName"], df["raw_zstd_ratio"], marker='D', label="raw_zstd_ratio")
# plt.plot(df["DatasetName"], df["standard_zstd"], marker='x', label="standard_zstd")
# plt.yscale("log")
# plt.xlabel("DatasetName")
# plt.ylabel("log(Compression Ratio)")
# plt.title("Compression Ratio by Dataset for ZSTD Variants")
# plt.xticks(rotation=45, ha="right")
# plt.yscale("log")  # if you also want a logarithmic y-axis, otherwise remove this line
# plt.legend()
# plt.tight_layout()
# plt.savefig("/mnt/c/Users/jamalids/Downloads/transformation-plot3.png")
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── VLDB style ──
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

# ── Load main compression data ──
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/joined_compression_stats1.csv")

# ── Load dataset ID mapping ──
df_entropy = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")

# Normalize DatasetName (remove _f32, _f64, etc. if present)
df["DatasetName"] = df["DatasetName"].str.replace(r"_f\d+$", "", regex=True)
df_entropy["DatasetName"] = df_entropy["DatasetName"].str.replace(r"_f\d+$", "", regex=True)

# Merge on DatasetName
df = df.merge(df_entropy[["DatasetName", "DatasetID"]], on="DatasetName", how="inner")

# Ensure DatasetID is a string for plotting
df["DatasetID"] = df["DatasetID"].astype(str)

# Extract numeric sort key from ID (e.g., D12 → 12)
df["DatasetID_SortKey"] = df["DatasetID"].str.extract(r'D(\d+)').astype(int)
df = df.sort_values(by="DatasetID_SortKey")

# Rename columns for plotting
df = df.rename(columns={
    "TDT_zstd": "TDT_zstd",
    "bitshuffle_zstd_ratio": "bitshuffle_zstd",
    "blosc_zstd_byteshuffle_ratio": "byteshuffle_zstd",  # optional if you include it
    "raw_zstd_ratio_y": "standard_zstd"
})

# ── Plot ──
plt.figure(figsize=(6.8, 3))
plt.plot(df["DatasetID"], df["TDT_zstd"], marker='o', label="TDT+Zstd")
plt.plot(df["DatasetID"], df["bitshuffle_zstd"], marker='s', label="Bitshuffle+Zstd")
plt.plot(df["DatasetID"], df["standard_zstd"], marker='x', label="Zstd")

plt.yscale("log")
plt.ylabel("Compression Ratio (log scale)", labelpad=4)
#plt.xlabel("Dataset ID", labelpad=4)
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(pad=0.5)
plt.legend(loc='upper right')

# ── Save plot ──
plt.savefig("/mnt/c/Users/jamalids/Downloads/transformation-plot-vldb.pdf", bbox_inches="tight")
plt.savefig("/mnt/c/Users/jamalids/Downloads/transformation-plot-vldb.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ VLDB-style plot saved with DatasetID on x-axis.")
