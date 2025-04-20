import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1 · Load and tidy the data
# --------------------------------------------------
csv_path = "/home/jamalids/Documents/combined_median_rows.csv"
df = pd.read_csv(csv_path)

# Trim accidental whitespace in headers, then give the ratio column a safe name
df.columns = [c.strip() for c in df.columns]
#df = df.rename(columns={"compression Ratio": "CompRatio"})

keep_cols = ["Dataset", "Mode", "CompThroughput", "DecompThroughput", "CompRatio"]
df = df[keep_cols]

# Map internal mode names → friendly labels
mode_label = {"Whole": "Standard", "Component": "TDT"}
modes      = ["Whole", "Component"]
datasets   = df["Dataset"].unique()

# --------------------------------------------------
# 2 · Build THREE vertically stacked sub‑plots
# --------------------------------------------------
fig, (axC, axD, axR) = plt.subplots(
    nrows=3, ncols=1,
    figsize=(14, 10),
    sharex=True,
    constrained_layout=True
)

# ---- A · Compression throughput -------------------
for m in modes:
    sub = df[df["Mode"] == m]
    axC.plot(sub["Dataset"], sub["CompThroughput"],
             marker="o", label=mode_label[m])

axC.set_title("Compression Throughput by Mode")
#axC.set_yscale('log')
axC.set_ylabel("Comp Throughput")
axC.grid(alpha=0.3)
axC.legend(title="Mode", ncol=len(modes))

# ---- B · Decompression throughput -----------------
for m in modes:
    sub = df[df["Mode"] == m]
    axD.plot(sub["Dataset"], sub["DecompThroughput"],
             marker="x", linestyle="--", label=mode_label[m])

axD.set_title("Decompression Throughput by Mode")
#axD.set_yscale('log')
axD.set_ylabel("Decomp Throughput ")
axD.grid(alpha=0.3)
axD.legend(title="Mode", ncol=len(modes))

# ---- C · Compression ratio (NOT throughput) -------
for m in modes:
    sub = df[df["Mode"] == m]
    axR.plot(sub["Dataset"], sub["CompRatio"],
             marker="s", linestyle=":", label=mode_label[m])

axR.set_title("Compression Ratio by Mode")
axR.set_xlabel("Dataset")

axR.set_ylabel("Compression Ratio")
axR.set_xticks(range(len(datasets)))
axR.set_xticklabels(datasets, rotation=45, ha="right")
axR.grid(alpha=0.3)
axR.legend(title="Mode", ncol=len(modes))

# --------------------------------------------------
# 3 · Save
# --------------------------------------------------
out_png = "/home/jamalids/Documents/gpu.png"
plt.savefig(out_png, dpi=300)
print(f"Plot saved to {out_png}")
