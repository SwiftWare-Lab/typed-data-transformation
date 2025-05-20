import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded dataset
df = pd.read_csv("/home/jamalids/Documents/frame/new3/big-data-compression/modeling/cross-entropy/synthetic_all_modes.csv")

# Filter only FeatureMode == "entropy"
df_entropy = df[df["FeatureMode"] == "entropy"].copy()

# Ensure cluster config is treated as string for x-axis labeling
df_entropy["ClusterConfig"] = df_entropy["ClusterConfig"].astype(str)

# Prepare figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

x = df_entropy["ClusterConfig"]

# ---------------- First subplot ----------------
ax1.plot(x, df_entropy["HC_H1_weighted"], label="HC_H1_weighted", marker="o")
ax1.plot(x, df_entropy["HC_H2_weighted"], label="HC_H2_weighted", marker="s")
ax1.set_ylabel("Weighted Entropy")
ax1.legend(loc="upper left")
ax1.set_xticks(range(len(x)))
ax1.set_xticklabels(x, rotation=45, ha='right')

# Twin Y-axis
ax1b = ax1.twinx()
ax1b.plot(x, df_entropy["DecomposedRatio_Row_F"], label="DecomposedRatio_Row_F", color="black", linestyle="--", marker="x")
ax1b.plot(x, df_entropy["ReorderedRatio_Row_F"], label="ReorderedRatio_Row_F", color="grey", linestyle="--", marker="*")
ax1b.set_ylabel("Compression Ratio (Row_F)")
ax1b.legend(loc="upper right")

# ---------------- Second subplot ----------------
ax2.plot(x, df_entropy["MutualInfo"], label="MutualInfo", marker="o")
ax2.plot(x, df_entropy["WithinSTD"], label="WithinSTD", marker="^")
ax2.set_ylabel("Mutual Info / STD")
ax2.legend(loc="upper left")
ax2.set_xlabel("Cluster Configurations")

# Twin Y-axis for subplot 2
ax2b = ax2.twinx()
ax2b.plot(x, df_entropy["DecomposedRatio_Row_F"], label="DecomposedRatio_Row_F", color="black", linestyle="--", marker="x")
ax2b.plot(x, df_entropy["ReorderedRatio_Row_F"], label="ReorderedRatio_Row_F", color="grey", linestyle="--", marker="*")
ax2b.set_ylabel("Compression Ratio (Row_F)")
ax2b.legend(loc="upper right")

plt.tight_layout()
plt.savefig("entropy.png")
