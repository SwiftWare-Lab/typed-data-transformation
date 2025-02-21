import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving plots

# Load the merged CSV file
file_path = "/home/jamalids/Documents/merged.csv"
df = pd.read_csv(file_path)

# Extract dataset names
dataset_names = df["dataset_name"]

# Number of k values
num_k = 5  # Assuming k ranges from 0 to 4

# Store highest entropy ratio for each dataset based on "orig_k / custom_tdt_k"
highest_entropy_ratios = []
best_ks = []

for i in range(len(df)):
    avg_orig = np.array([df.loc[i, f"orig_k{k}"] for k in range(num_k)])
    avg_custom = np.array([df.loc[i, f"custom_k{k}"] for k in range(num_k)])

    ratio_custom = avg_orig / avg_custom
    max_ratio_custom = np.max(ratio_custom)
    best_k_custom = np.argmax(ratio_custom)

    highest_entropy_ratios.append(max_ratio_custom)
    best_ks.append(best_k_custom)

highest_entropy_ratios = np.array(highest_entropy_ratios)

# Extract Zstd compression ratios
zstd_standard = df["standard zstd ratio"].values
zstd_decomposed = df["decomposed zstd row-order ratio"].values

# Compute Pearson and Spearman correlations per dataset
pearson_corr_standard = []
spearman_corr_standard = []
pearson_corr_decomposed = []
spearman_corr_decomposed = []

for i in range(len(df)):
    # Check if the dataset has variation (to avoid constant input errors)
    if len(set([highest_entropy_ratios[i], zstd_standard[i]])) > 1:
        pearson_corr_standard.append(pearsonr([highest_entropy_ratios[i]], [zstd_standard[i]])[0])
        spearman_corr_standard.append(spearmanr([highest_entropy_ratios[i]], [zstd_standard[i]])[0])
    else:
        pearson_corr_standard.append(np.nan)
        spearman_corr_standard.append(np.nan)

    if len(set([highest_entropy_ratios[i], zstd_decomposed[i]])) > 1:
        pearson_corr_decomposed.append(pearsonr([highest_entropy_ratios[i]], [zstd_decomposed[i]])[0])
        spearman_corr_decomposed.append(spearmanr([highest_entropy_ratios[i]], [zstd_decomposed[i]])[0])
    else:
        pearson_corr_decomposed.append(np.nan)
        spearman_corr_decomposed.append(np.nan)

# Convert lists to numpy arrays
pearson_corr_standard = np.array(pearson_corr_standard)
spearman_corr_standard = np.array(spearman_corr_standard)
pearson_corr_decomposed = np.array(pearson_corr_decomposed)
spearman_corr_decomposed = np.array(spearman_corr_decomposed)

# Create bar plot for each dataset separately
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(dataset_names))

bar_width = 0.2
ax.bar(x - bar_width, pearson_corr_standard, width=bar_width, label="Pearson - Standard Zstd", color='blue')
ax.bar(x, spearman_corr_standard, width=bar_width, label="Spearman - Standard Zstd", color='cyan')
ax.bar(x + bar_width, pearson_corr_decomposed, width=bar_width, label="Pearson - Decomposed Zstd", color='red')
ax.bar(x + 2*bar_width, spearman_corr_decomposed, width=bar_width, label="Spearman - Decomposed Zstd", color='orange')

ax.set_xlabel("Dataset Name")
ax.set_ylabel("Correlation Coefficients")
ax.set_title("Per-Dataset Correlation: Highest Entropy Ratio vs. Zstd Compression")
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=90, fontsize=8)
ax.legend()
ax.grid(axis="y")

plt.tight_layout()
plt.savefig("/home/jamalids/per_dataset_correlation.png")
print("Saved correlation plot to /mnt/data/per_dataset_correlation.png")
