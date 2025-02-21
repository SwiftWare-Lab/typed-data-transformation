import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend (for saving plots)

# Load the merged CSV file
file_path = "/home/jamalids/Documents/merged.csv"
df = pd.read_csv(file_path)

# Extract dataset names and add decomposition
dataset_names = df["dataset_name"] + " (" + df["decomposition"] + ")"  # Append decomposition method

# Find the highest ratio for each dataset across all ks
num_k = 5  # Assuming k ranges from 0 to 4 (modify if needed)
ratios = []
best_ks = []  # Store the best k for each dataset

for i in range(len(df)):
    avg_orig = np.array([df.loc[i, f"orig_k{k}"] for k in range(num_k)])
    avg_tdt = np.array([df.loc[i, f"tdt_k{k}"] for k in range(num_k)])
    avg_custom = np.array([df.loc[i, f"custom_k{k}"] for k in range(num_k)])
    avg_after = np.array([df.loc[i, f"tdt_after_custom_k{k}"] for k in range(num_k)])

    ratio_tdt = avg_orig / avg_tdt
    ratio_custom = avg_orig / avg_custom
    ratio_after = avg_orig / avg_after

    # Find max ratios and corresponding k values
    max_ratio_tdt, best_k_tdt = np.max(ratio_tdt), np.argmax(ratio_tdt)
    max_ratio_custom, best_k_custom = np.max(ratio_custom), np.argmax(ratio_custom)
    max_ratio_after, best_k_after = np.max(ratio_after), np.argmax(ratio_after)

    ratios.append([max_ratio_tdt, max_ratio_custom, max_ratio_after])
    best_ks.append([best_k_tdt, best_k_custom, best_k_after])

ratios = np.array(ratios)

# Create bar plots
fig, ax1 = plt.subplots(figsize=(14, 6))

bar_width = 0.25
x = np.arange(len(dataset_names))

# Plot bars for max ratios
bars1 = ax1.bar(x - bar_width, ratios[:, 0], width=bar_width, label="orig_k / standard_tdt_k", color='b')
bars2 = ax1.bar(x, ratios[:, 1], width=bar_width, label="orig_k / custom_tdt_k", color='g')
bars3 = ax1.bar(x + bar_width, ratios[:, 2], width=bar_width, label="orig_k / tdt_after_custom_k", color='r')

ax1.set_xlabel("Dataset Name (with decomposition)")
ax1.set_ylabel("Highest Ratio (Log Scale)")
ax1.set_title("Maximum Ratio Comparison for Each Dataset")
ax1.set_xticks(x)
ax1.set_xticklabels(dataset_names, rotation=90, fontsize=9)
ax1.set_yscale("log")  # Set log scale on y-axis
ax1.grid(axis='y')

# Annotate best k values above bars
for i, (bar, best_k) in enumerate(zip([bars1, bars2, bars3], np.array(best_ks).T)):
    for rect, k in zip(bar, best_k):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, height * 1.1, f"k={k}", ha='center', fontsize=8, color="black")

# Second y-axis for Zstd ratios (Log Scale)
ax2 = ax1.twinx()
zstd_ratios = df[["standard zstd ratio", "decomposed zstd row-order ratio", "reordered zstd row-order ratio"]].values

ax2.plot(x, zstd_ratios[:, 0], marker='o', linestyle='-', color='c', label="Standard Zstd Ratio")
ax2.plot(x, zstd_ratios[:, 1], marker='s', linestyle='-', color='m', label="Decomposed Zstd row-Order Ratio")
ax2.plot(x, zstd_ratios[:, 2], marker='^', linestyle='-', color='y', label="Reordered Zstd row-Order Ratio")

ax2.set_ylabel("Zstd Ratios (Log Scale)")
ax2.set_yscale("log")  # **Apply log scale to the second y-axis as well**

# Create separate legends to avoid overlap
ax1.legend(loc="upper right", bbox_to_anchor=(1, 1))  # Move bar plot legend
ax2.legend(loc="upper left", bbox_to_anchor=(0, 1))   # Move Zstd ratio legend

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy-row.png")
print("Saved plot to /home/jamalids/Documents/entropy.png")
