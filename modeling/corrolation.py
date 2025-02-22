import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Load the dataset
file_path ="/home/jamalids/Documents/merged22.csv"
df = pd.read_csv(file_path)

# Define the k values to be plotted
k_values = range(0,5)  # k1 to k5

# Create subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

for i, k in enumerate(k_values):
    col_x = f"tdt_k{k}"
    col_y = "decomposed zstd row-order ratio"

    if col_x in df.columns and col_y in df.columns:
        df_subset = df[[col_x, col_y]].dropna()

        x = df_subset[col_x].values
        y = df_subset[col_y].values

        # Compute Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)

        # Create scatter plot with regression line
        sns.regplot(x=x, y=y, scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ci=None, ax=axes[i])
        axes[i].set_xlabel(col_x)
        axes[i].set_title(f"k={k}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")

axes[0].set_ylabel("Decomposed Zstd Row-Order Ratio")
plt.tight_layout()


plt.savefig("/home/jamalids/Documents/corr22_subplots.png")
plt.close()
