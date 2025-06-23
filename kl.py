import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Load and prepare data (stored in float16 for memory efficiency)
df = pd.read_csv("layers.9.attention.wo.weight.tsv", sep='\t', header=None)
weights = df.values.astype(np.float16) / 10000.0
N = weights.shape[0]
block_sizes = [64]  # You can add more sizes here

# Normalize helper (compute in float32, clip to avoid log(0))
def normalize_block(block):
    flat = block.flatten().astype(np.float32)
    flat -= flat.min()
    total = flat.sum()
    dist = flat / total if total > 0 else np.ones_like(flat) / len(flat)
    return np.clip(dist, 1e-10, 1.0)

for M in block_sizes:
    blocks = []

    # Partition into M×M blocks
    for i in range(0, N, M):
        for j in range(0, N, M):
            block = weights[i:i+M, j:j+M]
            blocks.append(normalize_block(block))

    B = len(blocks)
    kl_matrix = np.zeros((B, B), dtype=np.float16)  # final matrix in float16

    # Compute KL divergence using float32 math
    for i in range(B):
        for j in range(B):
            p = blocks[i].astype(np.float32)
            q = blocks[j].astype(np.float32)
            kl_val = entropy(p, q)        # safe in float32
            kl_matrix[i, j] = np.float16(kl_val)  # store in float16

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(kl_matrix.astype(np.float32), cmap="magma")
    plt.title(f"KL Divergence Between Blocks (Block Size = {M})")
    plt.tight_layout()
    plt.savefig(f"kl_heatmap_M{M}_float16.png")
    plt.show()

    # Histogram of off-diagonal values
    kl_vals = kl_matrix[np.triu_indices(B, k=1)].astype(np.float32)
    kl_vals = kl_vals[np.isfinite(kl_vals)]
    
    plt.figure(figsize=(8, 6))
    plt.hist(kl_vals, bins=50, color="darkcyan", edgecolor="black")
    plt.title(f"Histogram of KL Divergence (Block Size = {M})")
    plt.tight_layout()
    plt.savefig(f"kl_histogram_M{M}_float16.png")
    plt.show()
