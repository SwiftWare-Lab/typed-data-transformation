import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

# Load and prepare data (stored in float16)
df = pd.read_csv("layers.9.attention.wo.weight.tsv", sep='\t', header=None)
weights = df.values.astype(np.float16) / 10000.0
N = weights.shape[0]

# Compute dynamic range once (max – min of the full matrix)
data_range = float(weights.max() - weights.min())

block_sizes = [64]  # Add more sizes if needed

for M in block_sizes:
    raw_blocks = []

    # Partition into M×M blocks (keep in float16 for memory)
    for i in range(0, N, M):
        for j in range(0, N, M):
            block = weights[i:i+M, j:j+M]
            raw_blocks.append(block)

    B = len(raw_blocks)
    ssim_matrix = np.zeros((B, B), dtype=np.float32)  # store in float32

    # Compute SSIM using float32, passing data_range
    for i in range(B):
        for j in range(B):
            block1 = raw_blocks[i].astype(np.float32)
            block2 = raw_blocks[j].astype(np.float32)

            score, _ = ssim(
                block1,
                block2,
                data_range=data_range,
                full=True
            )
            ssim_matrix[i, j] = score

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(ssim_matrix, cmap="viridis")
    plt.title(f"SSIM Between Blocks (Block Size = {M})")
    plt.tight_layout()
    plt.savefig(f"ssim_heatmap_M{M}_float32.png")
    plt.show()

    # Histogram of off-diagonal SSIM values
    ssim_vals = ssim_matrix[np.triu_indices(B, k=1)]
    ssim_vals = ssim_vals[np.isfinite(ssim_vals)]

    plt.figure(figsize=(8, 6))
    plt.hist(ssim_vals, bins=50, color="mediumseagreen", edgecolor="black")
    plt.title(f"Histogram of SSIM Scores (Block Size = {M})")
    plt.tight_layout()
    plt.savefig(f"ssim_histogram_M{M}_float32.png")
    plt.show()
