import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and rescale
df = pd.read_csv("layers.9.attention.wo.weight.tsv", sep='\t', header=None)
original_weights = df.values / 10000.0  # Now in correct decimal scale

# Show two sample values before and after scaling
print("Original (scaled) float64:", original_weights[0, 0], original_weights[1, 1])

# Convert to float16
weights = original_weights.astype(np.float16)

# Show float16 versions to compare precision
print("Converted float16:", weights[0, 0], weights[1, 1])

# For stats, convert to float32 to avoid overflow
weights_stats = weights.astype(np.float32)

# Print stats
print("Shape:", weights.shape)
print("Min:", np.min(weights))
print("Max:", np.max(weights))
print("Mean:", np.mean(weights_stats))
print("Std Dev:", np.std(weights_stats))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(weights, cmap="viridis", cbar=True)
plt.title("Heatmap of Rescaled Attention Weights")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.savefig("layers9_scaled_heatmap.png")
plt.show()

# Histogram graph
plt.figure(figsize=(8, 6))
plt.hist(weights.flatten(), bins=100, color="steelblue", edgecolor="black")
plt.title("Distribution of Rescaled Weight Values")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("layers9_scaled_histogram.png")
plt.show()
