import pandas as pd
from scipy.stats import gmean

# Load the CSV file
file_path = "/mnt/c/Users/jamalids/Downloads/figs/results/joined_compression_stats1.csv"  # Update this to your actual path
df = pd.read_csv(file_path)

# Compute the Compression Ratio Improvement (CRI)
df["CRI"] = df["TDT_zstd"] / df["bitshuffle_zstd_ratio"]

# Drop any rows with missing CRI values
valid_cri = df["CRI"].dropna()

# Compute geometric mean
cri_gmean = gmean(valid_cri)

# Print result
print(f"Geometric Mean CRI (TDT_zstd / bitshuffle_zstd_ratio): {cri_gmean:.3f}")
