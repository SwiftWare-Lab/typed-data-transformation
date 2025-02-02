import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

# Use a non-interactive backend if needed.
matplotlib.use("Agg")

# ------------------------------
# Load and Prepare Data
# ------------------------------
file_path = "/home/jamalids/Documents/jw_mirimage_f32.csv"

# Attempt to reload the CSV with a more flexible approach
df = pd.read_csv(file_path, delimiter=",", engine="python")

# Display the first few rows to check the structure
df.head()

# Print column names to verify they match what you expect.
print("Columns in CSV:", df.columns)

# Exclude rows where BlockSize is "[0]" (or "N/A" if that is how it is marked) and drop any missing values.
df_numeric = df[df["BlockSize"] != "{ [0] }"].copy()
df_numeric = df_numeric.dropna(
    subset=["BlockSize", "TotalTimeCompressed", "TotalTimeDecompressed", "ConfigString", "CompressionRatio"]
)

# Convert BlockSize and relevant columns to numeric values.
df_numeric["BlockSize"] = pd.to_numeric(df_numeric["BlockSize"], errors="coerce")
df_numeric["TotalTimeCompressed"] = pd.to_numeric(df_numeric["TotalTimeCompressed"], errors="coerce")
df_numeric["TotalTimeDecompressed"] = pd.to_numeric(df_numeric["TotalTimeDecompressed"], errors="coerce")
df_numeric["CompressionRatio"] = pd.to_numeric(df_numeric["CompressionRatio"], errors="coerce")

# Drop any rows that might have become NaN after conversion.
df_numeric = df_numeric.dropna(subset=["BlockSize", "TotalTimeCompressed", "TotalTimeDecompressed", "CompressionRatio"])

# ------------------------------
# Do NOT remove "Full" from RunType.
# ------------------------------
run_types = df_numeric["RunType"].unique()

# ------------------------------
# Create a categorical mapping for BlockSize.
# ------------------------------
unique_block_sizes = sorted(df_numeric["BlockSize"].unique())
categories = [str(int(x)) for x in unique_block_sizes]
mapping = {size: i for i, size in enumerate(unique_block_sizes)}

# ------------------------------
# Group data by RunType and ConfigString.
# ------------------------------
df_numeric["RunType1"] = df_numeric["RunType"].astype(str) + " " + df_numeric["ConfigString"].astype(str)
grouped = df_numeric.groupby("RunType1")

# ------------------------------
# Plot: Compression Time vs Block Size (categorical x-axis)
# ------------------------------
plt.figure(figsize=(10, 6))
for label, group in grouped:
    x = group["BlockSize"].apply(lambda s: mapping[s]).values
    y = group["TotalTimeCompressed"].values
    plt.plot(x, y, marker="o", linestyle="-", label=label)

plt.xlabel("Block Size (bytes)")
plt.ylabel("Compression Time (seconds)")
plt.title("Compression Time vs Block Size")
plt.legend()
plt.grid(True)
plt.xticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/compression_time_vs_block_size.png")
plt.show()

# ------------------------------
# Plot: Decompression Time vs Block Size (categorical x-axis)
# ------------------------------
plt.figure(figsize=(10, 6))
for label, group in grouped:
    x = group["BlockSize"].apply(lambda s: mapping[s]).values
    y = group["TotalTimeDecompressed"].values
    plt.plot(x, y, marker="o", linestyle="-", label=label)

plt.xlabel("Block Size (bytes)")
plt.ylabel("Decompression Time (seconds)")
plt.title("Decompression Time vs Block Size")
plt.legend()
plt.grid(True)
plt.xticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/decompression_time_vs_block_size.png")
plt.show()

# ------------------------------
# Plot: Compression Ratio vs Block Size (categorical x-axis)
# ------------------------------
plt.figure(figsize=(10, 6))
for label, group in grouped:
    x = group["BlockSize"].apply(lambda s: mapping[s]).values
    y = group["CompressionRatio"].values
    plt.plot(x, y, marker="o", linestyle="-", label=label)

plt.xlabel("Block Size (bytes)")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Block Size")
plt.legend()
plt.grid(True)
plt.xticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/compression_ratio_vs_block_size.png")
plt.show()
