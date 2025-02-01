import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.ticker import FuncFormatter
# Use a non-interactive backend if needed.
matplotlib.use("Agg")

# ------------------------------
# Load and Prepare Data
# ------------------------------
file_path = '/home/jamalids/Documents/num_brain_f64.csv'
df = pd.read_csv(file_path, sep=";", engine="python")

# Exclude rows where BlockSize is "N/A" and where RunType is "Full"
df_numeric = df[(df['BlockSize'] != "N/A") & (df['RunType'] != "Full")].copy()

# Ensure BlockSize is treated as a string (we want categorical labels)
df_numeric['BlockSize'] = df_numeric['BlockSize'].astype(str)

# Identify all columns that contain "BlockCompRatio_"
block_comp_ratio_cols = df_numeric.filter(like="BlockCompRatio_").columns

# Compute the mean BlockCompRatio per row (averaging across all BlockCompRatio columns)
df_numeric['MeanBlockCompRatio'] = df_numeric[block_comp_ratio_cols].mean(axis=1)

# ------------------------------
# Group by RunType and BlockSize
# ------------------------------
# Group the data by RunType and BlockSize and compute the mean of MeanBlockCompRatio
grouped = df_numeric.groupby(['RunType', 'BlockSize'])['MeanBlockCompRatio'].mean().reset_index()

# ------------------------------
# Sort BlockSize as Categories by Their Numeric Value
# ------------------------------
# Assume BlockSize values are of the form "20K", "40K", etc.
def extract_numeric(bs_str):
    # Extract numeric characters and convert to int.
    return int(''.join(filter(str.isdigit, bs_str)))

# For each run type, we will sort the data by the numeric part of BlockSize.
run_types = grouped['RunType'].unique()

# ------------------------------
# Plotting
# ------------------------------
plt.figure(figsize=(10, 6))
for run in run_types:
    subdf = grouped[grouped['RunType'] == run].copy()
    # Create a new column with the numeric part of BlockSize for sorting.
    subdf['numeric_block'] = subdf['BlockSize'].apply(extract_numeric)
    subdf = subdf.sort_values(by='numeric_block')
    # Plot using the BlockSize column as categorical labels.
    plt.plot(subdf['BlockSize'], subdf['MeanBlockCompRatio'],
             marker='o', linestyle='-', label=run)

plt.xlabel("Block Size")
plt.ylabel("Mean Block Compression Ratio")
plt.title("Mean Block Compression Ratio vs Block Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_block_comp_ratio_vs_blocksize.png")
plt.close()

# ------------------------------
# Load the CSV file
# ------------------------------

# Exclude rows where BlockSize is "N/A" (non-blocking experiments) and drop any missing values.
df_numeric = df[df['BlockSize'] != "N/A"].copy()
df_numeric = df_numeric.dropna(subset=['BlockSize', 'TotalTimeCompressed', 'TotalTimeDecompressed'])

# Convert BlockSize and total times to numeric values.
df_numeric['BlockSize'] = pd.to_numeric(df_numeric['BlockSize'], errors='coerce')
df_numeric['TotalTimeCompressed'] = pd.to_numeric(df_numeric['TotalTimeCompressed'], errors='coerce')
df_numeric['TotalTimeDecompressed'] = pd.to_numeric(df_numeric['TotalTimeDecompressed'], errors='coerce')

# Drop any rows that might have become NaN after conversion.
df_numeric = df_numeric.dropna(subset=['BlockSize', 'TotalTimeCompressed', 'TotalTimeDecompressed'])

# ------------------------------
# Filter by RunType (exclude "Full")
# ------------------------------
run_types = [rt for rt in df_numeric['RunType'].unique() if rt != "Full"]
df_numeric = df_numeric[df_numeric['RunType'].isin(run_types)]

# ------------------------------
# Create a categorical mapping for BlockSize.
# ------------------------------
# Extract unique block sizes (in bytes) from the CSV file.
unique_block_sizes = sorted(df_numeric['BlockSize'].unique())
# Create string labels for the x-axis.
categories = [str(int(x)) for x in unique_block_sizes]
# Create a mapping from block size to an index.
mapping = {size: i for i, size in enumerate(unique_block_sizes)}

# ------------------------------
# Plot: Compression Time vs Block Size (categorical x-axis)
# ------------------------------
plt.figure(figsize=(10, 6))
for run in run_types:
    df_run = df_numeric[df_numeric['RunType'] == run]
    # Map the numeric BlockSize values to categorical indices.
    x = df_run['BlockSize'].apply(lambda s: mapping[s]).values
    y = df_run['TotalTimeCompressed'].values
    plt.plot(x, y, marker='o', linestyle='-', label=run)

plt.xlabel('Block Size (bytes)')
plt.ylabel('Compression Time (seconds)')
plt.title('Compression Time vs Block Size')
plt.legend()
plt.grid(True)
# Set the x-axis ticks to be at the categorical positions and label them with the original block sizes.
plt.xticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig('compression_time_vs_block_size.png')
plt.show()

# ------------------------------
# Plot: Decompression Time vs Block Size (categorical x-axis)
# ------------------------------
plt.figure(figsize=(10, 6))
for run in run_types:
    df_run = df_numeric[df_numeric['RunType'] == run]
    x = df_run['BlockSize'].apply(lambda s: mapping[s]).values
    y = df_run['TotalTimeDecompressed'].values
    plt.plot(x, y, marker='o', linestyle='-', label=run)

plt.xlabel('Block Size (bytes)')
plt.ylabel('Decompression Time (seconds)')
plt.title('Decompression Time vs Block Size')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(len(categories)), categories)
plt.tight_layout()
plt.savefig('decompression_time_vs_block_size.png')
plt.show()