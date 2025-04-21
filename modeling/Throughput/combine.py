import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
files = {
   # 'Lz4': '/home/jamalids/Documents/result-throughput/lz4.csv',
    'Snappy': '/home/jamalids/Documents/result-throughput/snappy.csv',
    'Zlib': '/home/jamalids/Documents/result-throughput/zlip.csv',
    'Zstd': '/home/jamalids/Documents/result-throughput/zstd.csv',
    'Bzip': '/home/jamalids/Documents/result-throughput/bzib.csv',
   # 'Lempel-Ziv': '/home/jamalids/Documents/result-throughput/fastlz.csv'
}

# Read and process each CSV
dfs = []
for comp_tool, path in files.items():
    df = pd.read_csv(path)
    # Add new column with the compression tool name
    df['compression_tool'] = comp_tool
    dfs.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Print the DataFrame columns to check available metric columns
print("Columns in the combined DataFrame:", combined_df.columns)

# Replace RunType values:
# "Chunked_Decompose_Parallel" or "Chunk-decompose_Parallel" become "TDT"
# "Full" becomes "standard"
# combined_df['RunType'] = combined_df['RunType'].replace({
#     "Chunked_Decompose_Parallel": "TDT",
#     "Chunk-decompose_Parallel": "TDT",
#     "Decompose_Chunk_Parallel": "TDT",
#     "Full": "standard"
# })

# Save the combined DataFrame to a CSV file
combined_csv_path = "/home/jamalids/Documents/result-throughput/combine-all.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved to: {combined_csv_path}")
######################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Load the CSV file
df = pd.read_csv("/home/jamalids/Documents/result-throughput/combine-all.csv")

# Group by compression_tool and RunType and calculate geometric mean of throughputs
grouped = df.groupby(['compression_tool', 'RunType'])
gmean_df = grouped.agg({
    'CompressionThroughput': lambda x: gmean(x[x > 0]),
    'DecompressionThroughput': lambda x: gmean(x[x > 0])
}).reset_index()
gmean_df['RunType'] = gmean_df['RunType'].replace({
    'Full': 'standard',
    'Chunked_Decompose_Parallel': 'first chunk then decompose',
    'Decompose_Chunk_Parallel': 'first decompose then chunk',
    'full': 'Standard'
})
# Pivot the table to prepare for plotting
pivot_df = gmean_df.pivot(index='compression_tool', columns='RunType', values=['CompressionThroughput', 'DecompressionThroughput'])


# Plotting setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

bar_width = 0.25
x = np.arange(len(pivot_df.index))

# Compression Throughput subplot
for i, runtype in enumerate(pivot_df['CompressionThroughput'].columns):
    ax1.bar(x + i * bar_width, pivot_df['CompressionThroughput'][runtype], width=bar_width, label=runtype)

ax1.set_title("Geometric Mean Compression Throughput")
ax1.set_ylabel("Throughput")
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(pivot_df.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(True)

# Decompression Throughput subplot
for i, runtype in enumerate(pivot_df['DecompressionThroughput'].columns):
    ax2.bar(x + i * bar_width, pivot_df['DecompressionThroughput'][runtype], width=bar_width, label=runtype)

ax2.set_title("Geometric Mean Decompression Throughput")
ax2.set_ylabel("Throughput")
ax2.set_xlabel("Compression Tool")
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(pivot_df.index, rotation=45, ha='right')
ax2.legend()
ax2.grid(True)

# Finalize layout
plt.tight_layout()


plt.savefig('/home/jamalids/Documents/Throughput.png')
###############################avrage##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("/home/jamalids/Documents/result-throughput/combine-all.csv")

# Group by compression_tool and RunType and calculate average of throughputs
grouped = df.groupby(['compression_tool', 'RunType'])
mean_df = grouped.agg({
    'CompressionThroughput': lambda x: x[x > 0].mean(),
    'DecompressionThroughput': lambda x: x[x > 0].mean()
}).reset_index()

# Pivot the table to prepare for plotting
pivot_df = mean_df.pivot(index='compression_tool', columns='RunType', values=['CompressionThroughput', 'DecompressionThroughput'])

# Plotting setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

bar_width = 0.25
x = np.arange(len(pivot_df.index))

# Compression Throughput subplot
for i, runtype in enumerate(pivot_df['CompressionThroughput'].columns):
    ax1.bar(x + i * bar_width, pivot_df['CompressionThroughput'][runtype], width=bar_width, label=runtype)

ax1.set_title("Average Compression Throughput")
ax1.set_ylabel("Throughput")
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(pivot_df.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(True)

# Decompression Throughput subplot
for i, runtype in enumerate(pivot_df['DecompressionThroughput'].columns):
    ax2.bar(x + i * bar_width, pivot_df['DecompressionThroughput'][runtype], width=bar_width, label=runtype)

ax2.set_title("Average Decompression Throughput")
ax2.set_ylabel("Throughput")
ax2.set_xlabel("Compression Tool")
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(pivot_df.index, rotation=45, ha='right')
ax2.legend()
ax2.grid(True)

# Finalize layout
plt.tight_layout()

plt.savefig('/home/jamalids/Documents/Throughputavrage.png')
