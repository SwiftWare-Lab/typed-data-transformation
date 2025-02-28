import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Read in your CSV file (adjust the path if needed)
df = pd.read_csv("/home/jamalids/Documents/row-col-order/row-orde/fastlz1/results/max_decompression_throughput_pairs.csv")

# Pivot the data so that each dataset is indexed and run types become separate columns.
# Assumes the CSV has columns: 'DatasetName', 'RunType', and 'CompressionRatio'
pivot_df = df.pivot_table(index='DatasetName', columns='RunType', values='CompressionRatio')

# Compute the geometric mean for each dataset:
# For two values, the geometric mean is sqrt(Full * Chunked_Decompose_Parallel)
pivot_df['gmean'] = np.sqrt(pivot_df['Full'] * pivot_df['Decompose_Block_Parallel'])

# Save the pivot dataframe (including the computed gmean) as a CSV file for reference.
csv_out = "/home/jamalids/Documents/compression_ratio.csv"
pivot_df.to_csv(csv_out)
print(f"CSV file saved to: {csv_out}")

# ---------------- Plot 1: Full vs Chunked_Decompose_Parallel ----------------
plt.figure(figsize=(10, 6))
plt.plot(pivot_df.index, pivot_df['Full'], marker='o', linestyle='-', label='Standard')
plt.plot(pivot_df.index, pivot_df['Decompose_Block_Parallel'], marker='o', linestyle='-', label='TDT')

plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratio Comparison: Standard vs TDT')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Set y-axis to logarithmic scale and customize tick locator for more detail.
plt.yscale('log')
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PNG file.
plot_out1 = "/home/jamalids/Documents/compression_ratio.png"
plt.savefig(plot_out1)
print(f"Plot saved to: {plot_out1}")


# ---------------- Plot 2: Geometric Mean (gmean) ----------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in your CSV file (adjust the path if needed)
df = pd.read_csv("/home/jamalids/Documents/row-col-order/row-orde/fastlz1/results/max_decompression_throughput_pairs.csv")

# Pivot the data so that each dataset is indexed and run types become separate columns.
# Assumes the CSV has columns: 'DatasetName', 'RunType', and 'CompressionRatio'
pivot_df = df.pivot_table(index='DatasetName', columns='RunType', values='CompressionRatio')

# Compute the geometric mean for each run type across datasets.
# Geometric mean is computed as: gmean = exp(mean(log(values)))
gmean_full = np.exp(np.log(pivot_df['Full']).mean())
gmean_chunked = np.exp(np.log(pivot_df['Decompose_Block_Parallel']).mean())

# Create a DataFrame with the geometric means and update run type labels.
gm_df = pd.DataFrame({
    'RunType': ['Standard', 'TDT'],  # Renaming: "Full" becomes "Standard" and "Chunked_Decompose_Parallel" becomes "TDT"
    'GeometricMean': [gmean_full, gmean_chunked]
})

print("Geometric means computed:")
print(gm_df)

# Create a bar chart comparing the geometric means for each run type.
plt.figure(figsize=(8, 6))
bars = plt.bar(gm_df['RunType'], gm_df['GeometricMean'], color=['blue', 'orange'])
plt.xlabel('RunType')
plt.ylabel('Geometric Mean of Compression Ratio')
plt.title('Geometric Mean Compression Ratio using snappy (Standard vs TDT)')

# Optionally add the geometric mean values on top of each bar for clarity.
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()

# Save the plot as a PNG file.
plot_out = "/home/jamalids/Documents/geometric_mean_barchart.png"
plt.savefig(plot_out)
print(f"Plot saved to: {plot_out}")

