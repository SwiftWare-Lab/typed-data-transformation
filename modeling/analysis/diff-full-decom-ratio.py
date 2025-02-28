import pandas as pd
import matplotlib.pyplot as plt

# Read in your CSV file (adjust the path if needed)
df = pd.read_csv("/home/jamalids/Documents/row-col-order/row-orde/fastlz1/results/max_decompression_throughput_pairs.csv")

# Pivot the data so that each dataset is indexed and run types become separate columns.
# Here we assume the CSV has columns: 'DatasetName', 'RunType', and 'CompressionRatio'
pivot_df = df.pivot_table(index='DatasetName', columns='RunType', values='CompressionRatio')

# Compute the ratio for each dataset:
# Ratio = CompressionRatio (Full) / CompressionRatio (Chunked_Decompose_Parallel)
pivot_df['Ratio(comp_ratio(Normal/TDT))'] = pivot_df['Full'] / pivot_df['Decompose_Block_Parallel']

# Save the pivot dataframe with the computed ratio as a CSV file.
csv_out = "/home/jamalids/Documents/compression_ratio_comparison.csv"
pivot_df.to_csv(csv_out)
print(f"CSV file saved to: {csv_out}")

# Create a bar plot for the computed ratio.
plt.figure(figsize=(10, 6))
pivot_df['Ratio(comp_ratio(Normal/TDT))'].plot(kind='bar', color='skyblue')
plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio (Standard / TDT)')
plt.title('Comparison of Compression Ratios: Standard vs TDT(fastlz)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot to a PNG file.
plot_out = "/home/jamalids/Documents/compression_ratio_comparison.png"
plt.savefig(plot_out)
print(f"Plot saved to: {plot_out}")

