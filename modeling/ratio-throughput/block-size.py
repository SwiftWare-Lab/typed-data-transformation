import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the CSV Data ---
data = pd.read_csv('/home/jamalids/Documents/combined_all_data.csv')

# --- 2. Filter the Data for the RunType "Decompose_Chunk_Parallel" ---
filtered = data[data['RunType'] == 'Decompose_Chunk_Parallel']

# --- 3. Group by BlockSize and Compute Mean Metrics ---
grouped = filtered.groupby('BlockSize').agg({
    'CompressionThroughput': 'mean',
    'DecompressionThroughput': 'mean',
    'CompressionRatio': 'mean'
}).reset_index()

# Sort by BlockSize (for clarity in plotting)
grouped.sort_values('BlockSize', inplace=True)

# --- 4. Normalize the Metrics to [0, 1] ---
grouped['CompressionThroughput_norm'] = (
    (grouped['CompressionThroughput'] - grouped['CompressionThroughput'].min()) /
    (grouped['CompressionThroughput'].max() - grouped['CompressionThroughput'].min())
)
grouped['DecompressionThroughput_norm'] = (
    (grouped['DecompressionThroughput'] - grouped['DecompressionThroughput'].min()) /
    (grouped['DecompressionThroughput'].max() - grouped['DecompressionThroughput'].min())
)
grouped['CompressionRatio_norm'] = (
    (grouped['CompressionRatio'] - grouped['CompressionRatio'].min()) /
    (grouped['CompressionRatio'].max() - grouped['CompressionRatio'].min())
)

# --- 5. Combine Throughput and Compression Ratio Metrics ---
# Average the two throughput metrics
grouped['Throughput_norm'] = (
    (grouped['CompressionThroughput_norm'] + grouped['DecompressionThroughput_norm']) / 2
)
# Create a combined score (you can adjust weights if needed)
grouped['Combined_Score'] = grouped['Throughput_norm'] + grouped['CompressionRatio_norm']

# --- 6. Identify the Best BlockSize Based on the Highest Combined Score ---
best_row = grouped.loc[grouped['Combined_Score'].idxmax()]
best_blocksize = best_row['BlockSize']

print("Best BlockSize:", best_blocksize)
print("Metrics for Best BlockSize:")
print(best_row)

# --- 7. Plot the Normalized Metrics with BlockSize as Text Labels on the X-Axis ---
fig, ax = plt.subplots(figsize=(10, 6))
# Use the index positions as the x-axis
x = np.arange(len(grouped))

# Plot each metric
ax.plot(x, grouped['Throughput_norm'], 'g-o', label='Throughput (norm)')
ax.plot(x, grouped['CompressionRatio_norm'], 'b-o', label='Compression Ratio (norm)')
ax.plot(x, grouped['Combined_Score'], 'r-o', label='Combined Score')

ax.set_xlabel('BlockSize')
ax.set_ylabel('Normalized Value')
ax.set_title('Normalized Throughput, Compression Ratio, and Combined Score by BlockSize')

# Set x-ticks using the index positions and label them with exact BlockSize values (as text)
ax.set_xticks(x)
ax.set_xticklabels(grouped['BlockSize'].astype(str), rotation=45, ha='right')

ax.legend(loc='best')
plt.tight_layout()

# Save and close the plot
output_plot_path = '/home/jamalids/Documents/plot_best_blocksize_dual_axis.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Dual-axis plot saved to {output_plot_path}")
