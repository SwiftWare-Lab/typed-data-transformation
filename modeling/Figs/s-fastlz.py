import pandas as pd
from scipy.stats import gmean

# Load the uploaded file
file_path = "/mnt/c/Users/jamalids/Downloads/figs/results/max_compression_throughput_pairs.csv"
df = pd.read_csv(file_path)

# Filter for TDT and Standard (Full)
df = df[df['RunType'].isin(['Full', 'Decompose_Chunk_Parallel'])].copy()
df['RunType'] = df['RunType'].replace({
    'Decompose_Chunk_Parallel': 'TDT',
    'Full': 'Standard'
})

# Compute geometric mean compression ratio
gmean_df = df.groupby('RunType').agg({
    'CompressionRatio': lambda x: gmean(x[x > 0])
}).reset_index()

gmean_df.rename(columns={
    'RunType': 'Mode',
    'CompressionRatio': 'GMean_CompressionRatio'
}, inplace=True)
print("\n🔍 Geometric Mean Compression Ratio (Based on Uploaded File):")
for _, row in gmean_df.iterrows():
    print(f"→ {row['Mode']}: {row['GMean_CompressionRatio']:.4f}")