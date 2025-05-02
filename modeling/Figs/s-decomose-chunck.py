import pandas as pd
import numpy as np
from scipy.stats import gmean

# Load data
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/result-throughput/combine-all.csv")

# Filter valid throughput values
df = df[(df['CompressionThroughput'] > 0) & (df['DecompressionThroughput'] > 0)]

# Normalize RunType names
df['RunType'] = df['RunType'].replace({
    'Full': 'standard',
    'full': 'standard',
    'Chunked_Decompose_Parallel': 'first chunk then decompose',
    'Decompose_Chunk_Parallel': 'first decompose then chunk'
})

# Group by tool and RunType, calculate geometric means
grouped = df.groupby(['compression_tool', 'RunType']).agg({
    'CompressionThroughput': lambda x: gmean(x),
    'DecompressionThroughput': lambda x: gmean(x)
}).reset_index()

# Pivot the grouped data for easier comparison
pivot = grouped.pivot(index='compression_tool', columns='RunType', values=['CompressionThroughput', 'DecompressionThroughput'])

# Compute geometric mean of ratios across tools
results = []

for method in ['first chunk then decompose', 'first decompose then chunk']:
    if method in pivot['CompressionThroughput'].columns:
        comp_ratios = pivot['CompressionThroughput'][method] / pivot['CompressionThroughput']['standard']
        decomp_ratios = pivot['DecompressionThroughput'][method] / pivot['DecompressionThroughput']['standard']

        # Compute geometric mean across tools
        gmean_comp = gmean(comp_ratios.dropna())
        gmean_decomp = gmean(decomp_ratios.dropna())

        results.append((method, gmean_comp, gmean_decomp))

# Print the summary
print("\nGeometric Mean Throughput Ratios vs Standard:\n")
for method, gmean_comp, gmean_decomp in results:
    print(f"{method.upper()}:")
    print(f"  Compression Ratio:   {gmean_comp:.3f}×")
    print(f"  Decompression Ratio: {gmean_decomp:.3f}×\n")
