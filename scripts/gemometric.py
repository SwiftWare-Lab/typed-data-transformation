from scipy.stats import gmean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/geometric/standard.csv"  # Change this to your actual path

data = pd.read_csv(file_path)

# Group the data by 'RunType' and calculate the geometric mean for each group
# geometric_means = data.groupby('RunType')['CompressionRatio'].apply(gmean).reset_index()
# geometric_CompThrough = data.groupby('RunType')['CompressionThroughput'].apply(gmean).reset_index()
# geometric_DecompThrough = data.groupby('RunType')['DecompressionThroughput'].apply(gmean).reset_index()
# print(geometric_means)
# print("Compression Throughput", geometric_CompThrough)
# print("Decompression Throughput", geometric_DecompThrough)


# Calculate the geometric means for the entire columns
geometric_mean_compr_ratio = gmean(data['Geometric Mean of Compression Ratio'])
geometric_mean_compr_throughput = gmean(data['Gmean CT'])
geometric_mean_decompr_throughput = gmean(data['Gmean DT'])

# Print results
print("Geometric Mean of Compression Ratio:", geometric_mean_compr_ratio)
print("Geometric Mean of Compression Throughput:", geometric_mean_compr_throughput)
print("Geometric Mean of Decompression Throughput:", geometric_mean_decompr_throughput)
