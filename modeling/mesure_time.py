import pandas as pd
import numpy as np
file_path = '/home/jamalids/CLionProjects/compress_paralle/cmake-build-debug/10.csv'
df= pd.read_csv(file_path)
# Calculate the maximum compression time for each group across all runs
max_compression_time = df.groupby('Group')['Compression Time (s)'].max().reset_index()
#max_compression_time = df.groupby('Group Name')['Compression Time (Level 3)'].max().reset_index()

# Rename columns for clarity
max_compression_time.columns = ['Group', 'Max Compression Time (s)']

# Display the results
print(max_compression_time)
