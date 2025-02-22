import pandas as pd
from scipy.stats import gmean

# Replace 'your_file.csv' with the path to your CSV file.
df = pd.read_csv('/home/jamalids/Documents/max_decompression_throughput_pairs.csv')

# Check that the required columns exist
if 'RunType' not in df.columns or 'CompressionRatio' not in df.columns:
    raise ValueError("CSV must contain 'runtype' and 'CompressionRatio' columns.")

# Group by 'runtype' and calculate the geometric mean of 'CompressionRatio'
gmean_by_runtype = df.groupby('RunType')['CompressionRatio'].apply(lambda x: gmean(x)).reset_index()

print(gmean_by_runtype)
