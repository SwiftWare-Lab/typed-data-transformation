import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
###################
directories = ['/home/jamalids/Documents/results1']
dataframes = []

for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                # Read CSV file using semicolon as delimiter.
                df = pd.read_csv(file_path, sep=';')
                df.fillna(0, inplace=True)
                # Ensure a "DatasetName" column exists.
                if 'DatasetName' not in df.columns and 'dataset' in df.columns:
                    df.rename(columns={'dataset': 'DatasetName'}, inplace=True)
                if 'DatasetName' not in df.columns:
                    df['DatasetName'] = os.path.basename(file_path).replace('.csv', '')
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if not dataframes:
    print("No CSV files were processed. Please check the directories.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.fillna(0, inplace=True)
if 'Index' in combined_df.columns:
    combined_df.drop(columns=['Index'], inplace=True)

print("Columns in combined_df:", combined_df.columns.tolist())

all_data_output_path = '/home/jamalids/Documents/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')

# ====================================
# 2. Median Aggregation
# ====================================
required_cols = [
    'DatasetName', 'Threads', 'RunType', 'BlockSize', 'ConfigString',
    'TotalTimeCompressed', 'TotalTimeDecompressed',
    'CompressionThroughput', 'DecompressionThroughput', 'CompressionRatio'
]
missing_cols = [col for col in required_cols if col not in combined_df.columns]
if missing_cols:
    print("Missing columns in combined_df:", missing_cols)
    exit()

group_columns = ['DatasetName', 'Threads', 'RunType', 'BlockSize', 'ConfigString']
median_df = combined_df.groupby(group_columns, as_index=False).median(numeric_only=True)

median_output_path = '/home/jamalids/Documents/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')
##################compression Throughput import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
import seaborn as sns


# Load the uploaded CSV file
file_path = '/home/jamalids/Documents/combined_median_rows.csv'
df = pd.read_csv(file_path)

# Filter the DataFrame based on the specified conditions
filtered_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]

# Group by DatasetName and BlockSize, then calculate geometric means
gmean_df = filtered_df.groupby(['DatasetName', 'BlockSize']).agg({
    'CompressionRatio': gmean,
    'CompressionThroughput': gmean,
    'DecompressionThroughput': gmean
}).reset_index()

# Prepare data for plotting
datasets = gmean_df['DatasetName'].unique()
gmean_df_sorted = gmean_df.sort_values(by='DatasetName')
# Convert BlockSize to string for clarity in line plot
gmean_df_sorted['BlockSize'] = gmean_df_sorted['BlockSize'].astype(str)

# Set up subplots again with updated formatting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)

# First subplot: CompressionRatio with BlockSize (line instead of bar)
ax1 = axes[0]
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='CompressionRatio', ax=ax1, marker='o', label='GMean CompressionRatio', color='skyblue')
ax1.set_ylabel('GMean CompressionRatio', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Twin y-axis for BlockSize
ax1b = ax1.twinx()
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='BlockSize', ax=ax1b, color='orange', marker='s', label='BlockSize')
ax1b.set_ylabel('BlockSize (as string)', color='orange')
ax1b.tick_params(axis='y', labelcolor='orange')

# Second subplot: CompressionThroughput and DecompressionThroughput with BlockSize
ax2 = axes[1]
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='CompressionThroughput', ax=ax2, marker='o', label='GMean CompressionThroughput', color='green')
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='DecompressionThroughput', ax=ax2, marker='^', label='GMean DecompressionThroughput', color='purple')
ax2.set_ylabel('GMean Throughput')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(loc='upper left')

# Twin y-axis for BlockSize
ax2b = ax2.twinx()
sns.lineplot(data=gmean_df_sorted, x='DatasetName', y='BlockSize', ax=ax2b, color='orange', marker='s', label='BlockSize')
ax2b.set_ylabel('BlockSize (as string)', color='orange')
ax2b.tick_params(axis='y', labelcolor='orange')

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/Cot.png')
########################################
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Load the CSV file
file_path = '/home/jamalids/Documents/combined_median_rows.csv' # Change path if needed
df = pd.read_csv(file_path)

# Filter for RunType = Chunked_Decompose_Parallel and Threads = 16
chunked_df = df[(df['RunType'] == 'Chunked_Decompose_Parallel') & (df['Threads'] == 16)]

# Compute geometric mean by BlockSize
gmean_chunked = chunked_df.groupby('BlockSize').agg({
    'CompressionRatio': gmean,
    'CompressionThroughput': gmean,
    'DecompressionThroughput': gmean
}).reset_index()



# Convert BlockSize to string to treat them as categorical x-axis labels
gmean_chunked['BlockSize'] = gmean_chunked['BlockSize'].astype(str)

# Plotting again with BlockSize as string so all values are shown on x-axis
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

# Compression Ratio
axes[0].plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionRatio'], marker='o', color='blue', label='GMean CompressionRatio')
axes[0].set_ylabel('GMean CompressionRatio')
axes[0].set_title('GMean Compression Ratio vs BlockSize')
axes[0].legend()
axes[0].grid(True)

# Compression and Decompression Throughput
axes[1].plot(gmean_chunked['BlockSize'], gmean_chunked['CompressionThroughput'], marker='o', color='green', label='GMean CompressionThroughput')
axes[1].plot(gmean_chunked['BlockSize'], gmean_chunked['DecompressionThroughput'], marker='^', color='purple', label='GMean DecompressionThroughput')
axes[1].set_ylabel('GMean Throughput')
axes[1].set_xlabel('BlockSize')
axes[1].set_title('GMean Compression & Decompression Throughput vs BlockSize')
axes[1].legend()
axes[1].grid(True)

# Rotate x-axis labels for better visibility
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/Throughput.png')
