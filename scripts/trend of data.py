import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set dataset path and gather datasets
dataset_path = "/home/jamalids/Documents/2D/data1/HPC/H/wave_f32.tsv"
#datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]
datasets = [dataset_path ]

# Since you have 7 datasets
datasets = datasets[:7]

# Initialize lists to store the values for each dataset and DataFrames
all_values = []
dataframes = []

# Function to load dataset and return raw values
def load_dataset_values(dataset_path):
    """Load the dataset and return the raw values."""
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    group = ts_data1.drop(ts_data1.columns[0], axis=1)
    group = group.iloc[0:4000000, :]  # Limit to 4,000,000 rows for each dataset
    group = group.T
    values = group.astype(np.float32).to_numpy().reshape(-1)
    return values

# Load data from each dataset and store the raw values
for dataset_path in datasets:
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    print("Processing dataset:", dataset_name)
    values = load_dataset_values(dataset_path)
    all_values.append((dataset_name, values))

# Find the maximum length of the datasets
max_len = max(len(values) for _, values in all_values)

# Pad datasets to have the same length using NaN
padded_values = {dataset_name: np.pad(values, (0, max_len - len(values)), constant_values=np.nan)
                 for dataset_name, values in all_values}

# Create 2 DataFrames, one with 4 datasets and one with 3 datasets
df1 = pd.DataFrame({dataset_name: padded_values[dataset_name] for dataset_name, _ in all_values[:4]})
df2 = pd.DataFrame({dataset_name: padded_values[dataset_name] for dataset_name, _ in all_values[4:]})

# Save each DataFrame to a CSV file
df1.to_csv('dataframe1.csv', index=False)
df2.to_csv('dataframe2.csv', index=False)

# Create a figure with 7 subplots (arranged in 2 rows and 4 columns)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot the raw values for each dataset in a separate subplot
for i, (dataset_name, values) in enumerate(all_values):
    ax = axes[i]
    ax.plot(values, label=dataset_name)
    ax.set_title(f'{dataset_name}')
    ax.set_xlabel('Data Points (Index)')
    ax.set_ylabel('Values')
    ax.legend()

# Hide the last empty subplot (since there are 7 datasets and 8 plots)
axes[-1].axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the plots to a file
plt.savefig("Trend_of_8_Datasets.png")

# Show the plots
plt.show()
