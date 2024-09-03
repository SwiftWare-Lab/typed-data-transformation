import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate required metrics from the dataframe
def calculate_metrics(df):
    # Calculating averages for specified columns
    avg_com_ratio_b1 = df['com_ratio_b1'].max()
    avg_t_com_ratio_b1 = df['t_com_ratio_b1'].max()
    avg_com_ratio_b2 = df['com_ratio_b2'].max()
    avg_t_com_ratio_b2 = df['t_com_ratio_b2'].max()
    avg_com_ratio_b3 = df['com_ratio_b3'].max()
    avg_t_com_ratio_b3 = df['t_com_ratio_b3'].max()
    # Selecting the maximum average compression ratio across the b1, b2, b3
    max_avg_com_ratio = max(avg_com_ratio_b1, avg_com_ratio_b2, avg_com_ratio_b3)
    max_avg_t_com_ratio = max(avg_t_com_ratio_b1, avg_t_com_ratio_b2, avg_t_com_ratio_b3)
    # Additional metrics from the dataframe
    comp_ratio_zstd_default = df['comp_ratio_zstd_default'].max()
    comp_ratio_l22 = df['comp_ratio_l22'].max()
    Non_uniform_1x4 = df['Non_uniform_1x4'].max()
    return {
        'avg_com_ratio_b1': avg_com_ratio_b1,
        'avg_t_com_ratio_b1': avg_t_com_ratio_b1,
        'avg_com_ratio_b2': avg_com_ratio_b2,
        'avg_t_com_ratio_b2': avg_t_com_ratio_b2,
        'avg_com_ratio_b3': avg_com_ratio_b3,
        'avg_t_com_ratio_b3': avg_t_com_ratio_b3,
        'max_avg_com_ratio': max_avg_com_ratio,
        'max_avg_t_com_ratio':max_avg_t_com_ratio,
        'comp_ratio_zstd_default': comp_ratio_zstd_default,
        'comp_ratio_l22': comp_ratio_l22,
        'Non_uniform_1x4': Non_uniform_1x4
    }

#dataset_path = "/home/jamalids/Documents/compression-part3/big-data-compression/scripts/results"
dataset_path ="/home/jamalids/Documents/compression-part3/Fcbench/logE"
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.csv')]
results_df = pd.DataFrame()  # Initialize an empty DataFrame to hold all results

# Process each CSV file
for dataset_path in datasets:
    df = pd.read_csv(dataset_path)  # Read the CSV file
    metrics = calculate_metrics(df)  # Calculate metrics
    metrics['dataset'] = os.path.basename(dataset_path).replace('.csv', '')  # Adding the dataset name
    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)  # Append the results to the DataFrame
results_df.to_csv("second_fcbench.csv")
# Number of datasets per subplot
datasets_per_subplot = 4

# Create figure and axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through the axes and assign data
for i, ax in enumerate(axes):
    start_idx = i * datasets_per_subplot
    end_idx = start_idx + datasets_per_subplot
    subset = results_df[start_idx:end_idx]
    width = 0.15  # Narrower bar width to fit all bars

    # Define x positions for the groups of bars
    x = np.arange(len(subset))  # the label locations
    ax.bar(x - 1.5*width, subset['max_avg_com_ratio'], width, label='Com Ratio_Decompose')
    ax.bar(x - 0.5*width, subset['max_avg_t_com_ratio'], width, label='Com Ratio_Decompose_Dict')
    ax.bar(x + 0.5*width, subset['comp_ratio_zstd_default'], width, label='Zstd Default Comp Ratio')
    ax.bar(x + 1.5*width, subset['comp_ratio_l22'], width, label='Zstd 22 Comp Ratio')
    ax.bar(x + 2.5*width, subset['Non_uniform_1x4'], width, label='Non-uniform 1x4')

    ax.set_title(f'Datasets {start_idx+1} to {min(end_idx, len(results_df))}')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['dataset'], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylabel("Compression Ratio")  # Set y-axis label

plt.tight_layout()
plt.savefig('FCbench2.jpg', format='jpg', dpi=300)
plt.show()