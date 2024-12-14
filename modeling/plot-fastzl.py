import matplotlib.pyplot as plt
import pandas as pd
import os

# File path
file_path = '/home/jamalids/Documents/solar_wind_f32.csv'

# Load the dataset
data = pd.read_csv(file_path)



# Setup the figure and axes for the subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True, sharey=True)  # Adjust based on number of datasets
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Unique datasets
datasets = data['dataset'].unique()

# Width of the bars in the bar chart
bar_width = 0.35

# Plot data for each dataset
for i, dataset in enumerate(datasets):
    # Subset data for the current dataset
    dataset_data_full = data[(data['dataset'] == dataset) & (data['Component'] == 'Full Data')]
    dataset_data_decomp = data[(data['dataset'] == dataset) & (data['Component'] == 'decomposition_ratio')]

    # Compute positions for the two sets of bars
    r1 = range(len(dataset_data_full))
    r2 = [x + bar_width for x in r1]

    # Create bar charts
    axes[i].bar(r1, dataset_data_full['Compression Ratio'], color='b', width=bar_width, label='Full Data')
    axes[i].bar(r2, dataset_data_decomp['Compression Ratio'], color='r', width=bar_width, label='Decomposition Ratio')

    # Add labels, title and custom x-axis tick labels
    axes[i].set_title(dataset)
    axes[i].set_xticks([r + bar_width/2 for r in range(len(r1))])
    axes[i].set_xticklabels(dataset_data_full['Configuration'], rotation=45)
    axes[i].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
