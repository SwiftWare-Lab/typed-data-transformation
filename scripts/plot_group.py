import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/jamalids/Documents/compression-part3/Fcbench/combined_data.csv'
data = pd.read_csv(file_path)

# Assuming 'dataset' and 'm' columns exist and are used to differentiate the data
# You might need to adjust column names based on your actual data structure

# Plotting three separate figures for different metrics
metrics_to_plot = ['max_com_ratio', 't-max_com_ratio', 'Non_uniform_1x4', 'comp_ratio_l22', 'comp_ratio_zstd_default']
titles = ['decompose_comp_Ratio', 'dict_decompose_comp_Ratio', 'Non-uniform 1x4', 'Compression Ratio L22', 'Zstd Default Compression Ratio']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))

# Filter and plot data for each subplot
for i, ax in enumerate(axes.flatten()):
    if i < len(metrics_to_plot):  # Check to avoid index errors if less than 3 metrics
        metric = metrics_to_plot[i]
        # Pivot data for better plotting by 'dataset' and 'm'
        pivot_df = data.pivot_table(index=['dataset', 'M'], values=metric, aggfunc='mean').reset_index()
        for key, grp in pivot_df.groupby('dataset'):
            ax.plot(grp['M'], grp[metric], marker='o', linestyle='-', label=key)
        ax.set_title(titles[i])
        ax.set_xlabel('m Value')
        ax.set_ylabel('Compression Ratio')
        ax.legend(title='Dataset', loc='best')
    else:
        ax.set_visible(False)  # Hide unused subplots if any

plt.tight_layout()
plt.show()
