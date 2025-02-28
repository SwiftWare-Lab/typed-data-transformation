# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Read in your CSV file
# df = pd.read_csv("/home/jamalids/Documents/blosc.csv")
#
# # Assume the CSV has columns: 'dataset', 'row_order_ratio', 'col_order_ratio'
# # If you have multiple rows per dataset, you can aggregate (for example, using mean)
# agg_df = df.groupby('dataset name', as_index=False).agg({
#     'aggregated decomposed row-order blosc ratio': 'mean',
#     'aggregated decomposed col-order blosc ratio': 'mean'
# })
#
# # Create a grouped bar chart
# x = np.arange(len(agg_df))
# width = 0.35  # width of the bars
#
# fig, ax = plt.subplots(figsize=(8, 5))
# rects1 = ax.bar(x - width/2, agg_df['aggregated decomposed row-order blosc ratio'], width, label='Row-order')
# rects2 = ax.bar(x + width/2, agg_df['aggregated decomposed col-order blosc ratio'], width, label='Col-order')
#
# # Add labels, title, and custom x-axis tick labels
# ax.set_ylabel('Blosc Ratio')
# ax.set_title('Aggregated Decomposed Blosc Ratio by Order across Datasets')
# ax.set_xticks(x)
# ax.set_xticklabels(agg_df['dataset name'], rotation=45, ha='right')
# ax.legend()
#
# plt.tight_layout()
# plt.savefig("/home/jamalids/Documents/blosc.png")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in your CSV file
df = pd.read_csv("/home/jamalids/Documents/blosc.csv")

# Group by 'dataset name' and aggregate the ratios using the mean
agg_df = df.groupby('dataset name', as_index=False).agg({
    'aggregated decomposed row-order blosc ratio': 'mean',
    'aggregated decomposed col-order blosc ratio': 'mean'
})

# Create x positions for each dataset
x = np.arange(len(agg_df))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the line charts for row-order and col-order ratios
ax.plot(x, agg_df['aggregated decomposed row-order blosc ratio'],
        marker='o', color='blue', linestyle='-', label='Row-order')
ax.plot(x, agg_df['aggregated decomposed col-order blosc ratio'],
        marker='o', color='orange', linestyle='-', label='Col-order')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Blosc Ratio')
ax.set_title('Aggregated Decomposed Blosc Ratio by Order across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(agg_df['dataset name'], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/blosc1.png")

