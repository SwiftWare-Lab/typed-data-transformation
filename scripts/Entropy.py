import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file (update the path as needed)
file_path = "/home/jamalids/Documents/WE/64-High-Entropy/combined_64H_data.csv"  # Change this to your actual path
df = pd.read_csv(file_path)

# Adjusting the plot to add space between the bars of different datasets
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis index for line plots with space between the points
index = range(len(df) * 2)

# Line plot for entropies
ax.plot(index[::2], df['entropy_float'], color='r', marker='o', label='Entropy of dataset')
ax.plot(index[::2], df['entropy_remainig'], color='b', marker='o', label='Entropy after Decomposition')

# Set labels and title
ax.set_xlabel('Dataset')
ax.set_ylabel('Entropy')
ax.set_xticks([i for i in index[::2]])
ax.set_xticklabels(df['dataset_name'], rotation=90)
ax.set_ylim(1, ax.get_ylim()[1])  # Set y-axis limits starting from 1
ax.legend(loc='upper left')

# Set the title
plt.title('Entropies for Datasets with Low Entropy After Removing Repetitive Consecutive Values')
plt.tight_layout()

# Save the plot
plt.savefig("/home/jamalids/Documents/WE/64-High-Entropy/combined_64H_data.png")

plt.show()
