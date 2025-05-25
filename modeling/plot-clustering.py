import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the main results CSV for your dataset
# Adjust the filename as needed
file_path = '/mnt/c/Users/jamalids/Downloads/dataset/HPC/test/turbulence_f32_hc_filtered.csv'

df = pd.read_csv(file_path)

# Ensure K is sorted correctly
df = df.sort_values(['Mode', 'K'])

# Define the modes and their subplot positions
modes = ['frequency', 'entropy', 'all', 'delta']
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axes = axes.flatten()

# Plot for each mode
for ax, mode in zip(axes, modes):
    sub = df[df['Mode'] == mode]
    x = sub['K']
    y = sub['DecomposedRatio']
    ax.scatter(x, y, color='C1', s=80)
    for _, row in sub.iterrows():
        ax.annotate(row['Partition'],
                    (row['K'], row['DecomposedRatio']),
                    textcoords='offset points',
                    xytext=(0, 8),
                    ha='center',
                    fontsize=10,
                    rotation=45)
    ax.set_title(f"{mode.capitalize()} Mode")
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Common Y label
fig.text(0.04, 0.5, 'DecomposedRatio', va='center', rotation='vertical', fontsize=12)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])


plt.savefig("/mnt/c/Users/jamalids/Downloads/dataset/HPC/test/turbulence_f32_hc_filtered_clustring.png")

