import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = '/home/jamalids/Documents/resultsnumber/final.csv'
data= pd.read_csv(file_path)
# Extract unique decomposition types
decompositions = data['decompose'].unique()
x = np.arange(len(decompositions))  # Indices for the x-axis

# Extract the relevant data for plotting
pattern_entropy = data['Pattern-entropy']
full_entropy = data['total_entropy']
weighted_entropy = data['wighted_entropy']
compression_ratios = data['CompressionRatio']

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar chart for compression ratios
bars = ax1.bar(x, compression_ratios, width=0.3, label='Compression Ratio', color='orange', alpha=0.7)
ax1.set_ylabel('Compression Ratio')
ax1.set_xlabel('Decompositions')
ax1.set_xticks(x)
ax1.set_xticklabels(decompositions, rotation=45, ha='right')
ax1.grid(axis='y')

# Line plots for entropy values
ax2 = ax1.twinx()
ax2.plot(x, pattern_entropy, label='Pattern Entropy', color='blue', marker='o', linestyle='-')
ax2.plot(x, full_entropy, label='Full Entropy', color='green', marker='s', linestyle='--')
ax2.plot(x, weighted_entropy, label='Weighted Entropy', color='red', marker='^', linestyle=':')
ax2.set_ylabel('Entropy')

# Combine legends and position them outside the plot
lines, labels = ax2.get_legend_handles_labels()
bars, bar_labels = ax1.get_legend_handles_labels()
ax1.legend(lines + bars, labels + bar_labels, loc='center left', bbox_to_anchor=(1.05, 0.5))

# Title and layout adjustments
plt.title('Entropy and Compression Ratios for Each Decomposition')
plt.tight_layout()

# Show the plot
plt.show()
