import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/home/jamalids/Documents/c-32-clustering/combined_all_data.csv'
data = pd.read_csv(file_path)

# Filter the data to include only 'Full' and 'Parallel' RunTypes
filtered_data = data[data['RunType'].isin(['Full', 'Parallel'])].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Combine DatasetName and ConfigString for better x-axis labels
filtered_data.loc[:, 'Dataset_Config'] = filtered_data['DatasetName'] + " | " + filtered_data['ConfigString']

# Switch to a compatible backend if needed
import matplotlib
matplotlib.use('Agg')  # You can also use 'TkAgg' or other compatible options

# Create the bar plot
plt.figure(figsize=(16, 8))
sns.barplot(
    data=filtered_data,
    x='Dataset_Config',
    y='CompressionRatio',
    hue='RunType'
)

# Enhance plot aesthetics
plt.title('Compression Ratio Comparison for Full and Parallel Runs by Dataset and ConfigString', fontsize=16)
plt.xlabel('Dataset and ConfigString', fontsize=14)
plt.ylabel('Compression Ratio', fontsize=14)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.legend(title='Run Type', fontsize=12)
plt.tight_layout()

# Save the plot as an image file (if rendering issues persist with plt.show())
output_path = '/home/jamalids/Documents/c-32-clustering/compression_ratio_plot.png'
plt.savefig(output_path)

# Show the plot (optional, may not work with certain backends)
plt.show()
##############################
