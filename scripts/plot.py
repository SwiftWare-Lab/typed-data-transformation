import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np  # For logarithmic computation

# Set the backend to avoid interactive issues
matplotlib.use("Agg")  # Use non-interactive backend for saving plots

# Load the CSV file
file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/all/compression_results/combined_top_1_compression_results.csv"
data = pd.read_csv(file_path)

# Preprocess the data for clarity in visualization
# Extract relevant columns
plot_data = data[['dataset name', 'method', 'variation', 'compression ratio']].copy()

# Remove "_compress compressed size (B)" or "compressed size (B)" from variation
plot_data['variation'] = plot_data['variation'].str.replace(
    r"(_compress compressed size \(B\)|compressed size \(B\))",
    "",
    regex=True
).str.strip()

# Remove rows where the variation contains the word "reordered"
plot_data = plot_data[~plot_data['variation'].str.contains("reordered ", case=False, na=False)]

# Add log compression ratio
plot_data['log_compression_ratio'] = np.log(plot_data['compression ratio'])

# Generate pair bar plots for each method
unique_methods = plot_data['method'].unique()

# Iterate over unique methods
for method in unique_methods:
    method_data = plot_data[plot_data['method'] == method]

    # Standard compression ratio plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=method_data,
        x='dataset name',
        y='compression ratio',
        hue='variation',
        palette='viridis'
    )
    plt.title(f'Compression Ratio Comparison: {method.capitalize()}')
    plt.xlabel('Dataset Name')
    plt.ylabel('Compression Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Variation')
    plt.tight_layout()
    #plt.savefig(f"compression_ratio_{method}.png")
    plt.close()

    # Log compression ratio plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=method_data,
        x='dataset name',
        y='log_compression_ratio',
        hue='variation',
        palette='viridis'
    )
    plt.title(f'Log(Compression Ratio) Comparison: {method.capitalize()}')
    plt.xlabel('Dataset Name')
    plt.ylabel('Log(Compression Ratio)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Variation')
    plt.tight_layout()
    plt.savefig(f"log_compression_ratio_{method}.png")
    plt.close()

print("Plots have been saved as PNG files.")
