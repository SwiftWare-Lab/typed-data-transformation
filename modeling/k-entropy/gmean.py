import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('zstd/combine.csv')

# List of column names for the five ratio columns
ratio_columns = [
    'standard zstd ratio',
    'decomposed zstd col-order ratio',
    'decomposed zstd row-order ratio',
    'reordered zstd col-order ratio',
    'reordered zstd row-order ratio'
]

# Define a function to compute the geometric mean
def geometric_mean(series):
    # Filtering out non-positive values if needed
    positive_vals = series[series > 0]
    return np.exp(np.mean(np.log(positive_vals)))

# Compute the geometric mean for each column
gmeans = [geometric_mean(df[col]) for col in ratio_columns]

# Define categories for plotting
categories = ['Standard', 'Decomposed (col)', 'Decomposed (row)', 'Reordered (col)', 'Reordered (row)']

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, gmeans)
plt.ylabel('Geometric Mean Ratio')
plt.title('Geometric Mean of zstd Compression Ratios')

# Annotate each bar with its geometric mean value
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("zstd-gmean.png")
