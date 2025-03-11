import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define a function to compute the geometric mean for a pandas Series.
def geometric_mean(series):
    # Filter out non-positive values to avoid math errors.
    positive_vals = series[series > 0]
    return np.exp(np.mean(np.log(positive_vals)))


# Dictionary mapping each method to its CSV file.
methods = {
    'fastlz': 'combine_fastlz.csv',
    'huffman': 'combine_huffman.csv',
    'rle': 'combine_rle.csv'
}

# Define common categories for the bar charts.
categories = ['Standard', 'Decomposed (col)', 'Decomposed (row)', 'Reordered (col)', 'Reordered (row)']

# Create a figure with three subplots (one row, three columns).
fig, axes = plt.subplots(3, 1, figsize=(9, 18))

# Loop over each method and its corresponding file.
for ax, (method, filename) in zip(axes, methods.items()):
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(filename)

    # Build the list of column names based on the method.
    ratio_columns = [
        f'standard {method} ratio',
        f'decomposed {method} col-order ratio',
        f'decomposed {method} row-order ratio',
        f'reordered {method} col-order ratio',
        f'reordered {method} row-order ratio'
    ]

    # Compute the geometric mean for each of the ratio columns.
    gmeans = [geometric_mean(df[col]) for col in ratio_columns]

    # Create a bar chart for the current method.
    bars = ax.bar(categories, gmeans)
    ax.set_ylabel('Geometric Mean Ratio')
    ax.set_title(f'Geometric Mean of {method.capitalize()} Compression Ratios')

    # Annotate each bar with its geometric mean value.
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

# Adjust the layout and save the plot to a file.
plt.tight_layout()
plt.savefig("compression_gmean.png")
#plt.show()
