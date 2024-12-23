import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the uploaded CSV file
file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/citytemp_f32_present.csv"  # Update this path
data = pd.read_csv(file_path)

# Assuming 'original_size' is a column in your data, replace with actual if different
original_size = data['original_size']  # Assuming the first row contains the original size

# Define method categories based on your compression methods and their configurations
compression_methods = {
    'Huffman': ['decomposed huffman_compress compressed size (B)',
                'reordered huffman_compress compressed size (B)',
                'standard huffman_compress compressed size (B)'],
    'Zstd': ['decomposed zstd compressed size (B)',
             'reordered zstd compressed size (B)',
             'standard zstd compressed size (B)'],
    'Zlib': ['decomposed zlib compressed size (B)',
             'reordered zlib compressed size (B)',
             'standard zlib compressed size (B)'],
    'BZ2': ['decomposed bz2 compressed size (B)',
            'reordered bz2 compressed size (B)',
            'standard bz2 compressed size (B)'],
    'Snappy': ['decomposed snappy compressed size (B)',
               'reordered snappy compressed size (B)',
               'standard snappy compressed size (B)'],
    'FastLZ': ['decomposed fastlz compressed size (B)',
               'reordered fastlz compressed size (B)',
               'standard fastlz compressed size (B)'],
    'RLE': ['decomposed rle compressed size (B)',
            'reordered rle compressed size (B)',
            'standard rle compressed size (B)']
}

# Calculate compression ratios
for category, methods in compression_methods.items():
    for method in methods:
        data[method + '_ratio'] = original_size / data[method]

# Plotting each category with their specific decompositions
fig, axes = plt.subplots(nrows=len(compression_methods), figsize=(12, 20))  # Adjust subplot size and number as needed
axes = axes.flatten()

for ax, (category, methods) in zip(axes, compression_methods.items()):
    # Extracting ratio data for each method
    ratios = [data[method + '_ratio'].iloc[0] for method in methods]  # Assumes single data point for simplicity
    ax.bar(methods, ratios, color=np.random.rand(len(methods), 3))  # Random colors for each method
    ax.set_title(f'{category} Compression Ratios')
    ax.set_xlabel('Methods')
    ax.set_ylabel('Compression Ratio')
    ax.set_xticklabels(labels=methods, rotation=45, ha='right')

plt.tight_layout()
plt.show()
