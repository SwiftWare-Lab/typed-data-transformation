import pandas as pd
import os

# Load the dataset
file_path = '/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/32/acs_wht_f32_decomposition_stats.csv'
data = pd.read_csv(file_path)

# Define compression methods grouped by type
grouped_methods = {
    'huffman': [
        'decomposed huffman_compress compressed size (B)',
        'decomposed row-ordered huffman_compress compressed size (B)',
        'reordered huffman_compress compressed size (B)',
        'reordered row-ordered huffman_compress compressed size (B)',
        'standard huffman_compress compressed size (B)'
    ],
    'zstd': [
        'decomposed zstd compressed size (B)',
        'decomposed row-ordered zstd compressed size (B)',
        'reordered zstd compressed size (B)',
        'reordered row-ordered zstd compressed size (B)',
        'standard zstd compressed size (B)'
    ],
    'zlib': [
        'decomposed zlib compressed size (B)',
        'decomposed row-ordered zlib compressed size (B)',
        'reordered zlib compressed size (B)',
        'reordered row-ordered zlib compressed size (B)',
        'standard zlib compressed size (B)'
    ],
    'bz2': [
        'decomposed bz2 compressed size (B)',
        'decomposed row-ordered bz2 compressed size (B)',
        'reordered bz2 compressed size (B)',
        'reordered row-ordered bz2 compressed size (B)',
        'standard bz2 compressed size (B)'
    ],
    'snappy': [
        'decomposed snappy compressed size (B)',
        'decomposed row-ordered snappy compressed size (B)',
        'reordered snappy compressed size (B)',
        'reordered row-ordered snappy compressed size (B)',
        'standard snappy compressed size (B)'
    ],
    'fastlz': [
        'decomposed fastlz compressed size (B)',
        'decomposed row-ordered fastlz compressed size (B)',
        'reordered fastlz compressed size (B)',
        'reordered row-ordered fastlz compressed size (B)',
        'standard fastlz compressed size (B)'
    ],
    'rle': [
        'decomposed rle compressed size (B)',
        'decomposed row-ordered rle compressed size (B)',
        'reordered rle compressed size (B)',
        'reordered row-ordered rle compressed size (B)',
        'standard rle compressed size (B)'
    ]
}

# Calculate compression ratios for each method
for methods in grouped_methods.values():
    for method in methods:
        if method in data.columns:
            ratio_column = method.replace('compressed size (B)', 'compression ratio')
            data[ratio_column] = data['original size'] / data[method]

# Create output directory if it does not exist
output_directory = '/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/32/compression_results'
os.makedirs(output_directory, exist_ok=True)

# Initialize a list to store the formatted top 5 results
formatted_top_5_results = []

# Reshape the data to store only the top 5 compression ratios for each variation in a single column
for group, methods in grouped_methods.items():
    for method in methods:
        ratio_column = method.replace('compressed size (B)', 'compression ratio')
        if ratio_column in data.columns:
            # Get the top 5 compression ratios for the current method and variation
            top_5 = data.nlargest(5, ratio_column)[
                ['dataset name', 'decomposition', ratio_column]
            ].copy()
            for _, row in top_5.iterrows():
                formatted_top_5_results.append({
                    'dataset name': row['dataset name'],
                    'decomposition': row['decomposition'],
                    'method': group,
                    'variation': method,
                    'compression ratio': row[ratio_column]
                })

# Convert the list of dictionaries to a DataFrame
formatted_top_5_results_df = pd.DataFrame(formatted_top_5_results)

# Save the formatted top 5 results to a CSV file
output_file = os.path.join(output_directory, "formatted_top_5_compression_results.csv")
formatted_top_5_results_df.to_csv(output_file, index=False)

print(f"Formatted top 5 results saved to: {output_file}")
