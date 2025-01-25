import pandas as pd
import os
import glob

# Define the folder containing all input CSV files
input_folder = '//home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/32/'
output_directory = os.path.join(input_folder, 'compression_results')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

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

# Initialize a list to store top 5 results across all datasets
all_datasets_top_5_results = []

# Iterate over all CSV files in the input folder
for file_path in glob.glob(os.path.join(input_folder, '*.csv')):
    print(f"Processing file: {file_path}")

    # Load the dataset
    data = pd.read_csv(file_path)

    # Calculate compression ratios for each method
    for methods in grouped_methods.values():
        for method in methods:
            if method in data.columns:
                ratio_column = method.replace('compressed size (B)', 'compression ratio')
                data[ratio_column] = data['original size'] / data[method]

    # Initialize a list to store the formatted top 5 results for the current file
    formatted_top_5_results = []

    # Extract the top 5 results for each method and variation
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

    # Save the top 5 results for the current file
    output_file = os.path.join(output_directory, f"{os.path.basename(file_path).split('.')[0]}_top_5_results.csv")
    formatted_top_5_results_df.to_csv(output_file, index=False)
    print(f"Top 5 results for {file_path} saved to: {output_file}")

    # Append the current file's results to the combined list
    all_datasets_top_5_results.extend(formatted_top_5_results)

# Combine all datasets' results into a single DataFrame
combined_results_df = pd.DataFrame(all_datasets_top_5_results)

# Save the combined results to a single CSV file
combined_output_file = os.path.join(output_directory, "combined_top_5_compression_results.csv")
combined_results_df.to_csv(combined_output_file, index=False)

print(f"Combined top 5 results saved to: {combined_output_file}")
