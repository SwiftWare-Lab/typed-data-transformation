import pandas as pd
import os
import ast

# --------------------------- Configuration --------------------------- #

# Define the input CSV file path
file_path = r"C:\Users\jamalids\Downloads\results\OneDrive_4_12-23-2024\CR-Ct-DT\python-results\logs\wave_f32_decomposition_stats.csv"

# Define the output directory where top 5 results and specific decompositions will be saved
output_directory = r'C:\Users\jamalids\Downloads\compression_top_5_hdr_night_f32\\'
os.makedirs(output_directory, exist_ok=True)

# Define the specific decompositions you want to analyze
specific_decompositions = [( (0,1), (2,), (3,), )]

# --------------------------- Load Data --------------------------- #

print("Loading data...")
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully.\n")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit(1)

# --------------------------- Parse Decomposition Tuples --------------------------- #

# Assuming 'decomposition' column contains tuples as strings, e.g., "(3,0,1)"
print("Parsing decomposition tuples...")
if 'decomposition' in data.columns:
    data['decomposition'] = data['decomposition'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    print("Decomposition tuples parsed successfully.\n")
else:
    print("Error: 'decomposition' column not found in the dataset.")
    exit(1)

# --------------------------- Define Compression Methods --------------------------- #

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

# --------------------------- Calculate Compression Ratios --------------------------- #

print("Calculating compression ratios...")
# Ensure 'original size' column exists
if 'original size' not in data.columns:
    print("Error: 'original size' column not found in the dataset.")
    exit(1)

# Calculate compression ratios for each method
for group, methods in grouped_methods.items():
    for method in methods:
        if method in data.columns:
            ratio_column = method.replace('compressed size (B)', 'compression ratio')
            data[ratio_column] = data['original size'] / data[method]
            print(f"Calculated compression ratio for '{method}' as '{ratio_column}'.")
        else:
            print(f"Warning: Method '{method}' not found in the dataset columns.")

print("\nCompression ratio calculation completed.\n")

# --------------------------- Identify Top 5 Compression Ratios --------------------------- #

print("Identifying top 5 compression ratios for each compression method group...\n")
for group, methods in grouped_methods.items():
    # Initialize an empty DataFrame to store top 5 for this group
    group_top_5 = pd.DataFrame(columns=['compression ratio', 'dataset name', 'decomposition', 'method'])

    for method in methods:
        ratio_column = method.replace('compressed size (B)', 'compression ratio')
        # Check if the ratio column was successfully created
        if ratio_column in data.columns:
            # Select relevant columns
            subset = data[['dataset name', 'decomposition', ratio_column]].copy()
            subset.rename(columns={ratio_column: 'compression ratio'}, inplace=True)
            # Add a new column to indicate the method
            subset['method'] = method
            # Get top 5 entries based on compression ratio
            top_5 = subset.nlargest(5, 'compression ratio')
            # Append to the group's top 5 DataFrame
            group_top_5 = pd.concat([group_top_5, top_5], ignore_index=True)
        else:
            print(f"Warning: Compression ratio column '{ratio_column}' does not exist for method '{method}'.")

    # Sort the group's top 5 by compression ratio descending
    group_top_5.sort_values(by='compression ratio', ascending=False, inplace=True)
    # Drop duplicates if any (optional)
    group_top_5.drop_duplicates(subset=['dataset name', 'decomposition', 'method'], inplace=True)
    # Select only top 5
    group_top_5 = group_top_5.head(5)

    # Save to a CSV file
    output_file = os.path.join(output_directory, f"{group}_top_5.csv")
    group_top_5.to_csv(output_file, index=False)
    print(f"Top 5 compression ratios for '{group}' saved to '{output_file}'.")

print("\nTop 5 compression ratios identification completed.\n")

# --------------------------- Displaying Top 5 with Decomposition --------------------------- #

# Optionally, display the top 5 results for each group
for group in grouped_methods.keys():
    output_file = os.path.join(output_directory, f"{group}_top_5.csv")
    if os.path.exists(output_file):
        print(f"Top 5 compression ratios for '{group}':")
        print(pd.read_csv(output_file))
        print("\n")
    else:
        print(f"No top 5 data found for '{group}'.\n")

# --------------------------- Filter and Display Specific Decompositions --------------------------- #

print("Filtering and displaying compression ratios for specific decompositions...\n")
# Filter data for the specified decompositions
filtered_data = data[data['decomposition'].isin(specific_decompositions)]

if filtered_data.empty:
    print(f"No data found for the specified decompositions: {specific_decompositions}\n")
else:
    # Initialize an empty DataFrame to store filtered results
    specific_top = pd.DataFrame(columns=['compression ratio', 'dataset name', 'decomposition', 'method'])

    for group, methods in grouped_methods.items():
        for method in methods:
            ratio_column = method.replace('compressed size (B)', 'compression ratio')
            if ratio_column in filtered_data.columns:
                # Select relevant columns
                subset = filtered_data[['dataset name', 'decomposition', ratio_column]].copy()
                subset.rename(columns={ratio_column: 'compression ratio'}, inplace=True)
                # Add a new column to indicate the method
                subset['method'] = method
                # Append to the specific_top DataFrame
                specific_top = pd.concat([specific_top, subset], ignore_index=True)
            else:
                print(f"Warning: Compression ratio column '{ratio_column}' does not exist for method '{method}'.")

    # Sort by compression ratio descending
    specific_top.sort_values(by='compression ratio', ascending=False, inplace=True)
    # Reset index
    specific_top.reset_index(drop=True, inplace=True)

    # Save to a CSV file
    specific_output_file = os.path.join(output_directory, f"specific_decompositions_compression_ratios.csv")
    specific_top.to_csv(specific_output_file, index=False)
    print(f"Compression ratios for specific decompositions saved to '{specific_output_file}'.\n")

    # Display the filtered data
    print(f"Compression ratios for specified decompositions {specific_decompositions}:")
    print(specific_top)
    print("\n")

# --------------------------- End of Script --------------------------- #
