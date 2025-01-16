import pandas as pd
import os

# Load the uploaded CSV file
file_path = r"C:\Users\jamalids\Downloads\results\OneDrive_4_12-23-2024\CR-Ct-DT\python-results\logs\tpch_order_f64_decomposition_stats.csv"
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

# Identify the top 5 compression ratios for each group
output_directory = r'C:\Users\jamalids\Downloads\tpch_order_f64_decomposition_stats\\'
os.makedirs(output_directory, exist_ok=True)

# ------------------------------------------------------------
# NEW: Create a list to collect top-5 data from ALL groups
all_groups_top_5_list = []
# ------------------------------------------------------------

for group, methods in grouped_methods.items():
    # Collect top 5 ratios for the current group
    group_top_5 = pd.concat([
        data.nlargest(10, method.replace('compressed size (B)', 'compression ratio'))[
            [method.replace('compressed size (B)', 'compression ratio'), 'dataset name', 'decomposition']
        ]
        for method in methods if method.replace('compressed size (B)', 'compression ratio') in data.columns
    ])

    # Save to a CSV file
    output_file = os.path.join(output_directory, f"{group}_top_5.csv")
    group_top_5.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # NEW: Append this group's top-5 data to our master list
    all_groups_top_5_list.append(group_top_5)
    # --------------------------------------------------------

# ----------------------------------------------------------------
# NEW: After the loop, combine all groups into a single CSV
all_groups_top_5 = pd.concat(all_groups_top_5_list, ignore_index=True)
all_groups_csv = os.path.join(output_directory, "ALL_GROUPS_top_5.csv")
all_groups_top_5.to_csv(all_groups_csv, index=False)
# ----------------------------------------------------------------

print(f"All top 5 compression ratio files have been saved to: {output_directory}")
print(f'Additionally, "ALL_GROUPS_top_5.csv" with all top-5 data is in {output_directory}')
