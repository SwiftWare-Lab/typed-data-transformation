import pandas as pd

# Load the data
file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/jw_mirimage_f32_decomposition_stats.csv"
data = pd.read_csv(file_path)

# Define FastLZ and Zstd compression methods
fastlz_methods = {
    'decomposed row-ordered fastlz compressed size (B)': 'Decomposed Fast LZ',
    'reordered row-ordered fastlz compressed size (B)': 'Reordered Fast LZ',
    'standard fastlz compressed size (B)': 'Standard Fast LZ'
}

zstd_methods = {
    'decomposed row-ordered zstd compressed size (B)': 'Decomposed Zstd',
    'reordered row-ordered zstd compressed size (B)': 'Reordered Zstd',
    'standard zstd compressed size (B)': 'Standard Zstd'
}

# Debug: Check columns
print("Columns in DataFrame:", data.columns)

# Check if 'original size' exists
if 'original size' not in data.columns:
    print("Error: 'original size' column is missing.")
else:
    # Calculate compression ratios for FastLZ
    for method in fastlz_methods.keys():
        if method in data.columns:
            print(f"Processing column: {method}")
            print(data[method].head())
            data[method.replace(' size (B)', ' ratio')] = data['original size'] / data[method]
        else:
            print(f"Column missing: {method}")

    # Calculate compression ratios for Zstd
    for method in zstd_methods.keys():
        if method in data.columns:
            print(f"Processing column: {method}")
            print(data[method].head())
            data[method.replace(' size (B)', ' ratio')] = data['original size'] / data[method]
        else:
            print(f"Column missing: {method}")

    # Debug: Check ratio columns
    print("FastLZ ratio columns:", [col for col in data.columns if 'fastlz compressed size (B) ratio' in col])
    print("Zstd ratio columns:", [col for col in data.columns if 'zstd compressed size (B) ratio' in col])

    # Find the best FastLZ method decomposition
    fastlz_best_ratios = data[[m.replace(' size (B)', ' ratio') for m in fastlz_methods.keys() if m in data.columns]]
    print("FastLZ best ratios preview:")
    print(fastlz_best_ratios.head())

    if not fastlz_best_ratios.empty:
        data['FastLZ best ratio'] = fastlz_best_ratios.max(axis=1)
        data['FastLZ best method'] = fastlz_best_ratios.idxmax(axis=1)
        print("Successfully calculated 'FastLZ best ratio'.")

        # Extract the top 20 rows by best FastLZ compression ratio
        top_20_best = data.nlargest(20, 'FastLZ best ratio')[
            [
                'FastLZ best ratio',
                'decomposition'
            ]
        ]

        # Print the results
        print("Top 20 Best Compression Ratios (FastLZ):")
        print(top_20_best)
    else:
        print("Error: No valid FastLZ ratio columns to calculate best ratios.")
