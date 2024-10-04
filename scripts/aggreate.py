import os
import pandas as pd

# Load the CSV files
dataset_path = "/home/jamalids/Documents/compression-part3/big-data-compression/modeling/results_final/"
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.csv')]

results = []
for dataset_path in datasets:
    data = pd.read_csv(dataset_path)  # Assuming the files are tab-separated
    # Calculate the maximum values for specific columns
    grouped_data = data.groupby('dataset_name').agg({
        'entropy_remainig': 'max',
        'entropy_float': 'max',
        'max_Decom+zstd_22_com_ratio': 'max',
        'max_Decom+zstd_com_ratio': 'max',
        'max_Decom+gzip_com_ratio': 'max',
        'comp_ratio_zstd_default': 'max',
        'comp_ratio_l22': 'max',
        'comp_ratio_gzip': 'max',
    }).reset_index()

    results.append(grouped_data)

# Concatenate all DataFrames in the results list
final_result = pd.concat(results, ignore_index=True)

# Save the concatenated DataFrame to a CSV file
final_result.to_csv("ALL_agg.csv", index=False)

print("Aggregation complete. Results saved to 'hpc_l_agg.csv'")
