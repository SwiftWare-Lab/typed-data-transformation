
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or use another backend if desired
import matplotlib.pyplot as plt
import re
directories = ['/home/jamalids/Documents/logs1-xor']
dataframes = []

for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                # Read CSV file using semicolon as delimiter.
                df = pd.read_csv(file_path, sep=',')
                df.fillna(0, inplace=True)
                # Ensure a "DatasetName" column exists.
                if 'DatasetName' not in df.columns and 'dataset name' in df.columns:
                    df.rename(columns={'dataset name': 'DatasetName'}, inplace=True)
                if 'DatasetName' not in df.columns:
                    df['DatasetName'] = os.path.basename(file_path).replace('.csv', '')
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if not dataframes:
    print("No CSV files were processed. Please check the directories.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.fillna(0, inplace=True)
if 'Index' in combined_df.columns:
    combined_df.drop(columns=['Index'], inplace=True)

print("Columns in combined_df:", combined_df.columns.tolist())

all_data_output_path = '/home/jamalids/Documents/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')
# Reload uploaded files
main_path = "/home/jamalids/Documents/combined_all_data.csv"
meta_path = "/home/jamalids/Documents/datasetname/dataset_id_mapping.csv"

main_df = pd.read_csv(main_path)
meta_df = pd.read_csv(meta_path)

# Merge the data on 'dataset name'
merged_df = pd.merge(main_df, meta_df, on="DatasetName")

# Calculate the ratio of decomposed xor compression ratio to standard xor compression ratio
merged_df['Ratio'] = merged_df['decomposed xor compression ratio'] / merged_df['standard xor compression ratio']

# Sort by domain and then entropy
sorted_df = merged_df.sort_values(by=['Domain', 'Entropy'])

# Generate an ID for each dataset (e.g., 1, 2, 3...) for x-axis
sorted_df['dataset_id'] = range(1, len(sorted_df) + 1)

# Plotting with dataset_id on x-axis
plt.figure(figsize=(14, 6))
bars = plt.bar(sorted_df['DatasetID'], sorted_df['Ratio'])

# Show dataset_id as x-axis ticks (numeric only)
plt.xticks(sorted_df['DatasetID'], sorted_df['DatasetID'], rotation=0, fontsize=10)
plt.ylabel("TDT / Standard Compression Ratio using(xor)")
plt.xlabel("Dataset ID ")
#plt.title("Decomposed vs Standard Compression Ratio using  XOR")

# Add value labels above each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    dataset_name = sorted_df.iloc[i]['DatasetName']
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
             ha='center', va='bottom', fontsize=10, rotation=90)

plt.tight_layout()

# Save the plot
plt.savefig("/home/jamalids/Documents/fundation-plot-xor.png")
plt.savefig("/home/jamalids/Documents/fundation-plot-xor.pdf", format='pdf')
plt.show()
