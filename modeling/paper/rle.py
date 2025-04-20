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
                if 'DatasetName' not in df.columns and 'dataset' in df.columns:
                    df.rename(columns={'dataset': 'DatasetName'}, inplace=True)
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
###################################
df = pd.read_csv("/home/jamalids/Documents/combined_all_data.csv")
# Extracting necessary columns
plot_data = df[['dataset name', 'decomposed xor compression ratio', 'standard xor compression ratio']]

# Drop duplicate dataset names if they exist
plot_data = plot_data.drop_duplicates(subset='dataset name')

# Plotting
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = range(len(plot_data))

plt.bar(x, plot_data['decomposed xor compression ratio'], width=bar_width, label='TDT')
plt.bar([i + bar_width for i in x], plot_data['standard xor compression ratio'], width=bar_width, label='Standard ')

plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratio: TDT XOR vs Standard XOR')
plt.xticks([i + bar_width / 2 for i in x], plot_data['dataset name'], rotation=90)
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.savefig('/home/jamalids/Documents/rle_comparison.png')

