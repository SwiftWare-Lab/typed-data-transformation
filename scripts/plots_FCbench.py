import os
import pandas as pd

# Specify the directories containing the CSV files
directories = ['/home/jamalids/Documents/compression-part3/Fcbench/logE', '/home/jamalids/Documents/compression-part3/Fcbench/Results_10000K']

# List to hold DataFrames
dataframes = []

# Loop through each directory
for directory_path in directories:
    # Loop through each file in the directory
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(directory_path, file)  # Get the full path of the file
            df = pd.read_csv(file_path)  # Read the CSV file
            df['dataset'] = os.path.basename(file_path).replace('.csv', '')  # Adding the dataset name
            dataframes.append(df)  # Append the DataFrame to the list

# Merge all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_path = '/home/jamalids/Documents/compression-part3/Fcbench/combined_data.csv'
combined_df.to_csv(output_path, index=False)

print(f'Combined CSV saved to {output_path}')
