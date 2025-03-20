import os
import pandas as pd

# Specify the directories containing the CSV files
directories = ["/home/jamalids/Documents/all-config-cluster/logs-zstd"]

# List to hold DataFrames
dataframes = []

# Loop through each directory to load all CSV files
for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(directory_path, file)  # Get the full path of the file
            try:
                df = pd.read_csv(file_path)  # Read the CSV file
               # df['dataset'] = os.path.basename(file_path).replace('.csv', '')  # Add the dataset name
                print( os.path.basename(file_path).replace('.csv', ''))
                dataframes.append(df)  # Append the DataFrame to the list
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all DataFrames into a single DataFrame
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame with all data to a CSV file
    all_data_output_path = "/home/jamalids/Documents/all-config-cluster/combine.csv"
    combined_df.to_csv(all_data_output_path, index=False)
    print(f'Combined CSV with all data saved to {all_data_output_path}')


