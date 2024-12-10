import os
import pandas as pd

# Specify the directories containing the CSV files
directories = ['/home/jamalids/Documents/32-H']

# List to hold DataFrames
dataframes = []

# Loop through each directory to load all CSV files
for directory_path in directories:
    # Loop through each file in the directory
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(directory_path, file)  # Get the full path of the file
            df = pd.read_csv(file_path)  # Read the CSV file
            df['dataset'] = os.path.basename(file_path).replace('.csv', '')  # Add the dataset name
            dataframes.append(df)  # Append the DataFrame to the list

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame with all data to a CSV file
all_data_output_path = '/home/jamalids/Documents/32-H/combined_32H_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')