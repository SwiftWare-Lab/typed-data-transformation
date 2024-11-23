import os
import pandas as pd

# Specify the directories containing the CSV files
directories = ['/home/jamalids/Documents/2D/data1/Fcbench/llog-gzip-H']

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
all_data_output_path = '/home/jamalids/Documents/2D/data1/Fcbench/llog-gzip-H/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')

# Find the median row for each dataset and Type
median_rows = []

for (dataset, type_), group in combined_df.groupby(['dataset', 'Type']):
    # Sort the group by TotalTimeCompressed and TotalTimeDecompressed
    sorted_group = group.sort_values(by=['TotalTimeCompressed', 'TotalTimeDecompressed']).reset_index(drop=True)

    # Find the median index
    median_index = len(sorted_group) // 2

    # Get the row at the median index
    median_row = sorted_group.iloc[median_index]

    # Append to the list as a DataFrame for concatenation
    median_rows.append(median_row)

# Concatenate all median rows into a single DataFrame
median_rows_df = pd.DataFrame(median_rows)

# Save the final DataFrame with only the median rows to a separate CSV file
median_data_output_path = '/home/jamalids/Documents/2D/data1/Fcbench/llog-gzip-H/combined_median_rows.csv'
median_rows_df.to_csv(median_data_output_path, index=False)

print(f'Combined CSV with median rows saved to {median_data_output_path}')
