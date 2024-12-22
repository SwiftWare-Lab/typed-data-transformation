import os
import pandas as pd

# Specify the directories containing the CSV files
directories = ['/home/jamalids/Documents/2D/CR-Ct-DT/bz2-Cpp/logbz32-H']

# List to hold DataFrames
dataframes = []

# Loop through each directory to load all CSV files
for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(directory_path, file)  # Get the full path of the file
            try:
                df = pd.read_csv(file_path)  # Read the CSV file
                df['dataset'] = os.path.basename(file_path).replace('.csv', '')  # Add the dataset name
                dataframes.append(df)  # Append the DataFrame to the list
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all DataFrames into a single DataFrame
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame with all data to a CSV file
    all_data_output_path = '/home/jamalids/Documents/2D/CR-Ct-DT/bz2-Cpp/combined_all_data.csv'
    combined_df.to_csv(all_data_output_path, index=False)
    print(f'Combined CSV with all data saved to {all_data_output_path}')

    # Find the median row for each dataset and RunType
    median_rows = []
    for (dataset, run_type, component_sizes), group in combined_df.groupby(['dataset', 'RunType', 'ComponentSizes']):
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
    median_data_output_path = '/home/jamalids/Documents/2D/CR-Ct-DT/bz2-Cpp/combined_median_rows.csv'
    median_rows_df.to_csv(median_data_output_path, index=False)
    print(f'Combined CSV with median rows saved to {median_data_output_path}')

    # Load the combined_median_rows.csv
    try:
        median_rows_df = pd.read_csv(median_data_output_path)

        # Initialize a list to store results
        max_comp_ratio_rows = []

        # Group by dataset
        grouped = median_rows_df.groupby('dataset')

        for dataset, group in grouped:
            # Find the row with the maximum compression ratio for RunType = Parallel
            parallel_rows = group[group['RunType'] == 'Parallel']
            if not parallel_rows.empty:
                # Create a copy of parallel_rows before modifying it to avoid the warning
               # parallel_rows = parallel_rows.copy()

                # Add rounded compression ratio column
               # parallel_rows['RoundedCompressionRatio'] = parallel_rows['CompressionRatio']  # Round to 3 decimal places

                max_compression_ratio = parallel_rows['CompressionRatio'].max()  # Find max rounded compression ratio
                filtered_rows = parallel_rows[parallel_rows['CompressionRatio'] == max_compression_ratio]  # Filter rows with max compression ratio
                max_parallel_row = filtered_rows.loc[filtered_rows['TotalTimeCompressed'].idxmin()]  # Find the row with min TotalTimeCompressed among filtered rows
                max_comp_ratio_rows.append(max_parallel_row)

                # Find the corresponding Full row with the same ComponentSizes
                full_rows = group[(group['RunType'] == 'Full') &
                                  (group['ComponentSizes'] == max_parallel_row['ComponentSizes'])]
                if not full_rows.empty:
                    corresponding_full_row = full_rows.iloc[0]  # Select the first matching Full row
                    max_comp_ratio_rows.append(corresponding_full_row)
        # Create a DataFrame from the results
        max_comp_ratio_df = pd.DataFrame(max_comp_ratio_rows)

        # Save the result to a new CSV file
        max_comp_ratio_output_path = '/home/jamalids/Documents/2D/CR-Ct-DT/bz2-Cpp/max_comp_ratio_parallel_full.csv'
        max_comp_ratio_df.to_csv(max_comp_ratio_output_path, index=False)

        print(f'CSV with max compression ratio for parallel and corresponding full RunType saved to {max_comp_ratio_output_path}')

    except Exception as e:
        print(f"Error processing median rows: {e}")

else:
    print("No CSV files were processed. Please check the directories.")