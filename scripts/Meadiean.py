import numpy as np
import pandas as pd
file_path = '/home/jamalids/Documents/results/combine/combined_all_data.csv'
combined_df = pd.read_csv(file_path)
# Find the median row for each dataset and Type
median_rows = []

for (dataset, type_,ComponentSizes_,Threads_), group in combined_df.groupby(['DatasetName', 'RunType','ConfigString','Threads']):
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
median_data_output_path = '/home/jamalids/Documents/results/combine/combined_median.csv'
median_rows_df.to_csv(median_data_output_path, index=False)

print(f'Combined CSV with median rows saved to {median_data_output_path}')
