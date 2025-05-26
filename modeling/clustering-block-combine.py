# import os
# import pandas as pd
# # Specify the directories containing the CSV files
# directories = [ '/mnt/c/Users/jamalids/Downloads/dataset/HPC/block2']
#
# # List to hold DataFrames
# dataframes = []
#
# # Loop through each directory to load all CSV files
# for directory_path in directories:
#     for file in os.listdir(directory_path):
#         if file.endswith('.csv'):  # Check if the file is a CSV
#             file_path = os.path.join(directory_path, file)  # Get the full path of the file
#             try:
#                 df = pd.read_csv(file_path)  # Read the CSV file
#                # df['dataset'] = os.path.basename(file_path).replace('.csv', '')  # Add the dataset name
#                 print( os.path.basename(file_path).replace('.csv', ''))
#                 dataframes.append(df)  # Append the DataFrame to the list
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#
# # Combine all DataFrames into a single DataFrame
# if dataframes:
#     combined_df = pd.concat(dataframes, ignore_index=True)
#
#     # Save the combined DataFrame with all data to a CSV file
#     all_data_output_path =  '/mnt/c/Users/jamalids/Downloads/dataset/HPC/block2/combine.csv'
#     combined_df.to_csv(all_data_output_path, index=False)
#     print(f'Combined CSV with all data saved to {all_data_output_path}')
###################################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/mnt/c/Users/jamalids/Downloads/dataset/HPC/block2/combine.csv')

# Define parameters
k_values = [2, 3, 4]
scenarios = ["Frequency", "Entropy", "Delta", "All_Features"]

# Generate plots
for k in k_values:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(f'Compression Ratios for k={k}', fontsize=14)
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios):
        subset = df[(df['k'] == k) & (df['FeatureScenario'] == scenario)]
        subset = subset.sort_values('BlockIdx')

        ax.plot(subset['BlockIdx'], subset['StandardRatio'], label='StandardRatio')
        ax.plot(subset['BlockIdx'], subset['DecomposedRatio_ColOrder'], label='DecomposedRatio_ColOrder')
        #ax.plot(subset['BlockIdx'], subset['DecomposedRatio_RowOrder'], label='DecomposedRatio_RowOrder')

        ax.set_title(scenario)
        ax.set_xlabel('Block Index')
        ax.set_yscale('log')
        ax.set_ylabel('log(Ratio)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/mnt/c/Users/jamalids/Downloads/{k}_1.png")
