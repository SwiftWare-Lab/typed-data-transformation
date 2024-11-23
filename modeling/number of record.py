import os
import pandas as pd

# Define the dataset path
dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/High-Entropy/"

# List all datasets with .tsv extension
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]

# Initialize a list to store results
results = []

# Iterate over each dataset
for dataset_path in datasets:
    # Extract dataset name and load data
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

    # Number of records (rows) in the dataset
    num_records = ts_data1.shape[0]

    # Append the dataset name and number of records to results
    results.append({"Dataset Name": dataset_name, "Number of Records": num_records})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
results_df.to_csv("/home/jamalids/Documents/2D/data1/Fcbench/High-Entropy/H-Number.csv")
# Display the resulting DataFrame
print(results_df)
