import pandas as pd
import numpy as np
import os

# Define tasks and data types
tasks = ['tpcds_catalog_f32', 'tpcds_store_f32','tpcds_web_f32','tpch_lineitem_f32'
         ,'tpch_order_f64','tpcxbb_store_f64','tpcxbb_web_f64']
dts = ['f32', 'f32','f32','f32','f64','f64','f64']

# Define the data type dictionary
DT = {'f32': np.float32, 'f64': np.float64}

# Specify the directory containing the data
datadir = "/home/jamalids/Documents/2D/data1/tmp/"
datadir1 = "/home/jamalids/Documents/2D/data1/tmp/"
# Iterate over each task and corresponding data type
for K in range(len(tasks)):
    task = tasks[K]
    dt = dts[K]
    path_to_X = os.path.join(datadir, task)

    try:
        # Read data from file assuming it is stored in a binary format with the specified dtype
        X = np.fromfile(path_to_X, dtype=DT[dt])
        print(f"File read successfully for task {task}. Number of elements: {len(X)}")

        # Convert the NumPy array to a pandas DataFrame
        df = pd.DataFrame(X, columns=['Data'])
        #df1 = df.T
        #df1.insert(0, "feature_index", 1)

        # Save the DataFrame to a file named after the task
        file_path = os.path.join(datadir1, f"{task}.tsv")
        df.to_csv(file_path, sep='\t', header=None)
        print(f"DataFrame saved to {file_path}")

    except Exception as e:
        print(f"Failed to read the file for task {task}:", e)