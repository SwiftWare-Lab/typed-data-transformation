#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# generate fbench dataset


# In[ ]:


import pandas as pd
import numpy as np
import os

# Define tasks and data types
tasks = ['rsim_f32', 'h3d_temp_f32','num_brain_f64','num_control_f64'
         ,'jw_mirimage_f32','hst_wfc3_uvis_f32','hst_wfc3_ir_f32','citytemp_f32','acs_wht_f32']
dts = ['f32', 'f32','f64','f64','f32','f32','f32','f32','f32']

# Define the data type dictionary
DT = {'f32': np.float32, 'f64': np.float64}

# Specify the directory containing the data
datadir = "/media/samira/sa/result-compression/data/"
datadir1 = "/media/samira/sa/result-compression/data1/"
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

