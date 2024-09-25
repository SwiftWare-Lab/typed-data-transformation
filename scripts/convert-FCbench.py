import pandas as pd
import numpy as np
import os

# Define tasks and data types
tasks = ['astro_mhd_f64', 'astro_pt_f64','g24_78_usb2_f32','hdr_night_f32'
         ,'hdr_palermo_f32','jane_street_f64','miranda3d_f32','msg_bt_f64',
         'nyc_taxi2015_f64','phone_gyro_f64','solar_wind_f32','spain_gas_price_f64',
         'spitzer_irac_f32','ts_gas_f32','turbulence_f32','wave_f32','wesad_chest_f64']
dts = ['f64', 'f64','f32','f32',
       'f32','f64','f32','f64'
       ,'f64','f64','f32','f64'
       ,'f32','f32','f32','f32','f64']

# Define the data type dictionary
DT = {'f32': np.float32, 'f64': np.float64}

# Specify the directory containing the data
datadir = "/home/jamalids/Documents/2D/data1/all data/FCBench/"
datadir1 = "/home/jamalids/Documents/2D/data1/all data/FCBench/"
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