#!/usr/bin/env python
# coding: utf-8

# In[53]:


import os
import pandas as pd
import matplotlib.pyplot as plt

def merge_csv_files(folder_path):
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
    # Initialize an empty DataFrame to store all data
    merged_df = pd.DataFrame()
    
    # Read each CSV file and append its contents to the merged DataFrame
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        merged_df = merged_df.append(df, ignore_index=True)
    
    return merged_df

def plot_data(data):
    # Assuming the DataFrame has columns 'x' and 'y', change these names accordingly
    plt.plot(data['x'], data['y'])
    plt.xlabel('X-axis label')
    plt.ylabel('Y-axis label')
    plt.title('Plot Title')
    plt.grid(True)
    plt.show()

# Path to the folder containing CSV files
folder_path = '/media/samira/sa/result-compression/logF-B/'

# Merge CSV files into a DataFrame
merged_data = merge_csv_files(folder_path)

file_path = '/media/samira/sa/result-compression/fina_RES_f.csv'

    # Save the DataFrame to CSV
merged_data.to_csv(file_path, index=False)  # Set `index=False` if you do not want to include row indices in the CSV


# In[54]:


merged_data


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/jw_mirimage_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for jw_mirimage_f32')
plt.xlabel('timesteps')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/data1/rsim_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for rsim_f32')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/data1/num_brain_f64.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for num_brain_f64')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/data1/hst_wfc3_uvis_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for hst_wfc3_uvis_f32')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/data1/h3d_temp_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for h3d_temp_f32')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[47]:


dataset_path = "/media/samira/sa/result-compression/logF-B/combineResult.csv"

# Load the data
results = pd.read_csv(dataset_path)


# In[48]:


results


# In[58]:


import pandas as pd
import matplotlib.pyplot as plt


# Plotting the compression ratios for different algorithms based on Dataset Name
plt.figure(figsize=(14, 8))

# Bar width
bar_width = 0.1

# Positions of the bars on the x-axis
r1 = range(len(merged_data['Dataset Name']))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]
r7 = [x + bar_width for x in r6]
r8 = [x + bar_width for x in r7]

# Creating the bar plot
plt.bar(r1, merged_data['Ideal Ratio'], color='blue', width=bar_width, edgecolor='grey', label='Ideal Ratio')
plt.bar(r2, merged_data['Base Ratio'], color='green', width=bar_width, edgecolor='grey', label='Base Ratio')
plt.bar(r3, merged_data['Lookup Ratio'], color='red', width=bar_width, edgecolor='grey', label='Lookup Ratio')
plt.bar(r4, merged_data['Snappy'], color='cyan', width=bar_width, edgecolor='grey', label='Snappy')
plt.bar(r5, merged_data['FPZIP'], color='magenta', width=bar_width, edgecolor='grey', label='FPZIP')
plt.bar(r6, merged_data['Gorilla'], color='yellow', width=bar_width, edgecolor='grey', label='Gorilla')
plt.bar(r7, merged_data['zstd'], color='black', width=bar_width, edgecolor='grey', label='zstd')
plt.bar(r8, merged_data['LZ4'], color='orange', width=bar_width, edgecolor='grey', label='LZ4')

# Adding labels and title
plt.xlabel('Dataset Name', fontweight='bold')
plt.ylabel('Compression Ratio', fontweight='bold')
plt.title('Compression Ratios for Different Algorithms by Dataset')

# Adding the x-ticks
plt.xticks([r + 4 * bar_width for r in range(len(merged_data['Dataset Name']))], merged_data['Dataset Name'], rotation=45, ha='right')

plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# In[63]:





# In[68]:


import pandas as pd
import matplotlib.pyplot as plt


# Remove rows where the 'Dataset Name' is 'acs_wht_f32' or 'rsim_f32'
merged_data = merged_data[(merged_data['Dataset Name'] != 'acs_wht_f32')
                          & (merged_data['Dataset Name'] != 'rsim_f32')& (merged_data['Dataset Name'] != 'hst_wfc3_uvis_f32')]

# Define the columns
dataset_name_column = 'Dataset Name'
other_columns = ['Ideal Ratio', 'Base Ratio', 'Lookup Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']

# Check if the other columns exist in the dataset
missing_columns = [col for col in other_columns if col not in merged_data.columns]
if missing_columns:
    print(f"The following columns are missing from the dataset: {missing_columns}")
else:
    plt.figure(figsize=(14, 8))
    for col in other_columns:
        plt.plot(merged_data[dataset_name_column], merged_data[col], label=col, marker='o')
    plt.xlabel('Dataset Name', fontweight='bold')
    plt.ylabel('Compression Ratio', fontweight='bold')
    plt.title('Compression Ratios for Various Algorithms by Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[95]:


import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data_exclude(data, dataset_names):
    return data[~data['Dataset Name'].isin(dataset_names)]

def check_missing_columns(data, columns):
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(f"The following columns are missing from the dataset: {missing_columns}")
    return missing_columns

def plot_compression_ratios(data, dataset_name_column, other_columns, ax):
    for col in other_columns:
        ax.plot(data[dataset_name_column], data[col], label=col, marker='o')
    ax.set_xlabel('Dataset Name', fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontweight='bold')
    ax.set_title('Compression Ratios for Various Algorithms by Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True)

def plot_time_series(data_path, title, ax):
    ts_data = pd.read_csv(data_path, delimiter='\t', header=None)
    ts_data = ts_data.T
    ts_data = ts_data.iloc[1, :50000]  # Assuming you want to plot the first 50,000 columns
    ax.plot(ts_data.values.flatten())
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Values')

# Define file paths
compression_data_path = '/media/samira/sa/result-compression/fina_RES_f.csv'
ts_data_paths = [
    "/media/samira/sa/result-compression/data1/citytemp_f32.tsv",
    "/media/samira/sa/result-compression/data1/h3d_temp_f32.tsv",
    "/media/samira/sa/result-compression/data1/hst_wfc3_ir_f32.tsv",
    "/media/samira/sa/result-compression/data1/jw_mirimage_f32.tsv",
    "/media/samira/sa/result-compression/data1/num_brain_f64.tsv",
    "/media/samira/sa/result-compression/data1/num_control_f64.tsv"
]

# Load and handle compression data
compression_data = load_data(compression_data_path)
excluded_datasets = ['acs_wht_f32', 'rsim_f32', 'hst_wfc3_uvis_f32']
filtered_compression_data = filter_data_exclude(compression_data, excluded_datasets)
dataset_name_column = 'Dataset Name'
other_columns = ['Ideal Ratio', 'Base Ratio', 'Lookup Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']
missing = check_missing_columns(filtered_compression_data, other_columns)

# Create a large figure to hold the subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 20))

if not missing:
    plot_compression_ratios(filtered_compression_data, dataset_name_column, other_columns, axes[0, 0])
    axes[0, 1].set_visible(False)  # Hide the second plot in the first row

# Plot time series data
for i, ts_path in enumerate(ts_data_paths):
    plot_time_series(ts_path, f'Time Series {i + 1}', axes[(i//2)+1, i%2])

plt.tight_layout()
plt.show()


# In[83]:


import pandas as pd
import matplotlib.pyplot as plt

file_path = '/media/samira/sa/result-compression/fina_RES_f.csv'

    # Save the DataFrame to CSV
merged_data1= pd.read_csv(file_path )
filtered_data = merged_data1[(merged_data1['Dataset Name'] == 'acs_wht_f32') |
                            (merged_data1['Dataset Name'] == 'rsim_f32') |
                            (merged_data1['Dataset Name'] == 'hst_wfc3_uvis_f32')]

# Remove rows where the 'Dataset Name' is 'acs_wht_f32' or 'rsim_f32'
merged_data = filtered_data
# Define the columns
dataset_name_column = 'Dataset Name'
other_columns = ['Ideal Ratio', 'Base Ratio', 'Lookup Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']

# Check if the other columns exist in the dataset
missing_columns = [col for col in other_columns if col not in merged_data.columns]
if missing_columns:
    print(f"The following columns are missing from the dataset: {missing_columns}")
else:
    plt.figure(figsize=(14, 8))
    for col in other_columns:
        plt.plot(merged_data[dataset_name_column], merged_data[col], label=col, marker='o')
    plt.xlabel('Dataset Name', fontweight='bold')
    plt.ylabel('Compression Ratio', fontweight='bold')
    plt.title('Compression Ratios for Various Algorithms by Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
dataset_path = "/media/samira/sa/result-compression/data1/h3d_temp_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for h3d_temp_f32')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()

dataset_path = "/media/samira/sa/result-compression/data1/rsim_f32.tsv"

# Load the data
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

# Transpose the data
ts_data = ts_data1.T

# Limit the data to the first 50,000 columns
ts_data = ts_data.iloc[1, 0:50000]
tensor = ts_data.to_numpy()
plt.plot(tensor)

plt.title('Patterns in Data Rows for rsim_f32')
plt.xlabel('Timestep')
plt.ylabel('Values')
plt.legend()
plt.show()


# In[84]:


merged_data


# In[91]:


import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data(data, dataset_names):
    return data[data['Dataset Name'].isin(dataset_names)]

def check_missing_columns(data, columns):
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(f"The following columns are missing from the dataset: {missing_columns}")
    return missing_columns

def plot_compression_ratios(data, dataset_name_column, other_columns, subplot_position):
    plt.subplot(2, 1, subplot_position)
    for col in other_columns:
        if col in data.columns:
            plt.plot(data[dataset_name_column], data[col], label=col, marker='o')
    plt.xlabel('Dataset Name', fontweight='bold')
    plt.ylabel('Compression Ratio', fontweight='bold')
    plt.title('Compression Ratios for Various Algorithms by Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)

def plot_time_series(data_path, title, subplot_position):
    ts_data = pd.read_csv(data_path, delimiter='\t', header=None)
    ts_data = ts_data.T
    ts_data = ts_data.iloc[1, 0:50000]
    tensor = ts_data.to_numpy()
    plt.subplot(2, 2, subplot_position)
    plt.plot(tensor)
    plt.title(title)
    plt.xlabel('Timestep')
    plt.ylabel('Values')
    plt.legend()

# Define file paths
compression_data_path = '/media/samira/sa/result-compression/fina_RES_f.csv'
ts_data_path_h3d = "/media/samira/sa/result-compression/data1/hst_wfc3_uvis_f32.tsv"
ts_data_path_rsim = "/media/samira/sa/result-compression/data1/rsim_f32.tsv"

# Load and handle compression data
compression_data = load_data(compression_data_path)
filtered_compression_data = filter_data(compression_data, ['hst_wfc3_uvis_f32', 'rsim_f32'])
dataset_name_column = 'Dataset Name'
other_columns = ['Ideal Ratio', 'Base Ratio', 'Lookup Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']
missing = check_missing_columns(filtered_compression_data, other_columns)

# Create a large figure to hold the subplots
plt.figure(figsize=(14, 12))

if not missing:
    plot_compression_ratios(filtered_compression_data, dataset_name_column, other_columns, 1)

# Plot time series data on the second row, split into two
plot_time_series(ts_data_path_h3d, 'Patterns in Data Rows for hst_wfc3_uvis_f32', 3)
plot_time_series(ts_data_path_rsim, 'Patterns in Data Rows for rsim_f32', 4)

plt.tight_layout()
plt.show()


# In[96]:


import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_data_exclude(data, dataset_names):
    return data[~data['Dataset Name'].isin(dataset_names)]

def check_missing_columns(data, columns):
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(f"The following columns are missing from the dataset: {missing_columns}")
    return missing_columns

def plot_compression_ratios(data, dataset_name_column, other_columns, ax):
    for col in other_columns:
        ax.plot(data[dataset_name_column], data[col], label=col, marker='o')
    ax.set_xlabel('Dataset Name', fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontweight='bold')
    ax.set_title('Compression Ratios for Various Algorithms by Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True)

def plot_time_series(data_path, dataset_name, ax):
    ts_data = pd.read_csv(data_path, delimiter='\t', header=None)
    ts_data = ts_data.T
    ts_data = ts_data.iloc[1, :50000]  # Assuming you want to plot the first 50,000 columns
    ax.plot(ts_data.values.flatten())
    ax.set_title(dataset_name)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Values')

# Define file paths and dataset names
compression_data_path = '/media/samira/sa/result-compression/fina_RES_f.csv'
dataset_paths = {
    "City Temperature": "/media/samira/sa/result-compression/data1/citytemp_f32.tsv",
    "H3D Temperature": "/media/samira/sa/result-compression/data1/h3d_temp_f32.tsv",
    "HST WFC3 IR": "/media/samira/sa/result-compression/data1/hst_wfc3_ir_f32.tsv",
    "JWST MIRI Image": "/media/samira/sa/result-compression/data1/jw_mirimage_f32.tsv",
    "Numerical Brain": "/media/samira/sa/result-compression/data1/num_brain_f64.tsv",
    "Numerical Control": "/media/samira/sa/result-compression/data1/num_control_f64.tsv"
}

# Load and handle compression data
compression_data = load_data(compression_data_path)
excluded_datasets = ['acs_wht_f32', 'rsim_f32', 'hst_wfc3_uvis_f32']
filtered_compression_data = filter_data_exclude(compression_data, excluded_datasets)
dataset_name_column = 'Dataset Name'
other_columns = ['Ideal Ratio', 'Base Ratio', 'Lookup Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']
missing = check_missing_columns(filtered_compression_data, other_columns)

# Create a large figure to hold the subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 20))

if not missing:
    plot_compression_ratios(filtered_compression_data, dataset_name_column, other_columns, axes[0, 0])
    axes[0, 1].set_visible(False)  # Hide the second plot in the first row

# Plot time series data
for (dataset_name, path), ax in zip(dataset_paths.items(), axes.ravel()[2:]):
    plot_time_series(path, dataset_name, ax)

plt.tight_layout()
plt.show()


# In[ ]:




