#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
folder_path = '/home/samira/Documents/git2/logUCR-C/'

# Merge CSV files into a DataFrame
merged_df = merge_csv_files(folder_path)
file_path = '/home/samira/Documents/git2/2.csv'
merged_df =merged_df.dropna(inplace=True)
merged_df['rate_base_vs_zstd'] = merged_df['Base Ratio'] / merged_df['zstd']
merged_df['rate_base_vs_lz4'] = merged_df['Base Ratio'] / merged_df['LZ4']
      


# In[1]:


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
folder_path = '/home/samira/Documents/git2/logUCR-C/'

# Merge CSV files into a DataFrame
merged_df = merge_csv_files(folder_path)

# Remove rows with any NaN values
merged_df.dropna(inplace=True)

# Add columns for rate_base_vs_zstd and rate_base_vs_lz4
merged_df['rate_base_vs_zstd'] = merged_df['Base Ratio'] / merged_df['zstd']
merged_df['rate_base_vs_lz4'] = merged_df['Base Ratio'] / merged_df['LZ4']
merged_df['rate_base_vs_Snappy'] = merged_df['Base Ratio'] / merged_df['Snappy']
merged_df['rate_base_vs_FPZIP'] = merged_df['Base Ratio'] / merged_df['FPZIP']
merged_df['rate_base_vs_Gorilla'] = merged_df['Base Ratio'] / merged_df['Gorilla']


# Path to save the merged DataFrame
file_path = '/home/samira/Documents/git2/2.csv'

# Save the DataFrame to CSV
merged_df.to_csv(file_path, index=False)  # Set `index=False` if you do not want to include row indices in the CSV

# Optionally plot the data
# plot_data(merged_df)  # Uncomment and modify if necessary to plot the data


# In[2]:


merged_df


# In[57]:


# Show rows where rate_base_vs_zstd and rate_base_vs_lz4 > 1
filtered_df1 = merged_df[(merged_df['rate_base_vs_zstd'] > 1) & (merged_df['rate_base_vs_lz4'] >1 )&
                        
                        (merged_df['rate_base_vs_Snappy']>1) & ( merged_df['rate_base_vs_FPZIP']>1)&
                        (merged_df['rate_base_vs_Gorilla']>1)]
count_rate=filtered_df.shape[0]
print(f"Rows where 'rate_base_vs_all' > 3:,{count_rate}")


###################################################
filtered_df = merged_df[(merged_df['rate_base_vs_zstd'] > 1.1) & (merged_df['rate_base_vs_lz4'] > 1.1)]
count_rate1=filtered_df.shape[0]
print(f"Rows where 'rate_base_vs_zstd' and 'rate_base_vs_lz4' >1 1:,{count_rate1}")


# Count of rows where rate_base_vs_lz4 > 1
count_rate_base_vs_lz4_gt_1 = merged_df[merged_df['rate_base_vs_lz4'] > 1].shape[0]
print(f"Number of rows where 'rate_base_vs_lz4' > 1: {count_rate_base_vs_lz4_gt_1}")

# Count of rows where rate_base_vs_lz4 < 1
count_rate_base_vs_lz4_lt_1 = merged_df[merged_df['rate_base_vs_lz4'] < 1].shape[0]
print(f"Number of rows where 'rate_base_vs_lz4' < 1: {count_rate_base_vs_lz4_lt_1}")

##########################
# Count of rows where rate_base_vs_lz4 > 1
count_rate_base_vs_zstd_gt_1 = merged_df[merged_df['rate_base_vs_zstd'] > 1].shape[0]
print(f"Number of rows where 'rate_base_vs_zstd' > 1: {count_rate_base_vs_zstd_gt_1}")

# Count of rows where rate_base_vs_lz4 < 1
count_rate_base_vs_zstd_lt_1 = merged_df[merged_df['rate_base_vs_zstd'] < 1].shape[0]
print(f"Number of rows where 'rate_base_vs_zstd' < 1: {count_rate_base_vs_zstd_lt_1}")


# In[58]:


h=filtered_df[(filtered_df['Dataset Name'] != 'Phoneme_TRAIN') & (filtered_df['Dataset Name'] != 'PigArtPressure_TRAIN')& (filtered_df['Dataset Name'] != 'PigArtPressure_TEST')]


# In[59]:


h


# In[65]:


pd.set_option('display.max_columns', None)
df=merged_df[(merged_df['Feature Index']==4) & (merged_df['Dataset Name'] == 'Car_TRAIN')]
df.head(30)


# In[42]:


file_path = '/home/samira/Documents/git2/DodgerLoopDay_TEST.csv'

# Save the DataFrame to CSV
df.to_csv(file_path, index=False)


# In[13]:


df = merged_df[(merged_df['Dataset Name'] == 'Phoneme_TRAIN')]


# In[60]:


##### number and shape
import struct
import sys
import os
import pandas as pd
import numpy as np
dataset_path='/media/samira/sa/result-compression/UCRArchive_2018/Car/Car_TRAIN.tsv'
ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
groups = ts_data.groupby(0)
dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
for group_id, group in groups:
    group = group.drop(columns=0)
    group.fillna(0, inplace=True)
    
    n_samples, n_timesteps = group.shape
    print(f"Group {group_id}: {n_samples} rows, {n_timesteps} columns")


# In[ ]:




