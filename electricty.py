#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd

# Step 1: Read the data
dataset_path = "/home/samira/Desktop/LD2011_2014.txt"
ts_data = pd.read_csv(dataset_path, delimiter=';', header=None)

# Step 2: Replace commas with periods
ts_data = ts_data.replace(',', '.', regex=True)
ts_data = ts_data.iloc[:,1:]
# Step 3: Transpose the data
ts_data = ts_data.T
column_to_modify = ts_data.iloc[:, 1]

# Removing "MT_" prefix from values in the column
column_to_modify = column_to_modify.str.replace("MT_", "")
ts_data=ts_data.iloc[:, 1:]
# Assigning the modified column back to the DataFrame
ts_data.iloc[:, 1] = column_to_modify
# Assuming the first row contains the column headers, reset the index and set the first row as the new headers
#ts_data.columns = ts_data.iloc[0]
#ts_data = ts_data.iloc[1:]

# Step 4: Split the DataFrame into 10 parts and add signal column
num_parts = 37
rows_per_part = ts_data.shape[0] // num_parts

output_dir = "/home/samira/Documents/output2/"
for i in range(num_parts):
    start_row = i * rows_per_part
    end_row = min((i + 1) * rows_per_part, ts_data.shape[0])  # Adjust end_row to handle uneven division
    print(start_row,end_row)
    part_df = ts_data.iloc[start_row:end_row,:]
    
    
    # Save each part
    part_file_path = f"{output_dir}LD2011_2014_part_{i+1}.tsv"
    # Save DataFrame to a TSV file
           
    part_df.to_csv(part_file_path, sep='\t',header=None)

print("Files have been saved successfully.")

####################
import pandas as pd
dataset_path = "/home/samira/Documents/output2/LD2011_2014_part_5.tsv"
ts_data = pd.read_csv(dataset_path, delimiter='\t',header=None)


# In[ ]:




