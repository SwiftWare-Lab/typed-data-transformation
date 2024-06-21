#!/usr/bin/env python
# coding: utf-8

# In[89]:


import sys
import os
import pandas as pd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the dataset
#dataset_path = '/home/jamalids/Documents/2D/UCRArchive_2018/CinCECGTorso/CinCECGTorso_TRAIN.tsv'
dataset_path ='/home/jamalids/Documents/2D/SYN_new/matrix_10000.tsv'
#ts_data = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data = pd.read_csv(dataset_path, header=None)
#group = ts_data[(ts_data[0]==1)]

group = ts_data[(ts_data[0]==1)]

group = group.drop(columns=0)
group.fillna(0, inplace=True)
n_samples, n_timesteps = group.shape
print(f"n_signals, n_timesteps:,{n_samples, n_timesteps}")
# Plotting
plt.figure(figsize=(12, 6))

for i in range(n_samples):
    plt.plot(range(n_timesteps), group.iloc[i], label=f'Signal {i+1}')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Group 1 Signals with Time Step and Value (Synthetic)')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()


# In[81]:





# In[90]:


# Compute the count of same values at each timestep
same_value_count = group.apply(lambda x: x.value_counts().max(), axis=0)

# Plotting
plt.figure(figsize=(12, 6))

for i in range(n_samples):
    plt.plot(range(n_timesteps), group.iloc[i], label=f'Signal {i+1}')

# Plot the same value count data
plt.plot(range(n_timesteps), same_value_count, label='Same Value Count', color='black', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Count of Same Values')
plt.title('Group Signals with Same Value Count at Each Timestep')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()


# In[83]:


# Improved function to count repetitions in a signal
def count_repetitions(signal):
    value_counts = signal.value_counts()
    repetitions = value_counts[value_counts > 1].sum() - len(value_counts[value_counts > 1])
    return repetitions

# Compute the repetition count for each signal
repetition_counts = group.apply(count_repetitions, axis=1)

# Print repetition counts for verification
print(f"Repetition counts for each signal: {repetition_counts.values}")

# Plotting
plt.figure(figsize=(12, 6))

n_samples = group.shape[0]
n_timesteps = group.shape[1]

for i in range(n_samples):
    plt.plot(range(n_timesteps), group.iloc[i], label=f'Signal {i+1} (Reps: {repetition_counts.iloc[i]})')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Group 1 Signals with Repetition Counts')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()


# In[84]:


#dataset_path = '/home/jamalids/Documents/2D/UCR_RESULT/logUCR-Huff/output_CinCECGTorso_TRAIN.csv'
dataset_path = '/home/jamalids/Documents/2D/SYN_new/1.csv'

filtered_df = pd.read_csv(dataset_path)
pd.set_option('display.max_columns', None)
h=filtered_df[ (filtered_df['Feature Index'] ==1)]


# In[85]:


h


# In[87]:


df=h 
# Rename the column in the DataFrame
df.rename(columns={'base_ratio_dict': 'base-ratio with compressed dict'}, inplace=True)
df.rename(columns={'huffman_codes_size': 'huffman-Dictionary-Size'}, inplace=True)

df['m_n'] = df.apply(lambda row: f"({row['m']}, {row['n']})", axis=1)
# Pivot the data to create a matrix where:
# - the index (rows) are 'm' values
# - the columns are 'n' values
# - the cells contain the average or sum of 'Total Occurrences' or any other statistical summary
num_items = 7  # Five methods plus two ratios
bar_width = 0.12  # Width of the bars
indices = np.arange(len(df['m_n']))  # the x locations for the groups
plt.figure(figsize=(18, 8))  # Increased width to accommodate more space for x-ticks

# Create the first subplot for bar charts
ax = plt.subplot(111)
# List including both ratios and compression methods
items = ['Ideal Ratio', 'Base Ratio','base-ratio with compressed dict' ,'Snappy', 'FPZIP',  'zstd', 'LZ4']
colors = ['gold', 'silver', 'skyblue', 'orange', 'green', 'red', 'purple']  # Colors for each item

for i, item in enumerate(items):
    if item in ['Ideal Ratio', 'Base Ratio']:
        ax.bar(indices + i * bar_width, df[item], bar_width, label=item, color=colors[i])
    else:
        ax.bar(indices + i * bar_width, df[item], bar_width, label=item, color=colors[i])

# Set x-tick labels
ax.set_xticks(indices + bar_width * (num_items - 1) / 2)
ax.set_xticklabels(df['m_n'], rotation=45, ha='right')  # Rotate labels for better readability

# Adding labels, title, and legend
ax.set_xlabel('(m, n)')
ax.set_ylabel('Values')
ax.set_title('Comparison of Compression  Ratios by (m, n)_signal ')
ax.legend(loc='upper left', title="Methods and Ratios")

# Show plot
plt.show()


# In[91]:


# Calculate compressed size
df['Compressed Size'] = df['Total Occurrences'] * df['Size Per Pattern Bit Roundup']

# Combine m and n into a single column as a tuple (m, n)
df['m_n'] = df.apply(lambda row: f"({row['m']}, {row['n']})", axis=1)

# Set the number of items (methods plus ratios)
num_items = 7  # Excluding dictionary size and compressed size from bars
bar_width = 0.12  # Width of the bars
indices = np.arange(len(df['m_n']))  # the x locations for the groups

plt.figure(figsize=(18, 8))  # Increased width to accommodate more space for x-ticks

# Create the first subplot for bar charts
ax = plt.subplot(111)
# List including both ratios and compression methods
items = ['Ideal Ratio', 'Base Ratio','base-ratio with compressed dict' ,'Snappy', 'FPZIP',  'zstd', 'LZ4']
colors = ['gold', 'silver', 'skyblue', 'orange', 'green', 'red', 'purple']  # Colors for each item

for i, item in enumerate(items):
    ax.bar(indices + i * bar_width, df[item], bar_width, label=item, color=colors[i])

# Plot the dictionary size and compressed size as lines
ax2 = ax.twinx()
ax2.plot(indices + bar_width * (num_items - 1) / 2, df['Size Dictionary'], label='Dictionary Size', color='black', linewidth=2, marker='o')
ax2.plot(indices + bar_width * (num_items - 1) / 2, df['huffman-Dictionary-Size'], label='huffman-Dictionary-Size', color='brown', linewidth=2, linestyle='--', marker='x')

# Set x-tick labels
ax.set_xticks(indices + bar_width * (num_items - 1) / 2)
ax.set_xticklabels(df['m_n'], rotation=45, ha='right')  # Rotate labels for better readability

# Adding labels, title, and legend
ax.set_xlabel('(m, n)')
ax.set_ylabel('compresion Ratio')
ax2.set_ylabel('Dictionary Size / Compressed Size')
ax.set_title('Compression Ratios and Dictionary Size by (m, n)')

# Combine legends
bars, labels = ax.get_legend_handles_labels()
lines, line_labels = ax2.get_legend_handles_labels()
ax.legend(bars + lines, labels + line_labels, loc='upper left', title="Methods, Ratios, and Dictionary Size")

# Show plot
plt.show()


# In[ ]:




