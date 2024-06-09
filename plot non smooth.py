#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the parameters for the synthetic data
num_rows = 10
num_columns = 100
total_points = num_rows * num_columns  # Total 1000 points
noise_level = 1.0
jump_chance = 0.1
jump_magnitude = 5

# Generate time points for each column (100 time points repeated for each row)
x = np.tile(np.linspace(0, 10, num_columns), num_rows)

# Start with a base linear trend, reshaped for each repeated sequence of x
y = 0.5 * x

# Introduce random noise
y_noisy = y + np.random.normal(0, noise_level, total_points)

# Introduce random jumps
for i in range(total_points):
    if np.random.rand() < jump_chance:
        y_noisy[i] += np.random.choice([-1, 1]) * jump_magnitude

# Reshape the data into a matrix of 10 rows and 100 columns
y_noisy_matrix = y_noisy.reshape(num_rows, num_columns)

# Create a DataFrame from the reshaped data
data = pd.DataFrame(y_noisy_matrix)

# Optionally save to CSV
data.to_csv('non_smooth_data_matrix.csv', index=False)

# Plotting all rows in a single plot
plt.figure(figsize=(10, 6))
for i in range(num_rows):
    plt.plot(np.linspace(0, 100, num_columns), data.iloc[i], label=f'Sample {i+1}')
plt.title('Synthetic Dataset with Non-Smooth Trend')
plt.xlabel('Timesteps')
plt.ylabel('Value')
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside the plot
plt.grid(True)
plt.show()
#############################
# Calculate the average of all samples at each timestep
average_values = data.mean(axis=0)

# Plotting all rows in a single plot with the average
plt.figure(figsize=(12, 7))
for i in range(num_rows):
    plt.plot(np.linspace(0, 100, num_columns), data.iloc[i], label=f'Sample {i+1}')
# Plot the average line, make it thicker and more prominent
plt.plot(np.linspace(0, 100, num_columns), average_values, label='Average', color='black', linewidth=2, linestyle='--')
plt.title('Synthetic Dataset with Non-Smooth Trend and Average')
plt.xlabel('Timesteps')
plt.ylabel('Value')
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside the plot
plt.grid(True)
plt.show()
#####################################################
# Calculate the average of all samples at each timestep
average_values = data.mean(axis=0)

# Plotting the average in a separate plot
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 100, num_columns), average_values, label='Average', color='red', linewidth=2)
plt.title('Average Trend Across All Samples')
plt.xlabel('Timesteps')
plt.ylabel('Average Value')
plt.legend()
plt.grid(True)
plt.show()
############################################################
ts_data1=data
ts_data1=ts_data1.insert(0, "feature_index", 1)
df=data
df['average_repeats_per_pattern'] = df['Total Occurrences'] / df['num_patterns']
df['m_n'] = df.apply(lambda row: f"({row['m']}, {row['n']})", axis=1)
data.to_csv('/home/jamalids/Documents/2D/non_smooth_data_matrix.tsv', index=False)


# In[17]:


ts_data1=data
ts_data1=ts_data1.insert(0, "feature_index", 1)


# In[31]:


data


# In[38]:


a = pd.read_csv('/home/jamalids/Documents/2D/non_smooth_result.csv')


# In[39]:


a


# In[60]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# Load your dataframe
# df = pd.read_csv('your_file.csv')
df=a
# Combine m and n into a single column as a tuple (m, n)
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
items = ['Ideal Ratio', 'Base Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']
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
ax.set_title('Comparison of Compression  Ratios by (m, n)_signal 1')
ax.legend(loc='upper left', title="Methods and Ratios")

# Show plot
plt.show()


# In[57]:


df = df[df['m_n'].isin(['(1, 8)', '(2, 8)', '(5, 8)', '(10, 8)', '(1, 16)', '(2, 16)', '(5, 16)', '(10, 16)', '(1, 32)', '(2, 32)', '(5, 32)', '(10, 32)'])]

num_items = 7  # Number of methods plus ratios
bar_width = 0.12  # Width of the bars
indices = np.arange(len(df['m_n']))  # the x locations for the groups

plt.figure(figsize=(18, 8))  # Increased width to accommodate more space for x-ticks

# Create the first subplot for bar charts
ax = plt.subplot(111)
# List including both ratios and compression methods
items = ['Ideal Ratio', 'Base Ratio', 'Snappy', 'FPZIP', 'Gorilla', 'zstd', 'LZ4']
colors = ['gold', 'silver', 'skyblue', 'orange', 'green', 'red', 'purple']  # Colors for each item

for i, item in enumerate(items):
    ax.bar(indices + i * bar_width, df[item], bar_width, label=item, color=colors[i])

# Set x-tick labels
ax.set_xticks(indices + bar_width * (num_items - 1) / 2)
ax.set_xticklabels(df['m_n'], rotation=45, ha='right')  # Rotate labels for better readability

# Adding labels, title, and legend
ax.set_xlabel('(m, n)')
ax.set_ylabel('Values')
ax.set_title('Comparison of Compression Ratios by (m, n)')
ax.legend(loc='upper left', title="Methods and Ratios")

# Show plot
plt.show()


# In[70]:


# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Create bars for Average Repetitions per Pattern
bar_width = 0.35  # width of the bars
indices = range(len(df['m_n']))  # the x locations for the groups

ax1.bar(indices, df['average_repeats_per_pattern'], bar_width, label='Average Repetitions per Pattern', color='b')
ax1.set_xlabel('(m, n) pairs')
ax1.set_ylabel('Average Repetitions', color='b')
ax1.set_title('Base Ratio and Average Repetitions per Pattern for each (m, n)')
ax1.set_xticks(indices)
ax1.set_xticklabels(df['m_n'], rotation=45)  # Rotate labels for better readability
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for the Base Ratio
ax2 = ax1.twinx()
ax2.plot(indices, df['Base Ratio'], color='r', label='Base Ratio', marker='o', linestyle='-')
ax2.set_ylabel('Base Ratio', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding a legend that combines elements from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.show()


# In[64]:


df['average_repeats_per_pattern'] = df['Total Occurrences'] / df['num_patterns']


# In[73]:


# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 8))  # Increased figure size for better readability

# Create bars for Total Occurrences
indices = range(len(df['m_n']))  # the x locations for the groups
bar_width = 0.35  # width of the bars
rects1 = ax1.bar(indices, df['Total Occurrences'], bar_width, label='Total Occurrences', color='b')

# Set x-axis and labels
ax1.set_xlabel('(m, n) pairs')
ax1.set_ylabel('Total Occurrences', color='b')
ax1.set_title('Total Occurrences and Base Ratio for each (m, n)')
ax1.set_xticks(indices)
ax1.set_xticklabels(df['m_n'], rotation=90)  # Rotate labels to 90 degrees for better readability
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for the Base Ratio
ax2 = ax1.twinx()
ax2.plot(indices, df['Base Ratio'], color='r', label='Base Ratio', marker='o', linestyle='-')
ax2.set_ylabel('Base Ratio', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')


# In[ ]:




