#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "/media/samira/sa/result-compression/results/UCR.csv"

# Load the data
ts_data1 = pd.read_csv(dataset_path)


# In[2]:


ts_data1


# In[15]:



dataset_path = "/media/samira/sa/result-compression/results/UCR-TENSOR.csv"

# Load the data
ts_data1 = pd.read_csv(dataset_path)
# Calculate the mean of compression ratios across all datasets, ignoring the feature index
overall_means = ts_data1.drop('Feature Index', axis=1).groupby('Dataset Name').mean()
# Increase figure size and adjust tick parameters
fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure size
overall_means.plot(ax=ax, marker='o', linestyle='-')
ax.set_title('Overall Average Compression Ratios Across Datasets(UCR-TENSOR)')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)
ax.legend(title='Compression Algorithm')

# Set x-ticks to show every dataset name and further rotate labels if necessary
ax.set_xticks(range(len(overall_means.index)))
ax.set_xticklabels(overall_means.index, rotation=90)  # Increased rotation for clarity

plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset_path = "/media/samira/sa/result-compression/results/UCR-TENSOR.csv"
ts_data = pd.read_csv(dataset_path)

# Calculate the mean of compression ratios across all compression algorithms, ignoring the feature index
overall_means = ts_data.drop('Feature Index', axis=1).mean()

# Increase figure size and adjust tick parameters for plot clarity
fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for visibility
overall_means.plot(ax=ax, kind='bar', color='skyblue')  # Using a bar plot for better visualization of averages
ax.set_title('Overall Average Compression Ratios for Compression Algorithms (UCR-TENSOR)')
ax.set_xlabel('Compression Algorithm')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)

plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset_path = "/media/samira/sa/result-compression/results/Electricty.csv"
ts_data = pd.read_csv(dataset_path)

# Calculate the mean of compression ratios across all compression algorithms, ignoring the feature index
overall_means = ts_data.drop('Feature Index', axis=1).mean()

# Increase figure size and adjust tick parameters for plot clarity
fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for visibility
overall_means.plot(ax=ax, kind='bar', color='skyblue')  # Using a bar plot for better visualization of averages
ax.set_title('Overall Average Compression Ratios for Compression Algorithms (Electricty)')
ax.set_xlabel('Compression Algorithm')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)

plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[34]:


dataset_path = "/media/samira/sa/result-compression/results/Electricty.csv"

# Load the data
ts_data1 = pd.read_csv(dataset_path)
# Calculate the mean of compression ratios across all datasets, ignoring the feature index
overall_means = ts_data1.drop('Feature Index', axis=1).groupby('Dataset Name').mean()
# Increase figure size and adjust tick parameters
fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure size
overall_means.plot(ax=ax, marker='o', linestyle='-')
ax.set_title('Overall Average Compression Ratios Across Datasets(Electricty)')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)
ax.legend(title='Compression Algorithm')

# Set x-ticks to show every dataset name and further rotate labels if necessary
ax.set_xticks(range(len(overall_means.index)))
ax.set_xticklabels(overall_means.index, rotation=90)  # Increased rotation for clarity

plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset_path = "/media/samira/sa/result-compression/results/Electricty.csv"

ts_data = pd.read_csv(dataset_path)

# Calculate the mean of compression ratios across all datasets, ignoring the feature index
overall_means = ts_data.drop('Feature Index', axis=1).groupby('Dataset Name').mean()

# Determine how many plots are needed
num_plots = 5
datasets_per_plot = len(overall_means) // num_plots + 1

# Create multiple plots
for i in range(num_plots):
    fig, ax = plt.subplots(figsize=(18, 8))
    subset = overall_means.iloc[i*datasets_per_plot:(i+1)*datasets_per_plot]
    subset.plot(ax=ax, marker='o', linestyle='-')
    ax.set_title(f'Average Compression Ratios Across Datasets - Group {i+1}')
    ax.set_xlabel('Dataset Name')
    ax.set_ylabel('Average Compression Ratio')
    ax.grid(True)
    ax.legend(title='Compression Algorithm')
    ax.set_xticks(range(len(subset.index)))
    ax.set_xticklabels(subset.index, rotation=90)
    plt.tight_layout()
    plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset_path = "/media/samira/sa/result-compression/results/FC-Bench.csv"
ts_data = pd.read_csv(dataset_path)

# Calculate the mean of compression ratios across all compression algorithms, ignoring the feature index
overall_means = ts_data.drop('Feature Index', axis=1).mean()

# Increase figure size and adjust tick parameters for plot clarity
fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for visibility
overall_means.plot(ax=ax, kind='bar', color='skyblue')  # Using a bar plot for better visualization of averages
ax.set_title('Overall Average Compression Ratios for Compression Algorithms (FC-Bench)')
ax.set_xlabel('Compression Algorithm')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)

plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[32]:


dataset_path = "/media/samira/sa/result-compression/results/FC-Bench.csv"

# Load the data
ts_data1 = pd.read_csv(dataset_path)
# Calculate the mean of compression ratios across all datasets, ignoring the feature index
overall_means = ts_data1.drop('Feature Index', axis=1).groupby('Dataset Name').mean()
# Increase figure size and adjust tick parameters
fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure size
overall_means.plot(ax=ax, marker='o', linestyle='-')
ax.set_title('Overall Average Compression Ratios Across Datasets(UCR-TENSOR)')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)
ax.legend(title='Compression Algorithm')

# Set x-ticks to show every dataset name and further rotate labels if necessary
ax.set_xticks(range(len(overall_means.index)))
ax.set_xticklabels(overall_means.index, rotation=90)  # Increased rotation for clarity

plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset_path = "/media/samira/sa/result-compression/results/UCR.csv"
ts_data = pd.read_csv(dataset_path)

# Calculate the mean of compression ratios across all compression algorithms, ignoring the feature index
overall_means = ts_data.drop('Feature Index', axis=1).mean()

# Increase figure size and adjust tick parameters for plot clarity
fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figure size for visibility
overall_means.plot(ax=ax, kind='bar', color='skyblue')  # Using a bar plot for better visualization of averages
ax.set_title('Overall Average Compression Ratios for Compression Algorithms (UCR)')
ax.set_xlabel('Compression Algorithm')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)

plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[33]:


dataset_path = "/media/samira/sa/result-compression/results/UCR.csv"

# Load the data
ts_data1 = pd.read_csv(dataset_path)
# Calculate the mean of compression ratios across all datasets, ignoring the feature index
overall_means = ts_data1.drop('Feature Index', axis=1).groupby('Dataset Name').mean()
# Increase figure size and adjust tick parameters
fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure size
overall_means.plot(ax=ax, marker='o', linestyle='-')
ax.set_title('Overall Average Compression Ratios Across Datasets(UCR-TENSOR)')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Average Compression Ratio')
ax.grid(True)
ax.legend(title='Compression Algorithm')

# Set x-ticks to show every dataset name and further rotate labels if necessary
ax.set_xticks(range(len(overall_means.index)))
ax.set_xticklabels(overall_means.index, rotation=90)  # Increased rotation for clarity

plt.tight_layout()  # Adjust layout to accommodate label size
plt.show()


# In[ ]:




