import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results11.csv')

# Assigning a single dataset name for the whole dataset
df['dataset_name'] = 'ACSF1_TEST'

# Plot 1: Compression Ratio with and without Dictionary
plt.figure(figsize=(12, 6))

# Adjusting the bar width for comparison
bar_width = 0.1
index = range(len(df))

plt.bar(index, df['com_ratio'], bar_width, label='Com Ratio without Dictionary')
plt.bar([i + bar_width for i in index], df['com_ratio_dict'], bar_width, label='Com Ratio with Dictionary')

plt.title('Compression Ratio with and without Dictionary')
#plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio')
#plt.xticks([i + bar_width / 2 for i in index], df['dataset_name'])
plt.legend()
plt.show()

# Importing necessary library for logarithmic scale
import numpy as np

# Plot 2: Bar chart of Encoded and Dictionary Sizes with logarithmic scale
plt.figure(figsize=(12, 6))

plt.bar(index, np.log(df['l_encoded_size']), bar_width, label='Leading Encoded Size')
plt.bar([i + bar_width for i in index], np.log(df['l_dic_size_bits']), bar_width, label='Leading Dictonary Size Bits')
plt.bar([i + 2 * bar_width for i in index], np.log(df['t_encoded_size']), bar_width, label='Traling Encoded Size')
plt.bar([i + 3 * bar_width for i in index], np.log(df['t_dic_size_bits']), bar_width, label='Traling Dictionary Size Bits')
plt.bar([i + 4 * bar_width for i in index], np.log(df['c_encoded_size']), bar_width, label='Content Encoded Size')
plt.bar([i + 5 * bar_width for i in index], np.log(df['c_dic_size_bits']), bar_width, label='Content Dictionary Size Bits')
plt.bar([i + 6 * bar_width for i in index], np.log(df['tot_encoded_size']), bar_width, label='Normal Encoded Size')
plt.bar([i + 7 * bar_width for i in index], np.log(df['tot_dic_size_bits']), bar_width, label='Normal Dictionary Size Bits')

plt.title('Encoded Sizes and Dictionary Sizes (Log Scale)')
#plt.xlabel('Dataset Name')
plt.ylabel('Log Size (bits)')
plt.xticks([i + 1.5 * bar_width for i in index], df['dataset_name'])
plt.legend()

plt.tight_layout()
plt.show()

