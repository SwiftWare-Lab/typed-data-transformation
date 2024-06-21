#!/usr/bin/env python
# coding: utf-8

# In[157]:



#df=filtered_df.iloc[0:99285,:]


# In[ ]:




row_counts = [
    256, 5567, 4035, 3065, 2484, 2093, 1813, 1604, 1434, 1308,
    4841, 3817, 2647, 1997, 1599, 1337, 1143, 1003, 891, 803,
    4004, 2002, 1336, 1002, 802, 669, 572, 502, 446, 402
]

row_counts = [
    52, 622, 1020, 898, 783, 710, 653, 616, 573, 533,
    66, 737, 1057, 844, 679, 565, 485, 425, 378, 340,
    66, 737, 1057, 844, 679, 565, 485, 425, 378, 340
]
row_counts = [256,58524,100527,77415,64217,55413,48875,43762,39687,36325,51907,
94017,66651,50001,40000,33335,28573,25001,22225,20000,99830,
50000,33335,25001,20000,16668,14287,12501,11113,10000]


# In[203]:


import pandas as pd
dataset_path="/home/jamalids/Documents/2D/SYN_new/2.csv"
filtered_df = pd.read_csv(dataset_path) 
df=filtered_df

# Provided row counts
row_counts = [256,58524,100527,77415,64217,55413,48875,43762,39687,36325,51907,
94017,66651,50001,40000,33335,28573,25001,22225,20000,99830,
50000,33335,25001,20000,16668,14287,12501,11113,10000]


# Names for the DataFrames
base_names = ['df8_', 'df16_', 'df32_']
dataframes = {}

# Current index to keep track of the starting row
current_index = 0

# Create DataFrames based on the provided counts
for base_name in base_names:
    for i in range(1, 11):
        # Get the number of rows for this DataFrame
        num_rows = row_counts.pop(0)
        
        # Create the DataFrame
        dataframes[f"{base_name}{i}"] = df.iloc[current_index:current_index + num_rows].reset_index(drop=True)
        
        # Update the current index
        current_index += num_rows

# Display the counts of each DataFrame to verify
result_counts = {name: df.shape[0] for name, df in dataframes.items()}
print(result_counts)

# Save the DataFrames to separate CSV files
for name, df in dataframes.items():
    df.to_csv(f"/home/jamalids/Documents/2D/SYN_new/df/{name}.csv", index=False)


# In[204]:


df


# In[198]:


# Load the CSV file
file_path = "/home/jamalids/Documents/2D/df/df8_1.csv"
df = pd.read_csv(file_path)
# Create the frequency dictionary based on the 'Occurrences' column
frequencies = dict(zip(df['Pattern'], df['Occurrences']))

# Define the HuffmanNode class and the functions to build the Huffman tree
class HuffmanNode:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return f"{self.left}_{self.right}"

def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Build the Huffman tree
huffman_tree = build_huffman_tree(frequencies)

# Generate the Huffman codes
huffman_codes = {item[0]: item[1] for item in huffman_tree}
reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}

# Replace patterns in the dataframe with their corresponding Huffman codes
df['Compressed Pattern'] = df['Pattern'].map(huffman_codes)

# Function to decompress the patterns
def decompress_pattern(compressed_pattern):
    return reverse_huffman_codes[compressed_pattern]

# Decompress the patterns
df['Decompressed Pattern'] = df['Compressed Pattern'].apply(decompress_pattern)

# Check if decompressed data matches original data
decompression_match = df['Pattern'].equals(df['Decompressed Pattern'])

# Measure the sizes of original and compressed patterns in bits
original_size_bits = df.apply(lambda row: len(row['Pattern'].replace('\n', '').replace(' ', '').replace('[', '').replace(']', '')) , axis=1).sum()
compressed_size_bits = df.apply(lambda row: len(row['Compressed Pattern']) , axis=1).sum()

(decompression_match, original_size_bits, compressed_size_bits, df.head())


# In[114]:


from __future__ import annotations

import sys
from typing import Tuple


class Node:
  def __init__(self, char, freq):
    self.char = char
    self.freq = freq
    self.left = None
    self.right = None

  def __lt__(self, other):
    return self.freq < other.freq

def huffman_codes(freq_dict):
  """
  Computes Huffman codes for a given frequency distribution.

  Args:
      freq_dict: A dictionary mapping characters to their frequencies.

  Returns:
      A dictionary mapping characters to their corresponding Huffman codes.
  """
  nodes = [Node(char, freq) for char, freq in freq_dict.items()]
  while len(nodes) > 1:
    # Use min heap for efficient retrieval of nodes with lowest frequencies
    import heapq
    heapq.heapify(nodes)
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)
    root = Node(None, left.freq + right.freq)
    root.left = left
    root.right = right
    heapq.heappush(nodes, root)

  codes = {}
  curr_code = ""
  def traverse(node, code):
    nonlocal curr_code, codes
    if node is None:
      return
    if node.char:
      codes[node.char] = code
      return
    traverse(node.left, code + "0")
    traverse(node.right, code + "1")
  traverse(nodes[0], "")
  return codes

frequencies = dict(zip(df['Pattern'], df['Occurrences']))

huffman_code = huffman_codes(frequencies)


# In[212]:


import pandas as pd
import heapq
import os

# Define the HuffmanNode class and functions to build the Huffman tree
class HuffmanNode:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return f"{self.left}_{self.right}"

def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def process_file(file_path):
    df = pd.read_csv(file_path)
    frequencies = dict(zip(df['Pattern'], df['Occurrences']))
    
    
    # Build the Huffman tree
    huffman_tree = build_huffman_tree(frequencies)
    
    # Generate the Huffman codes
    huffman_codes = {item[0]: item[1] for item in huffman_tree}
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    
    # Replace patterns in the dataframe with their corresponding Huffman codes
    df['Compressed Pattern'] = df['Pattern'].map(huffman_codes)
    
    # Function to decompress the patterns
    def decompress_pattern(compressed_pattern):
        return reverse_huffman_codes[compressed_pattern]
    
    # Decompress the patterns
    df['Decompressed Pattern'] = df['Compressed Pattern'].apply(decompress_pattern)
    
    # Check if decompressed data matches original data
    decompression_match = df['Pattern'].equals(df['Decompressed Pattern'])
    
    # Measure the sizes of original and compressed patterns in bits
    original_size_bits = df.apply(lambda row: len(row['Pattern'].replace('\n', '').replace(' ', '').replace('[', '').replace(']', '')) , axis=1).sum()
    compressed_size_bits = df.apply(lambda row: len(row['Compressed Pattern']) , axis=1).sum()
    
    compressed_dict_size = len(huffman_codes) * 8  # assuming 1 byte per character in the symbol
    file_name = os.path.basename(file_path).replace('.csv', '')
    file_name=file_name.replace('df', '')
    return file_name, decompression_match, original_size_bits, compressed_size_bits

# Folder containing the CSV files
folder_path = "/home/jamalids/Documents/2D/SYN_new/df/"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Process each file and store the results
results = []
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    result = process_file(file_path)
    
    results.append(result)

# Create a dataframe to show the results
results_df = pd.DataFrame(results, columns=['n-m', 'Decompression Match', 'Original Size-Dict (bits)', 'Compressed Size-Dict (bits)'])
results_df['rate_ compression of Dict'] = results_df['Original Size-Dict (bits)'] / results_df['Compressed Size-Dict (bits)']
# Custom sort function to extract the part of the filename after the underscore
def custom_sort_key(file_name):
    base, number = file_name.rsplit('_', 1)
    number = int(number.split('.')[0])
    return (base, number)

# Sort the results dataframe by the custom key
sorted_results_df = results_df.sort_values(by='n-m', key=lambda x: x.map(custom_sort_key))


# In[214]:


sorted_results_df
sorted_results_df.to_csv("/home/jamalids/Documents/2D/SYN_new/result-Dict.csv")


# In[207]:


results_df=sorted_results_df
# Plotting the bar chart
plt.figure(figsize=(12, 8))

# Bar positions
bar_width = 0.35
r1 = range(len(results_df))
r2 = [x + bar_width for x in r1]

# Bars
plt.bar(r1, results_df['Original Size (bits)'], color='blue', width=bar_width, edgecolor='grey', label='Original Size (bits)')
plt.bar(r2, results_df['Compressed Size (bits)'], color='green', width=bar_width, edgecolor='grey', label='Compressed Size (bits)')

# Labels and titles
plt.xlabel('m-n', fontweight='bold')
plt.ylabel('Size (bits)', fontweight='bold')
plt.title('Comparison of Original Size and Compressed Size(Coffee_TRAIN)')
plt.xticks([r + bar_width/2 for r in range(len(results_df))], results_df['n-m'], rotation=45)

# Adding the legend
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.show()


# In[209]:


results_df=sorted_results_df
# Plotting the bar chart
plt.figure(figsize=(12, 8))

# Bar positions
bar_width = 0.35
r1 = range(len(results_df))
r2 = [x + bar_width for x in r1]

# Bars
plt.bar(r1, results_df['Original Size (bits)'], color='blue', width=bar_width, edgecolor='grey', label='Original Size of Dict (bits)')
plt.bar(r2, results_df['Compressed Size (bits)'], color='green', width=bar_width, edgecolor='grey', label='Compressed Size of Dict (bits)')

# Labels and titles
plt.xlabel('m-n', fontweight='bold')
plt.ylabel('Size (bits)', fontweight='bold')
plt.title('Comparison of Original Size of Dictionary and Compressed Size of Dictionary( Synthetic)')
plt.xticks([r + bar_width/2 for r in range(len(results_df))], results_df['n-m'], rotation=45)

# Adding the legend
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.show()


# In[ ]:




