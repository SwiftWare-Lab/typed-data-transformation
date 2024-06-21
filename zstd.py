#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import heapq
import os
import zstandard as zstd

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

def compress_with_zstd(bit_data):
    # Convert bit string to bytes
    byte_data = int(bit_data, 2).to_bytes((len(bit_data) + 7) // 8, byteorder='big')
    
    # Compress the byte data
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(byte_data)
    
    # Return the size of the compressed data in bits
    return len(compressed) * 8  # converting bytes to bits
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
    original_size_bits = df.apply(lambda row: len(row['Pattern'].replace('\n', '').replace(' ', '').replace('[', '').replace(']', '')), axis=1).sum()
    compressed_size_bits_huffman = df.apply(lambda row: len(row['Compressed Pattern']), axis=1).sum()

    # Compress patterns using Zstd
    patterns_str = ''.join(df['Pattern'].apply(lambda x: x.replace('\n', '').replace(' ', '').replace('[', '').replace(']', '')))
    compressed_size_bits_zstd = compress_with_zstd(patterns_str)

    file_name = os.path.basename(file_path).replace('.csv', '')
    file_name = file_name.replace('df', '')
    return file_name, decompression_match, original_size_bits, compressed_size_bits_huffman, compressed_size_bits_zstd

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
results_df = pd.DataFrame(results, columns=['n-m', 'Decompression Match', 'Original Size-Dict (bits)', 'Compressed Size-Huffman (bits)', 'Compressed Size-Zstd (bits)'])
results_df['Huffman Compression Ratio'] = results_df['Original Size-Dict (bits)'] / results_df['Compressed Size-Huffman (bits)']
results_df['Zstd Compression Ratio'] = results_df['Original Size-Dict (bits)'] / results_df['Compressed Size-Zstd (bits)']

# Custom sort function to extract the part of the filename after the underscore
def custom_sort_key(file_name):
    base, number = file_name.rsplit('_', 1)
    number = int(number.split('.')[0])
    return (base, number)

# Sort the results dataframe by the custom key
sorted_results_df = results_df.sort_values(by='n-m', key=lambda x: x.map(custom_sort_key))

# Display the sorted results
print(sorted_results_df)


# In[5]:


sorted_results_df


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming sorted_results_df is already defined
results_df = sorted_results_df

# Plotting the bar chart
plt.figure(figsize=(12, 8))

# Bar positions
bar_width = 0.25
r1 = range(len(results_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Bars
plt.bar(r1, results_df['Original Size-Dict (bits)'], color='blue', width=bar_width, edgecolor='grey', label='Original Size of dict (bits)')
plt.bar(r2, results_df['Compressed Size-Huffman (bits)'], color='green', width=bar_width, edgecolor='grey', label='Compressed Size-Huffman of dict (bits)')
plt.bar(r3, results_df['Compressed Size-Zstd (bits)'], color='red', width=bar_width, edgecolor='grey', label='Compressed Size-Zstd of dict (bits)')

# Labels and titles
plt.xlabel('m-n', fontweight='bold')
plt.ylabel('Size (bits)', fontweight='bold')
plt.title('Comparison of Original Size of Dictionary  and Compressed Size of dictionary (Synthetic)')
plt.xticks([r + bar_width for r in range(len(results_df))], results_df['n-m'], rotation=45)

# Adding the legend
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.show()


# In[ ]:




