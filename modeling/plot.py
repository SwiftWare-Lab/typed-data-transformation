#!/usr/bin/env python
# coding: utf-8

# In[6]:


import heapq
from collections import Counter

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, other):
        return self.freq < other.freq

def calculate_frequency(array):
    frequency = Counter(array)
    return frequency

def build_huffman_tree(frequency):
    heap = [Node(freq, sym) for sym, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(node, current_code, codes):
    if node is None:
        return

    if node.symbol and len(node.symbol) == 1:
        codes[node.symbol] = current_code

    generate_codes(node.left, current_code + '0', codes)
    generate_codes(node.right, current_code + '1', codes)

def huffman_encoding(array):
    frequency = calculate_frequency(array)
    huffman_tree = build_huffman_tree(frequency)
    codes = {}
    generate_codes(huffman_tree, '', codes)
    return codes, huffman_tree

def huffman_decoding(encoded_array, huffman_tree):
    decoded_array = []
    current_node = huffman_tree
    for bit in encoded_array:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.left is None and current_node.right is None:
            decoded_array.append(current_node.symbol)
            current_node = huffman_tree

    return ''.join(decoded_array)

def calculate_compression_ratio(original, encoded):
    original_size = len(original) * 8  # size in bits (assuming 1 char = 8 bits)
    encoded_size = len(encoded)  # size in bits (since it's already in binary form)
    return original_size / encoded_size

# Example array of large numbers
array = [
    1782157574603483334432236702284489977579524597447836768937945686793606347614076428939539534,
    1909596775468984488628926958582913280008662659190308629244396785051713869910586955658872082,
    1973316375901734086836499325227927087809834883698884897095882929182119296015168810466946768
]

# Convert each number to a string and then to individual characters for frequency calculation
array_str = ''.join(map(str, array))
array_chars = list(array_str)

codes, huffman_tree = huffman_encoding(array_chars)
print("Huffman Codes:", codes)

# Encode the array using the Huffman codes
encoded_array = ''.join([codes[ch] for ch in array_chars])
print("Encoded Array:", encoded_array)

# Decode the encoded array
decoded_array = huffman_decoding(encoded_array, huffman_tree)
print("Decoded Array:", decoded_array)

# Split the decoded string back into the original large numbers
split_index = 0
decoded_numbers = []
for num in array:
    length = len(str(num))
    decoded_numbers.append(int(decoded_array[split_index:split_index + length]))
    split_index += length

print("Decoded Numbers:", decoded_numbers)

# Check if the original array and decoded array are the same
if array == decoded_numbers:
    print("The original and decoded arrays are the same.")
else:
    print("The original and decoded arrays are different.")

# Calculate and print the compression ratio
compression_ratio = calculate_compression_ratio(array_str, encoded_array)
print("Compression Ratio:", compression_ratio)


# In[2]:


merged_df


# In[ ]:




