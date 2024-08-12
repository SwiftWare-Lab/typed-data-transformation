
import heapq  # for priority queue
import sys

import numpy as np

from utils import binary_to_int


class Node:
  """
  Node class for representing characters and their frequencies in the Huffman tree.
  """
  def __init__(self, char, freq):
    self.char = char
    self.freq = freq
    self.left = None
    self.right = None

  def __lt__(self, other):
    """
    Defines less-than comparison for priority queue (based on frequency).
    """
    return self.freq < other.freq

def create_huffman_tree(char_freq_map):

  nodes = [Node(char, freq) for char, freq in char_freq_map.items()]
  while len(nodes) > 1:
    # Get two nodes with the lowest frequencies
    min1, min2 = heapq.heappop(nodes), heapq.heappop(nodes)
    # Create a new parent node with combined frequency
    parent = Node(None, min1.freq + min2.freq)
    parent.left = min1
    parent.right = min2
    # Add the parent node back to the queue
    heapq.heappush(nodes, parent)
  return nodes[0]

def create_huffman_codes(node, code, codes):
  """
  Creates a dictionary mapping characters to their Huffman codes by traversing the tree.

  Args:
      node: The current node in the tree.
      code: The current code string for the path from the root to the current node.
      codes: A dictionary to store the character-code mappings.
  """
  if node is None:
    return
  if node.char is not None:
    codes[node.char] = code
  create_huffman_codes(node.left, code + '0', codes)
  create_huffman_codes(node.right, code + '1', codes)

  def binary_to_int(binary_list):
    """Convert a list of binary values (0/1) to an integer."""
    return int(''.join(str(int(b)) for b in binary_list), 2)


def encode_data(bool_array, m, n, huffman_codes):

  ts_m, ts_n = len(bool_array), len(bool_array[0])
  encoded_string = ""

  for i in range(0, ts_n, n):
    for j in range(0, ts_m, m):
      # Extract the m x n pattern
      rect = bool_array[j:j + m, i:i + n]
      rect_int = binary_to_int(rect)
      encoded_string += huffman_codes.get(rect_int, "")

  return encoded_string

def calculate_size_of_huffman_tree(node):
  """
  Calculate the total size of the Huffman tree in bytes.

  :param node: The root node of the Huffman tree.
  :return: The total size of the Huffman tree in bytes.
  """
  if node is None:
    return 0

  # Base size of the node itself
  size = sys.getsizeof(node)

  # Add the size of the character if it's a leaf node
  if node.char is not None:
    size += sys.getsizeof(node.char)

  # Add the size of left and right children recursively
  size += calculate_size_of_huffman_tree(node.left)
  size += calculate_size_of_huffman_tree(node.right)

  return size


def decode(encoded_text, root, m, n, original_shape):
  
  ts_m, ts_n = original_shape
  decoded_ints = []
  current_node = root

  # Step 1: Decode the encoded text back to integer patterns
  for bit in encoded_text:
    if bit == '0':
      current_node = current_node.left
    elif bit == '1':
      current_node = current_node.right

    # If a leaf node is reached, store the integer value and reset to root
    if current_node.char is not None:
      decoded_ints.append(current_node.char)
      current_node = root

  # Step 2: Convert decoded integers back to binary form
  decoded_binaries = [int_to_binary(value, m * n) for value in decoded_ints]
  # Step 3: Reconstruct the original 2D binary array
  binary_array = np.zeros(original_shape)
  pattern_idx = 0

  for j in range(0, ts_n, n):
    for i in range(0, ts_m, m):
      if pattern_idx >= len(decoded_binaries):
        break
      binary_pattern = decoded_binaries[pattern_idx]
      pattern_idx += 1

      # Fill the corresponding part of the array
      for bit_idx, bit in enumerate(binary_pattern):
        bit_j = bit_idx // n
        bit_i = bit_idx % n
        binary_array[i + bit_j, j + bit_i] = int(bit)

  return binary_array


def int_to_binary(n, length):
  """Convert an integer to a list of binary values (0/1) of a given length."""
  return list(map(int, list(bin(n)[2:].zfill(length))))


def create_huffman_tree_from_dict(huffman_dict):
    root = Node(char=None, freq=0)  # Create a root node with no char and zero frequency
    for char, code in huffman_dict.items():
      current_node = root
      for bit in code:
        if bit == '0':
          if current_node.left is None:
            current_node.left = Node(char=None, freq=0)
          current_node = current_node.left
        elif bit == '1':
          if current_node.right is None:
            current_node.right = Node(char=None, freq=0)
          current_node = current_node.right
      current_node.char = char
    return root
# # Example usage
# text = "This is an example text for Huffman coding."
# char_freq_map = {}
# for char in text:
#   char_freq_map[char] = char_freq_map.get(char, 0) + 1
#
# import heapq  # for priority queue
#
# # Create Huffman tree
# root = create_huffman_tree(char_freq_map)
#
# # Create Huffman codes dictionary
# codes = {}
# create_huffman_codes(root, "", codes)
#
# # Encode the text
# encoded_text = encode(text, codes)
#
# # Decode the encoded text
# decoded_text = decode(encoded_text, root)
#
# print("Original text:", text)
# print("Encoded text:", encoded_text)
# print("Decoded text:", decoded_text)
