
import heapq  # for priority queue

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
  """
  Creates a Huffman tree from a character-frequency map.

  Args:
      char_freq_map: A dictionary mapping characters to their frequencies.

  Returns:
      The root node of the Huffman tree.
  """
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

def encode(text, codes):
  """
  Encodes a text string using the Huffman codes.

  Args:
      text: The text string to encode.
      codes: A dictionary mapping characters to their Huffman codes.

  Returns:
      The encoded text as a binary string.
  """
  encoded_text = ""
  for char in text:
    encoded_text += codes[char]
  return encoded_text

def decode(encoded_text, root):
  """
  Decodes a binary string using the Huffman tree.

  Args:
      encoded_text: The encoded text as a binary string.
      root: The root node of the Huffman tree.

  Returns:
      The decoded text string.
  """
  decoded_text = ""
  current_node = root
  for bit in encoded_text:
    if bit == '0':
      current_node = current_node.left
    elif bit == '1':
      current_node = current_node.right
    if current_node.char is not None:
      decoded_text += current_node.char
      current_node = root
  return decoded_text

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
