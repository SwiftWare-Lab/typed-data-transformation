#
# import heapq  # for priority queue
#
# class Node:
#   """
#   Node class for representing characters and their frequencies in the Huffman tree.
#   """
#   def __init__(self, char, freq):
#     self.char = char
#     self.freq = freq
#     self.left = None
#     self.right = None
#
#   def __lt__(self, other):
#     """
#     Defines less-than comparison for priority queue (based on frequency).
#     """
#     return self.freq < other.freq
#
# def create_huffman_tree(char_freq_map):
#   """
#   Creates a Huffman tree from a character-frequency map.
#
#   Args:
#       char_freq_map: A dictionary mapping characters to their frequencies.
#
#   Returns:
#       The root node of the Huffman tree.
#   """
#   nodes = [Node(char, freq) for char, freq in char_freq_map.items()]
#   while len(nodes) > 1:
#     # Get two nodes with the lowest frequencies
#     min1, min2 = heapq.heappop(nodes), heapq.heappop(nodes)
#     # Create a new parent node with combined frequency
#     parent = Node(None, min1.freq + min2.freq)
#     parent.left = min1
#     parent.right = min2
#     # Add the parent node back to the queue
#     heapq.heappush(nodes, parent)
#   return nodes[0]
#
# def create_huffman_codes(node, code, codes):
#   """
#   Creates a dictionary mapping characters to their Huffman codes by traversing the tree.
#
#   Args:
#       node: The current node in the tree.
#       code: The current code string for the path from the root to the current node.
#       codes: A dictionary to store the character-code mappings.
#   """
#   if node is None:
#     return
#   if node.char is not None:
#     codes[node.char] = code
#   create_huffman_codes(node.left, code + '0', codes)
#   create_huffman_codes(node.right, code + '1', codes)
#
# def encode(text, codes):
#   """
#   Encodes a text string using the Huffman codes.
#
#   Args:
#       text: The text string to encode.
#       codes: A dictionary mapping characters to their Huffman codes.
#
#   Returns:
#       The encoded text as a binary string.
#   """
#   encoded_text = ""
#   for char in text:
#     encoded_text += codes[char]
#   return encoded_text
#
# def decode(encoded_text, root):
#   """
#   Decodes a binary string using the Huffman tree.
#
#   Args:
#       encoded_text: The encoded text as a binary string.
#       root: The root node of the Huffman tree.
#
#   Returns:
#       The decoded text string.
#   """
#   decoded_text = ""
#   current_node = root
#   for bit in encoded_text:
#     if bit == '0':
#       current_node = current_node.left
#     elif bit == '1':
#       current_node = current_node.right
#     if current_node.char is not None:
#       decoded_text += current_node.char
#       current_node = root
#   return decoded_text
#
# # # Example usage
# # text = "This is an example text for Huffman coding."
# # char_freq_map = {}
# # for char in text:
# #   char_freq_map[char] = char_freq_map.get(char, 0) + 1
# #
# # import heapq  # for priority queue
# #
# # # Create Huffman tree
# # root = create_huffman_tree(char_freq_map)
# #
# # # Create Huffman codes dictionary
# # codes = {}
# # create_huffman_codes(root, "", codes)
# #
# # # Encode the text
# # encoded_text = encode(text, codes)
# #
# # # Decode the encoded text
# # decoded_text = decode(encoded_text, root)
# #
# # print("Original text:", text)
# # print("Encoded text:", encoded_text)
# # print("Decoded text:", decoded_text)
#
# import heapq  # for priority queue
#
# class Node:
#   """
#   Node class for representing characters and their frequencies in the Huffman tree.
#   """
#   def __init__(self, char, freq):
#     self.char = char
#     self.freq = freq
#     self.left = None
#     self.right = None
#
#   def __lt__(self, other):
#     """
#     Defines less-than comparison for priority queue (based on frequency).
#     """
#     return self.freq < other.freq
#
# def create_huffman_tree(char_freq_map):
#   """
#   Creates a Huffman tree from a character-frequency map.
#
#   Args:
#       char_freq_map: A dictionary mapping characters to their frequencies.
#
#   Returns:
#       The root node of the Huffman tree.
#   """
#   nodes = [Node(char, freq) for char, freq in char_freq_map.items()]
#   while len(nodes) > 1:
#     # Get two nodes with the lowest frequencies
#     min1, min2 = heapq.heappop(nodes), heapq.heappop(nodes)
#     # Create a new parent node with combined frequency
#     parent = Node(None, min1.freq + min2.freq)
#     parent.left = min1
#     parent.right = min2
#     # Add the parent node back to the queue
#     heapq.heappush(nodes, parent)
#   return nodes[0]
#
# def create_huffman_codes(node, code, codes):
#   """
#   Creates a dictionary mapping characters to their Huffman codes by traversing the tree.
#
#   Args:
#       node: The current node in the tree.
#       code: The current code string for the path from the root to the current node.
#       codes: A dictionary to store the character-code mappings.
#   """
#   if node is None:
#     return
#   if node.char is not None:
#     codes[node.char] = code
#   create_huffman_codes(node.left, code + '0', codes)
#   create_huffman_codes(node.right, code + '1', codes)
#
# def encode(text, codes):
#   """
#   Encodes a text string using the Huffman codes.
#
#   Args:
#       text: The text string to encode.
#       codes: A dictionary mapping characters to their Huffman codes.
#
#   Returns:
#       The encoded text as a binary string.
#   """
#   encoded_text = ""
#   for char in text:
#     encoded_text += codes[char]
#   return encoded_text
#
# def decode(encoded_text, root):
#   """
#   Decodes a binary string using the Huffman tree.
#
#   Args:
#       encoded_text: The encoded text as a binary string.
#       root: The root node of the Huffman tree.
#
#   Returns:
#       The decoded text string.
#   """
#   decoded_text = ""
#   current_node = root
#   for bit in encoded_text:
#     if bit == '0':
#       current_node = current_node.left
#     elif bit == '1':
#       current_node = current_node.right
#     if current_node.char is not None:
#       decoded_text += current_node.char
#       current_node = root
#   return decoded_text
#
# # # Example usage
# # text = "This is an example text for Huffman coding."
# # char_freq_map = {}
# # for char in text:
# #   char_freq_map[char] = char_freq_map.get(char, 0) + 1
# #
# # import heapq  # for priority queue
# #
# # # Create Huffman tree
# # root = create_huffman_tree(char_freq_map)
# #
# # # Create Huffman codes dictionary
# # codes = {}
# # create_huffman_codes(root, "", codes)
# #
# # # Encode the text
# # encoded_text = encode(text, codes)
# #
# # # Decode the encoded text
# # decoded_text = decode(encoded_text, root)
# #
# # print("Original text:", text)
# # print("Encoded text:", encoded_text)
# # print("Decoded text:", decoded_text)
import heapq  # for priority queue
import pickle

class Node:
    """
    Node class for representing a byte symbol and its frequency in the Huffman tree.
    """
    def __init__(self, symbol, freq):
        self.symbol = symbol  # integer 0-255
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def create_huffman_tree(freq_map):
    """
    Build a Huffman tree for byte symbols from a frequency map.

    Args:
        freq_map (dict[int, int]): Mapping from byte value (0-255) to frequency.

    Returns:
        Node: Root of the Huffman tree.
    """
    # Create a min-heap of initial nodes
    heap = [Node(sym, f) for sym, f in freq_map.items()]
    heapq.heapify(heap)

    # Iteratively combine lowest-frequency nodes
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = Node(None, n1.freq + n2.freq)
        parent.left, parent.right = n1, n2
        heapq.heappush(heap, parent)

    return heap[0]


def create_huffman_codes(node, prefix="", codes=None):
    """
    Traverse the Huffman tree to generate bit codes for each byte symbol.
    """
    if codes is None:
        codes = {}
    if node.symbol is not None:
        # Leaf node: assign code (use '0' if tree has single node)
        codes[node.symbol] = prefix or "0"
    else:
        create_huffman_codes(node.left, prefix + "0", codes)
        create_huffman_codes(node.right, prefix + "1", codes)
    return codes


def compress_to_file(data: bytes, filename: str):
    """
    Compress a bytes sequence with Huffman coding, writing metadata + payload to file.

    File structure:
      [4-byte header length][header bytes][1-byte padding][payload bytes]
    Header (pickled dict): {'codes': {symbol: bitstring, ...}}
    """
    # 1. Compute frequencies
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    # 2. Build tree & codes
    root = create_huffman_tree(freq)
    codes = create_huffman_codes(root)

    # 3. Serialize metadata
    header = pickle.dumps({'codes': codes})
    header_len = len(header)

    # 4. Encode data to bitstring
    bitstr = ''.join(codes[b] for b in data)

    # 5. Pad to byte boundary
    padding = (8 - len(bitstr) % 8) % 8
    bitstr += '0' * padding
    payload = bytearray()
    for i in range(0, len(bitstr), 8):
        payload.append(int(bitstr[i:i+8], 2))

    # 6. Write file
    with open(filename, 'wb') as f:
        f.write(header_len.to_bytes(4, 'big'))
        f.write(header)
        f.write(padding.to_bytes(1, 'big'))
        f.write(payload)


def decompress_from_file(filename: str) -> bytes:
    """
    Decompress from file and return original bytes.
    """
    with open(filename, 'rb') as f:
        header_len = int.from_bytes(f.read(4), 'big')
        header = pickle.loads(f.read(header_len))
        codes = header['codes']
        padding = int.from_bytes(f.read(1), 'big')
        payload = f.read()

    # Build reverse map
    rev = {v: k for k, v in codes.items()}

    # Convert payload to bitstring
    bits = ''.join(f"{byte:08b}" for byte in payload)
    if padding:
        bits = bits[:-padding]

    # Decode bits
    out = bytearray()
    buf = ""
    for bit in bits:
        buf += bit
        if buf in rev:
            out.append(rev[buf])
            buf = ""
    return bytes(out)

