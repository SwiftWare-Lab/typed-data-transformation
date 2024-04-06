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
