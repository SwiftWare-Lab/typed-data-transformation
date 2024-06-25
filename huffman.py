import heapq
import numpy as np

class HuffmanNode:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

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

def calculate_huffman_codes(patterns):
    concatenated_pattern = np.concatenate(patterns).ravel()

    frequencies = {}
    byte_pattern = [tuple(concatenated_pattern[i:i + 8]) for i in range(0, len(concatenated_pattern), 8)]
    for byte in byte_pattern:
        if byte in frequencies:
            frequencies[byte] += 1
        else:
            frequencies[byte] = 1

    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = {item[0]: item[1] for item in huffman_tree}

    encoded_pattern = ''.join(huffman_codes[tuple(concatenated_pattern[i:i + 8])] for i in range(0, len(concatenated_pattern), 8))

    encoded_size_bits = len(encoded_pattern)

    return huffman_codes, encoded_pattern, encoded_size_bits

def decompress_huffman(encoded_data, huffman_codes):
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    buffer = ""
    for bit in encoded_data:
        buffer += str(bit)
        if buffer in reverse_huffman_codes:
            decoded_data.extend(reverse_huffman_codes[buffer])
            buffer = ""
    return np.array(decoded_data).astype(int)

def recreate_huffman_codes(encoded_huffman_codes):
    huffman_codes = {}
    for line in encoded_huffman_codes:
        symbol, code = line.strip().split(':')
        huffman_codes[eval(symbol)] = code
    return huffman_codes
