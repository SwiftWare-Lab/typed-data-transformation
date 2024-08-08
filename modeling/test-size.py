import sys

def calculate_huffman_dict_size(huffman_dict):
    total_size_bits = 0
    for key, value in huffman_dict.items():
        key_size_bits = key.bit_length()  # Number of bits required to represent the integer key
        value_size_bits = len(value)      # Number of bits in the Huffman code string
        total_size_bits += key_size_bits + value_size_bits
    return total_size_bits

# Example Huffman dictionary
huffman_dictionary = {0: '101', 1: '110', 3: '00', 9: '1000', 10: '1001', 11: '111', 12: '010', 15: '011'}

# Calculate the size
huffman_dict_size_bits = calculate_huffman_dict_size(huffman_dictionary)
print(f"Size of Huffman dictionary: {huffman_dict_size_bits} bits")