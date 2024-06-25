import os
import pandas as pd
import numpy as np
from huffman import decompress_huffman, recreate_huffman_codes

def decompress_data(encoded_patterns, reverse_table, original_shapes):
   # print("Encoded Patterns: ", encoded_patterns)  # Debugging statement
    decoded_data = decompress_huffman(encoded_patterns, reverse_table)
    print("Decoded Data: ", decoded_data)  # Debugging statement
    reconstructed_patterns = reconstruct_patterns(decoded_data, original_shapes)
    return reconstructed_patterns

def read_compressed_data(file_path):
    data = pd.read_csv(file_path)
    return data['Compressed Data'].tolist()

def read_huffman_codes(file_path):
    huffman_codes = pd.read_csv(file_path)
    encoded_huffman_codes = huffman_codes.apply(lambda row: f"{row['Pattern']}:{row['Huffman Code']}", axis=1).tolist()
    return encoded_huffman_codes

def recreate_pattern_list(encoded_huffman_codes):
    huffman_codes = recreate_huffman_codes(encoded_huffman_codes)
    reverse_huffman_codes = {v: k for v, k in huffman_codes.items()}
    patterns = list(huffman_codes.keys())
    return patterns, reverse_huffman_codes

def decompress(file_path, huffman_codes_file, output_path, original_shapes):
    encoded_patterns = read_compressed_data(file_path)
    encoded_huffman_codes = read_huffman_codes(huffman_codes_file)
    patterns, reverse_table = recreate_pattern_list(encoded_huffman_codes)

    decompressed_patterns = decompress_data(encoded_patterns, reverse_table, original_shapes)
    # Flatten the 3D array to 2D for saving
    flattened_patterns = np.array(decompressed_patterns).reshape(-1, decompressed_patterns[0].shape[-1])
    np.savetxt(output_path, flattened_patterns, fmt='%d')

    return decompressed_patterns

def reconstruct_patterns(decoded_data, original_shapes):
    reconstructed_patterns = []
    index = 0
    for shape in original_shapes:
        size = np.prod(shape)
        if index + size > len(decoded_data):
            print(f"Error: Not enough data to reshape. Required size: {size}, available: {len(decoded_data) - index}")
            print(f"Decoded data segment: {decoded_data[index:index + size]}")  # Print problematic segment
            raise ValueError(f"Cannot reshape array of size {len(decoded_data[index:index + size])} into shape {shape}")
        pattern = decoded_data[index:index + size].reshape(shape)
        reconstructed_patterns.append(pattern)
        index += size
    return reconstructed_patterns

def read_original_shapes(file_path):
    original_data = pd.read_csv(file_path, delimiter='\t', header=None)
    shapes = [(1, row.size) for _, row in original_data.iterrows()]
    return shapes

if __name__ == "__main__":
    compressed_file_path = 'pattern.csv'
    huffman_codes_file = 'huffman_codes.csv'
    original_shapes_file = 'image-shape.csv'
    output_path = 'decompressed_output.txt'

    original_shapes = read_original_shapes(original_shapes_file)

    decompressed_patterns = decompress(compressed_file_path, huffman_codes_file, output_path, original_shapes)

    print("Decompression completed.")
