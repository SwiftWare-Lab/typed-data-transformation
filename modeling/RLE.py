def rle_encode_bits(bit_data):
    encoding = []
    i = 0

    while i < len(bit_data):
        count = 1

        # Count occurrences of the same bit sequence
        while i + 1 < len(bit_data) and bit_data[i] == bit_data[i + 1]:
            count += 1
            i += 1

        # Append the bit sequence (as a string) and its count to the encoding
        encoding.append((''.join(bit_data[i]), count))
        i += 1

    return encoding

def rle_decode_bits(encoding):
    decoded_data = []

    # Reconstruct the original bit data from the encoding
    for bit_sequence, count in encoding:
        bit_sequence_list = list(bit_sequence)
        decoded_data.extend([bit_sequence_list] * count)

    return decoded_data

# Example usage:
bit_data = [[01], [0, 1], [1 0], ['1', '0'], ['1', '1'], ['0', '0']]
encoded_data = rle_encode_bits(bit_data)
print("Encoded Data:", encoded_data)

decoded_data = rle_decode_bits(encoded_data)
print("Decoded Data:", decoded_data)
