import numpy as np


def binary_to_int(binary_array):
    """
    Convert an array of binary digits (0s and 1s) to an integer.
    """
    binary_string = ''.join(map(str, binary_array))
    return int(binary_string, 2)


def convert_to_int_array(binary_arrays):
    """
    Convert a 2D numpy array of binary arrays (each row represents binary digits) to an array of integers.
    """
    return np.array([binary_to_int(row) for row in binary_arrays])


def rle_encode(data):
    """
    Run-Length Encoding (RLE) for an array of integers.
    """
    encoded = []
    count = 1
    previous = data[0]

    # Loop through the array starting from the second element
    for value in data[1:]:
        if value == previous:
            count += 1
        else:
            encoded.append((previous, count))
            count = 1
            previous = value

    # Append the last run
    encoded.append((previous, count))
    return encoded


def calculate_size_in_bits(array, bit_length_func):
    """
    Calculate the size of an array in bits.

    :param array: The array to calculate the size for.
    :param bit_length_func: A function that calculates the number of bits for a given element.
    :return: Total size in bits.
    """
    return sum(bit_length_func(int(value)) for value in array)  # Convert to Python int


def bit_length_for_int(value):
    """
    Calculate the number of bits required to represent an integer.
    """
    return value.bit_length()


def bit_length_for_rle(encoded_data):
    """
    Calculate the total number of bits for RLE encoded data.
    Each tuple in RLE contains a value and a count.
    """
    total_bits = 0
    for value, count in encoded_data:
        total_bits += int(value).bit_length()  # Convert to Python int
        total_bits += int(count).bit_length()  # Convert to Python int
    return total_bits


# Example usage
first_10_bits = np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                          [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                          [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]])  # Example binary arrays

# Step 1: Convert the first 10 bits to integers
array10 = convert_to_int_array(first_10_bits)

# Step 2: Apply RLE to the integer array
rle_encoded_array10 = rle_encode(array10)

# Step 3: Calculate the size of `array10` in bits
array10_size_in_bits = calculate_size_in_bits(array10, bit_length_for_int)

# Step 4: Calculate the size of `rle_encoded_array10` in bits
rle_encoded_array10_size_in_bits = bit_length_for_rle(rle_encoded_array10)

print("First 10 Bits as Integers:", array10)
print("RLE Encoded Integers:", rle_encoded_array10)
print("Size of array10 in bits:", array10_size_in_bits)
print("Size of rle_encoded_array10 in bits:", rle_encoded_array10_size_in_bits)
