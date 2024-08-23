import numpy as np


def decompress_final(final_decoded_data, metadata1):
    # Both final_decoded_data and metadata1 are in IEEE 754 float32 format

    # Step 1: Calculate the final size after decompression
    final_size = len(final_decoded_data)

    # Iterate through metadata1 to compute the additional size needed for the inserted values
    for i in range(0, len(metadata1), 3):
        count = np.float32(metadata1[i + 2]).view(np.int32)  # Convert IEEE 754 to int for count
        final_size += count  # Add the count to the final size

    # Step 2: Create a new array with the final size
    reconstructed_data = np.zeros(final_size, dtype=np.float32)

    # Step 3: Populate the new array with values from final_decoded_data and metadata1
    current_position = 0  # Track the current position in the reconstructed array

    # Keep track of the position in the original final_decoded_data
    original_position = 0

    # Process metadata1 to perform insertions
    for i in range(0, len(metadata1), 3):
        # Convert start_index and count to integers, as these are stored in IEEE 754 format
        start_index = np.float32(metadata1[i]).view(np.int32)  # Convert IEEE 754 to int for start_index
        value = metadata1[i + 1]  # Value remains in IEEE 754 format
        count = np.float32(metadata1[i + 2]).view(np.int32)  # Convert IEEE 754 to int for count

        # Copy data from final_decoded_data up to the start_index
        reconstructed_data[current_position:current_position + (start_index - original_position)] = final_decoded_data[
                                                                                                    original_position:start_index]
        current_position += (start_index - original_position)

        # Insert the value count times at the correct position
        reconstructed_data[current_position:current_position + count] = value
        current_position += count

        # Update the original position to continue copying from the original data
        original_position = start_index

    # Copy any remaining data from final_decoded_data after the last insertion
    reconstructed_data[current_position:] = final_decoded_data[original_position:]

    return reconstructed_data


# Example usage
# Both 'metadata1' and 'final_decoded_data' are assumed to be in IEEE 754 float32 format
metadata1 = np.array([1065353216, 1073741824, 1073741824,  # start_index = 2, value = 2.0, count = 2
                      1090519040, 1084227584, 1077936128], dtype=np.float32)  # start_index = 5, value = 4.5, count = 3
final_decoded_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Example data

# Decompress the final data
reconstructed_data = decompress_final(final_decoded_data, metadata1)

print(reconstructed_data)
