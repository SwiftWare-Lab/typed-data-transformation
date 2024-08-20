import numpy as np

def split_array_on_multiple_consecutive_values(data, threshold_percentage=20):
    total_length = len(data)
    threshold = total_length * (threshold_percentage / 100.0)

    consecutive_count = 1
    start_idx = 0
    split_arrays = []

    def are_equal(val1, val2):
        # This function checks if two values are equal, treating NaN as equal to NaN
        if np.isnan(val1) and np.isnan(val2):
            return True
        return val1 == val2

    for i in range(1, total_length):
        if are_equal(data[i], data[i - 1]):
            consecutive_count += 1
        else:
            if consecutive_count > threshold:
                # Append the array before the consecutive sequence
                if start_idx < i - consecutive_count:
                    split_arrays.append(data[start_idx:i - consecutive_count])
                # Append the consecutive sequence
                split_arrays.append(data[i - consecutive_count:i])
                # Update the start index for the next segment
                start_idx = i
            consecutive_count = 1

    # Handle the case where the array ends with consecutive values
    if consecutive_count > threshold:
        if start_idx < total_length - consecutive_count:
            split_arrays.append(data[start_idx:total_length - consecutive_count])
        split_arrays.append(data[total_length - consecutive_count:])
    else:
        # Append the final segment if no consecutive sequence at the end
        split_arrays.append(data[start_idx:])

    return split_arrays

# Example usage:
data = np.array([1.2, np.nan, np.nan, 2.5, np.nan, np.nan, np.nan, 4.4, 4.4, 4.4, np.nan, 5.1, 5.1, 6.2], dtype=np.float32)
result = split_array_on_multiple_consecutive_values(data)

# Display the resulting split arrays
for i, subarray in enumerate(result):
    print(f"Subarray {i}: {subarray}")
