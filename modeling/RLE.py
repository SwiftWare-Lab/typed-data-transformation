import numpy as np

def split_array_on_multiple_consecutive_values(data, threshold_percentage=0.1):
    total_length = len(data)
    threshold = total_length * (threshold_percentage / 100.0)

    consecutive_count = 1
    start_idx = 0
    non_consecutive_array = []
    metadata = []

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
                # Record metadata about the sequence
                metadata.append({
                    'value': data[i - 1],
                    'count': consecutive_count,
                    'start_index': i - consecutive_count
                })
            else:
                # Append the segment to the single non_consecutive_array
                non_consecutive_array.extend(data[start_idx:i])
            # Update the start index for the next segment
            start_idx = i
            consecutive_count = 1

    # Handle the end of the array
    if consecutive_count > threshold:
        metadata.append({
            'value': data[-1],
            'count': consecutive_count,
            'start_index': total_length - consecutive_count
        })
    else:
        non_consecutive_array.extend(data[start_idx:])

    return non_consecutive_array, metadata

# Example usage
data = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5])
non_consecutive_array, metadata = split_array_on_multiple_consecutive_values(data, threshold_percentage=20)
print("Non-consecutive array:", non_consecutive_array)
print("Metadata:", metadata)
