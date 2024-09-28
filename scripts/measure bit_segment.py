# Let's implement the calculation of bits required based on the provided code structure.
import math
import pandas as pd
import numpy as np

# Set dataset path and load dataset
dataset_path = "/home/jamalids/Documents/2D/data1/TS/L/ts_gas_f32.tsv"


# Function to load dataset and return raw values
def load_dataset_values(dataset_path):
    """Load the dataset and return the raw values."""
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    group_f = ts_data1.drop(ts_data1.columns[0], axis=1).T
    group_f = group_f.astype(np.float32).to_numpy().reshape(-1)
    return group_f


# Function to count consecutive segments of similar values
def count_segments(data, value):
    """Count the number of segments for a specific value."""
    segments = 0
    in_segment = False
    for item in data:
        if item == value:
            if not in_segment:
                segments += 1
                in_segment = True
        else:
            in_segment = False
    return segments


# Function to find the max consecutive length for a specific value
def max_consecutive_length(data, value):
    """Find the maximum consecutive length for a specific value."""
    max_len = 0
    current_len = 0
    for item in data:
        if item == value:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 0
    return max_len


# Function to calculate total bits required for encoding each segment (segment number, value, frequency)
def calculate_bits_needed(data, unique_values):
    """Calculate the total bits needed to encode each segment."""
    N = len(unique_values)  # Number of unique values (segments)
    F_max = max([max_consecutive_length(data, val) for val in unique_values])  # Maximum frequency length
    V = len(np.unique(data))  # Number of distinct values

    # Calculate bits for segment number, value, and frequency (length)
    bits_for_segment_number = math.ceil(math.log2(N))  # Segment number bits
    bits_for_value = math.ceil(math.log2(V))  # Value bits
    bits_for_frequency = math.ceil(math.log2(F_max))  # Frequency bits

    total_bits_per_segment = bits_for_segment_number + bits_for_value + bits_for_frequency

    return total_bits_per_segment


# Load the data
group_f = load_dataset_values(dataset_path)

# Let's assume we use the top 10 most frequent values for the calculation
unique_values, counts = np.unique(group_f, return_counts=True)
sorted_indices = np.argsort(-counts)
top_10_unique_values = unique_values[sorted_indices[:10]]

# Calculate total bits needed for top 10 unique values
total_bits_needed = calculate_bits_needed(group_f, top_10_unique_values)

print(total_bits_needed ) # Display the total bits per segment for the top 10 unique values

