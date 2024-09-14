import struct
import numpy as np
import pandas as pd

# Function to convert a floating-point number to its binary IEEE 754 representation
def float_to_ieee754(f):
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its IEEE 754 binary form."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return ''.join(binary_str)  # Return binary string

    if isinstance(f, np.ndarray):
        return [float_to_binary_array(single_f) for single_f in f.ravel()]
    else:
        return float_to_binary_array(f)


# Function to compare two binary strings and return the number of identical leading bits
def compare_binary_leading_same(bin1, bin2):
    same_bits_count = 0
    for i in range(len(bin1)):
        if bin1[i] == bin2[i]:
            same_bits_count += 1
        else:
            break  # Stop at the first difference
    return same_bits_count


# Function to compare two binary strings and return the number of identical trailing bits
def compare_binary_trailing_same(bin1, bin2):
    same_bits_count = 0
    for i in range(1, len(bin1) + 1):
        if bin1[-i] == bin2[-i]:
            same_bits_count += 1
        else:
            break  # Stop at the first difference
    return same_bits_count


# Function to process a dataset and compare consecutive values
def compare_consecutive_values(dataset, original_values):
    leading_same_bits_counts = []
    trailing_same_bits_counts = []
    binary_pairs = []
    float_pairs = []

    for i in range(len(dataset) - 1):
        binary1 = dataset[i]
        binary2 = dataset[i + 1]

        # Compare leading and trailing bits
        leading_same_bits = compare_binary_leading_same(binary1, binary2)
        trailing_same_bits = compare_binary_trailing_same(binary1, binary2)

        leading_same_bits_counts.append(leading_same_bits)
        trailing_same_bits_counts.append(trailing_same_bits)
        binary_pairs.append((binary1, binary2))

        # Store the original floating-point values as pairs
        float_pairs.append((original_values[i], original_values[i + 1]))

    return leading_same_bits_counts, trailing_same_bits_counts, binary_pairs, float_pairs


# Define the dataset path
dataset_path ="/home/jamalids/Documents/2D/data1/rsim_f32.tsv"

# Load the dataset
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1 = ts_data1.T
ts_data1 = ts_data1.iloc[0:1, 2000000:2100000]  # Select specific rows and columns as needed
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)

# Convert dataset to binary IEEE 754 representation
binary_level = float_to_ieee754(ts_data1)

# Compare consecutive values in the dataset for leading and trailing bits
leading_same_bits_counts, trailing_same_bits_counts, binary_pairs, float_pairs = compare_consecutive_values(binary_level, ts_data1)

# Create a DataFrame to store the results, including the original float values and both leading/trailing results
df_results = pd.DataFrame({
    "Value 1 (Float)": [pair[0] for pair in float_pairs],
    "Value 2 (Float)": [pair[1] for pair in float_pairs],
    "Value 1 (Binary)": [pair[0] for pair in binary_pairs],
    "Value 2 (Binary)": [pair[1] for pair in binary_pairs],
    "Number of Same Leading Bits": leading_same_bits_counts,
    "Number of Same Trailing Bits": trailing_same_bits_counts
})

# Save the DataFrame to a CSV file
df_results.to_csv("bit_with_float_leading_trailing.csv", index=False)

# Display the DataFrame
print(df_results)

# Optional: Calculate and display statistics about the same bits
average_leading_bits = np.mean(leading_same_bits_counts)
max_leading_bits = np.max(leading_same_bits_counts)
min_leading_bits = np.min(leading_same_bits_counts)

average_trailing_bits = np.mean(trailing_same_bits_counts)
max_trailing_bits = np.max(trailing_same_bits_counts)
min_trailing_bits = np.min(trailing_same_bits_counts)

print(f"Average number of same leading bits: {average_leading_bits}")
print(f"Maximum number of same leading bits: {max_leading_bits}")
print(f"Minimum number of same leading bits: {min_leading_bits}")

print(f"Average number of same trailing bits: {average_trailing_bits}")
print(f"Maximum number of same trailing bits: {max_trailing_bits}")
print(f"Minimum number of same trailing bits: {min_trailing_bits}")
