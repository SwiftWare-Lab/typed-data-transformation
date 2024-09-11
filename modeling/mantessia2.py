import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def float_to_ieee754(f):
    """
    Convert a float or a numpy array of floats to their IEEE 754 binary representation
    and return as an integer array.
    """
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its IEEE 754 binary form."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        # Apply the conversion to each element in the numpy array
        return np.array([float_to_binary_array(single_f) for single_f in f.ravel()]).reshape(f.shape + (32,))
    else:
        # Apply the conversion to a single float
        return float_to_binary_array(f)

def compute_difference(value1, value2):
    """
    Compute the difference between two floating-point numbers.
    """
    return value1 - value2


def find_significant_decimal_position(difference):
    """
    Find the decimal position where the difference starts showing non-zero values.
    """
    # Convert to string with sufficient precision
    str_diff = f"{difference:.20f}"

    if '.' in str_diff:
        decimal_part = str_diff.split('.')[1]
        first_non_zero_index = next((i for i, char in enumerate(decimal_part) if char != '0'), None)
        if first_non_zero_index is not None:
            return first_non_zero_index + 1
    return 0


def decimal_points_before_difference(value1, value2):
    """
    Find the number of decimal points that are the same before a significant difference appears.
    """
    str_value1 = f"{value1:.20f}"
    str_value2 = f"{value2:.20f}"

    if '.' in str_value1 and '.' in str_value2:
        decimal_part1 = str_value1.split('.')[1]
        decimal_part2 = str_value2.split('.')[1]

        min_length = min(len(decimal_part1), len(decimal_part2))
        same_digits_count = 0
        for i in range(min_length):
            if decimal_part1[i] == decimal_part2[i]:
                same_digits_count += 1
            else:
                break
        return same_digits_count
    return 0


def bits_required_for_digit(digit):
    """
    Return the number of bits required to represent a single decimal digit.
    """
    # Example mapping (you can adjust based on your requirements)
    bits_map = {
        '0': 2,
        '1': 2,
        '2': 2,
        '3': 3,
        '4': 3,
        '5': 3,
        '6': 3,
        '7': 3,
        '8': 4,
        '9': 4
    }
    return bits_map.get(digit, 4)  # Default to 4 bits if digit is not found


def estimate_total_bits(decimal_part):
    """
    Estimate the total number of bits required to represent a decimal part.
    """

    total_bits = 0
    for char in decimal_part:
        total_bits += bits_required_for_digit(char)
    return total_bits


# Define the dataset path
dataset_path = "/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"

# Load the dataset
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1 = ts_data1.T
ts_data1 = ts_data1.iloc[0:1, 0:100]  # Select specific rows and columns as needed
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)
binary_level=float_to_ieee754(ts_data1 )
mantissa_data = binary_level[:, 9:32]
print(binary_level)
print(mantissa_data)
results=[]
# Process the dataset to find differences and estimate bits
for i in range(len(ts_data1) - 1):
    value1 = ts_data1[i]
    value2 = ts_data1[i + 1]
    v1=float_to_ieee754(value1)
    v2 = float_to_ieee754(value2)
    mantissa_data1 = v1[ 9:32]
    mantissa_data2 = v2[ 9:32]


    difference = compute_difference(value1, value2)
    decimal_position = find_significant_decimal_position(difference)

    # Find decimal points that are the same before the significant difference
    same_decimal_points = decimal_points_before_difference(value1, value2)

    # Convert values to strings for bit estimation
    str_value1 = f"{value1:.20f}"
    str_value2 = f"{value2:.20f}"
    decimal_part = str_value1.split('.')[1][:same_decimal_points]
    if decimal_position != 0 and same_decimal_points == 0:
       # decimal_part = f"{difference:.20f}".split('.')[1][:decimal_position - 1]
        decimal_part = str_value1.split('.')[1][:decimal_position - 1]
    # Estimate bits required
    total_bits = estimate_total_bits(decimal_part)
    results.append({
        'Value1': value1,
        'Value2': value2,
        'Difference': difference,
        'Decimal Position': decimal_position,
        'Bits Required': total_bits,
        'mantissa_data1':mantissa_data1,
        'mantissa_data2':mantissa_data2,
        'v1':v1,
        'v2':v2

    })

    print(f"Value 1: {value1}")
    print(f"Value 2: {value2}")
    print(f"Difference: {difference}")
    print(f"Decimal Position: {decimal_position}")
    print(f"Same Decimal Points: {same_decimal_points}")
    print(f"Decimal Part for Bit Estimation: {decimal_part}")
    print(f"Estimated Total Bits: {mantissa_data2}")
    print(f"Estimated Total Bits: {total_bits}")
    print("------")
    # Create DataFrame and save to CSV
df_results = pd.DataFrame(results)

df_results.to_csv("1.csv", index=False)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_results['Decimal Position'], df_results['Bits Required'], marker='o', linestyle='-')
plt.title('Estimated Bits Required vs. Decimal Position')
plt.xlabel('Decimal Position')
plt.ylabel('Bits Required')
plt.grid(True)
plt.savefig(os.path.join( 'plot.png'))
plt.show()
