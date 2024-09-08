import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def float_to_ieee754(f):
    """
    Convert a float or a NumPy array of floats to their IEEE 754 binary representation
    and return as an integer array for float32.
    """
    def float_to_binary_array(single_f):
        """Convert a single float32 to a 32-bit binary array."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        # Apply the conversion to each element in the NumPy array
        return np.array([float_to_binary_array(single_f) for single_f in f.ravel()]).reshape(f.shape + (32,))
    else:
        # Apply the conversion to a single float
        return float_to_binary_array(f)


def ieee754_to_float(binary_array):
    """Convert a binary array back to float32."""
    if len(binary_array.shape) == 2:

        float_array = []
        for binary_row in binary_array:
            binary_str = ''.join(str(bit) for bit in binary_row)
            float_array.append(np.uint32(int(binary_str, 2)).view(np.float32))
        return np.array(float_array)
    else:
        # Convert a single binary array to float32
        binary_str = ''.join(str(bit) for bit in binary_array)
        return np.uint32(int(binary_str, 2)).view(np.float32)


def replace_first_n_bits_with_first_value(binary_array, n, first_value_bits):

    if len(binary_array.shape) == 1:
        binary_array[:n] = first_value_bits[:n]  # Replace first n bits with first value bits
    else:
        for i in range(binary_array.shape[0]):
            binary_array[i, :n] = first_value_bits[:n]
    return binary_array


# Define the dataset path
dataset_path = "/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"

# Load the dataset
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)  # Drop the first column if it's an index
ts_data1 = ts_data1.T  # Transpose the dataset
ts_data1 = ts_data1.iloc[0:1, 0:83]  # Select specific rows and columns (adjust as needed)

# Initialize a list to collect data for CSV
csv_data = []

for n in range(10, 32):

    binary_data = float_to_ieee754(ts_data1.values).copy()
    first_original_value=binary_data[0,0,:]


    binary_data_modified = np.array([replace_first_n_bits_with_first_value(binary_row, n, first_original_value)
                                     for binary_row in binary_data])


    modified_values = np.array([ieee754_to_float(row) for row in binary_data_modified])  # Shape: (1, 8)


    first_n_bits = ''.join(str(bit) for bit in first_original_value[:n])

    # Collect data for each column
    for col_idx, col in enumerate(ts_data1.columns):
        original_value = ts_data1.iloc[0, col_idx]
        modified_value = modified_values[0, col_idx]

        csv_data.append({
            'n_bits_replaced': n,
            'column': col,
            'original_value': original_value,
            'modified_value': modified_value,
            'first_n_bits': first_n_bits
        })

    # Plot the original vs modified values in the same plot for current n
    plt.figure(figsize=(10, 6))
    plt.plot(ts_data1.columns, ts_data1.values.flatten(), label="Original", linestyle='-', marker='o', color='b')
    plt.plot(ts_data1.columns, modified_values.flatten(), label=f"Modified ({n} bits replaced)", linestyle='--',
             marker='x', color='r')
    plt.title(f'Original vs Modified Values (First {n} Bits Replaced with First Value Bits)')
    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plot_filename = f'comparison_{n}_bits_replaced.png'
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close()

csv_df = pd.DataFrame(csv_data)

csv_output_path = 'original_modified_values.csv'

csv_df.to_csv(csv_output_path, index=False)
print(f"Saved CSV file: {csv_output_path}")
