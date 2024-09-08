import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the function to extract magnitude (sign, exponent, fraction)
def extract_magnitude(ieee754_binary):
    sign = ieee754_binary[0]
    exponent = ieee754_binary[1:9]  # 8 bits for exponent
    fraction = ieee754_binary[9:]   # 23 bits for mantissa
    return sign, exponent, fraction

# Function to convert a float or array of floats to IEEE 754 binary representation
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

# Define the dataset path
dataset_path = "/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"

# Load the dataset
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)

dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

# Drop the first column if it's an index, transpose the dataset, and select specific rows and columns
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)  # Drop the first column if it's an index
ts_data1 = ts_data1.T  # Transpose the dataset
ts_data1 = ts_data1.iloc[0:1, 0:83]  # Select specific rows and columns (adjust as needed)

# Assuming ts_data1 contains your time series floating-point values, flatten it to make it 1D
time_series = ts_data1.values.flatten()

# Extract the sign, exponent, and mantissa for each value in the time series
signs = [int(extract_magnitude(float_to_ieee754(v))[0]) for v in time_series]
exponents = [int(''.join(map(str, extract_magnitude(float_to_ieee754(v))[1])), 2) for v in time_series]
mantissas = [int(''.join(map(str, extract_magnitude(float_to_ieee754(v))[2])), 2) for v in time_series]

# Plot the sign over time
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(signs, label='Sign')
plt.title('Sign ')
plt.xlabel('column')
plt.ylabel('Sign')
plt.grid(True)

# Plot the exponent over time
plt.subplot(3, 1, 2)
plt.plot(exponents, label='Exponent', color='orange')
plt.title('Exponent Magnitude ')
plt.xlabel('column')
plt.ylabel('Magnitude (Exponent)')
plt.grid(True)

# Plot the mantissa over time
plt.subplot(3, 1, 3)
plt.plot(mantissas, label='Mantissa', color='green')
plt.title('Mantissa Magnitude')
plt.xlabel('column')
plt.ylabel('Mantissa')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.savefig("Magnitude.png")
plt.show()
