import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def float_to_ieee754(f):
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its IEEE 754 binary form."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return binary_str  # Return binary string

    if isinstance(f, np.ndarray):
        return [float_to_binary_array(single_f) for single_f in f.ravel()]
    else:
        return float_to_binary_array(f)

def binary_str_to_int_array(binary_str):
    """Convert a binary string to a NumPy array of integers (0s and 1s)."""
    return np.array([int(bit) for bit in binary_str])

# Load the dataset
dataset_path = "/home/jamalids/Documents/2D/data1/num_control_f64.tsv"
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1 = ts_data1.T
#ts_data1=ts_data1.iloc[:,1:10000000]
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)

# Convert dataset to binary IEEE 754 representation
binary_level = float_to_ieee754(ts_data1)

# Convert each binary string to a numerical array of 0s and 1s
binary_arrays = [binary_str_to_int_array(b) for b in binary_level]

# Stack binary arrays to form a 2D array where each row is a binary representation of a float
binary_matrix = np.vstack(binary_arrays)

# Perform XOR correlation between each pair of columns
num_columns = binary_matrix.shape[1]

# Matrix to store XOR correlation results
xor_correlation_matrix = np.zeros((num_columns, num_columns))

# Calculate XOR for each pair of columns
for i in range(num_columns):
    for j in range(i, num_columns):
        # XOR the two columns and sum the result to get a measure of correlation
        xor_result = np.sum(binary_matrix[:, i] ^ binary_matrix[:, j])
        xor_correlation_matrix[i, j] = xor_result
        xor_correlation_matrix[j, i] = xor_result  # Symmetric matrix

# Convert the XOR correlation matrix to a DataFrame for easier viewing
xor_correlation_df = pd.DataFrame(xor_correlation_matrix)
xor_correlation_df.to_csv("corrolation.csv")

# Display the XOR correlation matrix using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(xor_correlation_df, cmap='viridis', interpolation='nearest')
plt.colorbar(label='XOR Sum')
plt.title("XOR Correlation Matrix")
plt.xlabel("Column Index")
plt.ylabel("Column Index")
plt.savefig(f"corr2.png")
plt.show()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(xor_correlation_matrix, cmap='coolwarm')

# Add color bar
fig.colorbar(cax)

# Add correlation values to each cell
for i in range(num_columns):
    for j in range(num_columns):
        ax.text(j, i, f'{xor_correlation_matrix[i, j]:.0f}', ha='center', va='center', color='black')

ax.set_title("XOR Correlation Matrix with Values")
ax.set_xlabel("Column Index")
ax.set_ylabel("Column Index")
plt.savefig(f"corr1.png")
plt.show()
