import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Define the dataset path
dataset_path = "/home/jamalids/Documents/2D/data1/num_brain_f64.tsv"

# Load the dataset
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)
ts_data1 = ts_data1.T
ts_data1 = ts_data1.iloc[0:1, 77:82]  # Select specific rows and columns as needed
ts_data1 = ts_data1.astype(np.float32).to_numpy().reshape(-1)
print(ts_data1)

bit_array = float_to_ieee754(ts_data1)
print(bit_array)

# Extract mantissa part
mantissa_data = bit_array[:, 9:32]
print(mantissa_data)

# Count how many times the bits change in each column
bit_changes = np.zeros(mantissa_data.shape[1])

for col in range(mantissa_data.shape[1]):
    previous_bit = mantissa_data[0, col]
    for row in range(1, mantissa_data.shape[0]):
        current_bit = mantissa_data[row, col]
        if current_bit != previous_bit:
            bit_changes[col] += 1
        previous_bit = current_bit

# Iterate over possible window sizes (starting from 3)
min_window_size = 3
max_window_size = mantissa_data.shape[1]
best_window_size = min_window_size
lowest_change_sum = float('inf')
highest_change_sum = 0
best_interval = (0, 0)

window_sizes = []
min_changes_per_window = []
max_changes_per_window = []
min_intervals_per_window = []
max_intervals_per_window = []

for window_size in range(min_window_size, max_window_size + 1):
    lowest_sum_for_window = float('inf')
    highest_sum_for_window = 0
    min_interval_for_window = (0, 0)
    max_interval_for_window = (0, 0)


    for start in range(mantissa_data.shape[1] - window_size + 1):
        interval_sum = np.sum(bit_changes[start:start + window_size])
        if interval_sum < lowest_sum_for_window:
            lowest_sum_for_window = interval_sum
            min_interval_for_window = (start + 1, start + window_size)
        if interval_sum > highest_sum_for_window:
            highest_sum_for_window = interval_sum
            max_interval_for_window = (start + 1, start + window_size)


    window_sizes.append(window_size)
    min_changes_per_window.append(lowest_sum_for_window)
    max_changes_per_window.append(highest_sum_for_window)
    min_intervals_per_window.append(min_interval_for_window)
    max_intervals_per_window.append(max_interval_for_window)


    if lowest_sum_for_window < lowest_change_sum:
        lowest_change_sum = lowest_sum_for_window
        best_window_size = window_size
        best_interval = min_interval_for_window

    # Step 3: Print results for each window size
    print(f"Window Size {window_size}: Min interval is from column {min_interval_for_window[0]} to {min_interval_for_window[1]} with {lowest_sum_for_window} bit changes.")
    print(f"Window Size {window_size}: Max interval is from column {max_interval_for_window[0]} to {max_interval_for_window[1]} with {highest_sum_for_window} bit changes.")


print(f"\nOverall Best Window Size: {best_window_size}")
print(f"Best interval for min changes is from column {best_interval[0]} to {best_interval[1]} with {lowest_change_sum} bit changes")
float_values = ts_data1  # The original float values
consecutive_diff = np.abs(np.diff(float_values))

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column

# Plot 1: Min and Max bit changes for each window size
axs[0].plot(window_sizes, min_changes_per_window, marker='o', label='Min Changes per Window Size', color='blue')
axs[0].plot(window_sizes, max_changes_per_window, marker='x', label='Max Changes per Window Size', color='orange')

for i, (min_txt, max_txt) in enumerate(zip(min_intervals_per_window, max_intervals_per_window)):
    axs[0].annotate(f"Min:{min_txt[0]}-{min_txt[1]}", (window_sizes[i], min_changes_per_window[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    axs[0].annotate(f"Max:{max_txt[0]}-{max_txt[1]}", (window_sizes[i], max_changes_per_window[i]), textcoords="offset points", xytext=(0, -15), ha='center', color='orange')

axs[0].set_xticks(np.arange(min(window_sizes), max(window_sizes) + 1, 1))  # Show only integer ticks on the x-axis
axs[0].set_title('Minimum and Maximum Bit Changes for Each Window Size')
axs[0].set_xlabel('Window Size')
axs[0].set_ylabel('Bit Changes')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Consecutive float value changes
axs[1].plot(consecutive_diff, marker='o', color='purple', label='Consecutive Float Value Changes')
axs[1].set_title('Trend of Consecutive Float Value Changes')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Absolute Difference between Consecutive Values')
axs[1].legend()
axs[1].grid(True)

# Adjust layout
plt.tight_layout()

# Save and display the combined figure
plt.savefig("combined_plots.png")
plt.show()
