import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set dataset path and load dataset
dataset_path = "/home/jamalids/Documents/2D/data1/TS/L/ts_gas_f32.tsv"

# Function to load dataset and return raw values
def load_dataset_values(dataset_path):
    """Load the dataset and return the raw values."""
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    print(f"Processing dataset: {dataset_name}")

    group_f = ts_data1.drop(ts_data1.columns[0], axis=1).T
    group_f = group_f.astype(np.float32).to_numpy().reshape(-1)

    return dataset_name, group_f

# Function to calculate the Shannon entropy of quantized data
def calculate_entropy_float(data):
    """Calculate the Shannon entropy of quantized data."""
    value, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

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

# Load data
dataset_name, group_f = load_dataset_values(dataset_path)

# Determine total elements and break into blocks
total_records = len(group_f)
block_size = 1000000  # Block size (you can adjust as needed)
max_blocks = 10  # Maximum number of blocks to show

num_blocks = min(total_records // block_size, max_blocks)

# Create a figure for the plots
fig, axes = plt.subplots(math.ceil(num_blocks / 4), 4, figsize=(20, 10))  # Use math.ceil here
axes = axes.flatten()

# Process and plot each block of block_size records, also calculating unique values and consecutive similar values
for block_idx in range(num_blocks):
    start_idx = block_idx * block_size
    end_idx = start_idx + block_size
    block_data = group_f[start_idx:end_idx]

    # Calculate entropy for the block
    entropy_float_all = calculate_entropy_float(block_data)

    # Count unique values in the block
    unique_values_count = len(np.unique(block_data))

    # Plot the data
    ax = axes[block_idx]
    ax.plot(block_data, label=f'Block {block_idx + 1}')
    ax.set_title(
        f'Block {block_idx + 1}\nEntropy: {entropy_float_all:.2f}, Unique: {unique_values_count}')
    ax.set_xlabel('Data Points (Index)')
    ax.set_ylabel('Values')
    ax.legend()

# Hide any unused subplots
for i in range(num_blocks, len(axes)):
    axes[i].axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the plot
plt.savefig(f"{dataset_name}_blocks_plot_with_entropy_unique.png")

# Additional plot: Frequency of unique values in block 5
block_5_idx = 4  # Block 5 corresponds to index 4 (zero-based indexing)
start_idx_block_5 = block_5_idx * block_size
end_idx_block_5 = start_idx_block_5 + block_size
block_5_data = group_f[start_idx_block_5:end_idx_block_5]

# Calculate the frequency of unique values in block 5
unique_values, counts = np.unique(block_5_data, return_counts=True)

# Combine unique values and their counts into a sorted array by frequency
sorted_indices = np.argsort(-counts)  # Sort in descending order
sorted_unique_values = unique_values[sorted_indices]
sorted_counts = counts[sorted_indices]

# Limit to the top 50 most frequent values
top_n = 50
top_unique_values = sorted_unique_values[:top_n]
top_counts = sorted_counts[:top_n]

# Convert x-axis labels to string (considering them as text for better readability)
top_unique_values_str = [str(val) for val in top_unique_values]

# Plot the frequency of the top 50 unique values in block 5 with log scale on y-axis
plt.figure(figsize=(16, 8))  # Increase figure size for better readability
plt.bar(top_unique_values_str, top_counts, width=0.5, color='blue')

# Set the y-axis to a logarithmic scale
plt.yscale('log')

# Set the ticks for the x-axis as categorical text labels
plt.xticks(top_unique_values_str, rotation=90, fontsize=10)  # Rotate and set font size for readability

# Set labels and title
plt.xlabel('Unique Values (Top 50)')
plt.ylabel('Log Frequency')
plt.title(f'Top 50 Most Frequent Unique Values in Block 5 (Log Scale)')

# Adjust the layout to prevent overlapping of labels
plt.tight_layout()

# Save the plot for block 5
plt.savefig(f"{dataset_name}_block_5_top_50_unique_values_log_frequency_readable_text.png")

# Show the second plot
plt.show()

# Additional plot: Show number of segments for top 10 most frequent values
top_10_unique_values = sorted_unique_values[:10]

# Count the number of segments for each of the top 10 values
top_10_segments = [count_segments(block_5_data, val) for val in top_10_unique_values]

# Convert x-axis labels to string (considering them as text for better readability)
top_10_unique_values_str = [str(val) for val in top_10_unique_values]

# Plot the number of segments for the top 10 most frequent unique values
plt.figure(figsize=(16, 8))
plt.bar(top_10_unique_values_str, top_10_segments, color='green')

# Set labels and title
plt.xlabel('Unique Values (Top 10)')
plt.ylabel('Number of Segments')
plt.title('Number of Segments for Top 10 Most Frequent Unique Values in Block 5')

# Adjust the layout to prevent overlapping of labels
plt.tight_layout()

# Save the plot for the number of segments
plt.savefig(f"{dataset_name}_block_5_top_10_segments.png")

# Show the third plot
plt.show()

# Additional plot: Show max consecutive lengths for top 10 most frequent values
top_10_max_consecutive_lengths = [max_consecutive_length(block_5_data, val) for val in top_10_unique_values]

# Plot the max consecutive lengths for the top 10 most frequent unique values
plt.figure(figsize=(16, 8))
plt.bar(top_10_unique_values_str, top_10_max_consecutive_lengths, color='orange')

# Set labels and title
plt.xlabel('Unique Values (Top 10)')
plt.ylabel('Max Consecutive Length')
plt.title('Max Consecutive Lengths for Top 10 Most Frequent Unique Values in Block 5')

# Adjust the layout to prevent overlapping of labels
plt.tight_layout()

# Save the plot for max consecutive lengths
plt.savefig(f"{dataset_name}_block_5_top_10_max_consecutive_lengths.png")

# Show the fourth plot
plt.show()

import numpy as np

# Remove top 10 most frequent unique values from Block 5 and calculate how many values remain
remaining_block_5_data = block_5_data[~np.isin(block_5_data, top_10_unique_values)]

# Calculate how many values remain after removal of top 10
remaining_values_count = len(remaining_block_5_data)
EEE=calculate_entropy_float(remaining_block_5_data)
print('EEE=', EEE)

print(f"Number of remaining values after removing the top 10: {remaining_values_count}")

