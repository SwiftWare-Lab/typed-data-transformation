import pickle
import matplotlib.pyplot as plt

# Reading the dictionary back from the pickle file
with open('rsim_f32.pkl', 'rb') as pickle_file:
    dict_data_loaded = pickle.load(pickle_file)

dict_in = dict_data_loaded

# Function to delta encode dictionary values
# Function to delta encode dictionary values
def delta_encode_dict_values(dict_in):
    # Sort the dictionary by values and extract the sorted values
    sorted_items = sorted(dict_in.items(), key=lambda item: item[1])
    sorted_values = [item[1] for item in sorted_items]
    deltas = delta_encode(sorted_values)
    return deltas, sorted_items


# Delta Encoding Function
def delta_encode(data):
    if not data:
        return []  # Return an empty list if data is empty

    deltas = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        deltas.append(data[i] - data[i - 1])
    return deltas

# Delta Decoding Function for verification
def delta_decode(deltas):
    if not deltas:
        return []

    data = [deltas[0]]
    for i in range(1, len(deltas)):
        data.append(data[-1] + deltas[i])
    return data

# Apply delta encoding and decoding
encoded_deltas,sorted_original_items = delta_encode_dict_values(dict_in)
decoded_values = delta_decode(encoded_deltas)  # Generate decoded values
#
# Check if the decoded values match the original sorted values
original_sorted_values = [item[1] for item in sorted_original_items]
verification_result = original_sorted_values == decoded_values

print("Verification of Decoded Values Match the Original Sorted Values:", verification_result)


# Calculate differences of consecutive values
differences = [decoded_values[i] - decoded_values[i - 1] for i in range(1, len(decoded_values))]

# Plotting the original and decoded values
indices = range(len(decoded_values))  # Indices for x-axis
plt.figure(figsize=(14, 7))  # Set figure size

# Plot 1: Decoded Values
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(indices, decoded_values, 'bo-', label='Decoded Values')  # Line plot with markers
plt.title('Codeword Indices vs. Values(rsim_f32)')
plt.xlabel('Codewords (Index)')
plt.ylabel('Values')
plt.grid(True)  # Enable grid for better readability
plt.legend()

# Plot 2: Differences Between Consecutive Values
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(range(1, len(differences) + 1), differences, 'ro-', label='Differences')  # Line plot with markers
plt.title('Codeword Indices vs. Differences Between Consecutive Values (rsim_f32')
plt.xlabel('Codewords (Index)')
plt.ylabel('Difference (Consecutive Values)')
plt.grid(True)  # Enable grid for better readability
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('rsim_f32_delta.jpg', format='jpg', dpi=300)  # Save as JPEG with high resolution
plt.show()
