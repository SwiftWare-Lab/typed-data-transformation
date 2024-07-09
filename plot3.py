import pickle
import matplotlib.pyplot as plt

# Reading the dictionary back from the pickle file
with open('num_brain_f64.pkl', 'rb') as pickle_file:
    dict_data_loaded = pickle.load(pickle_file)

dict_in = dict_data_loaded

def delta_encode_dict_values(dict_in):
    # Sort the dictionary by values and extract the sorted values
    sorted_items = sorted(dict_in.items(), key=lambda item: item[1])
    sorted_values = [item[1] for item in sorted_items]
    deltas = delta_encode(sorted_values)
    return deltas, sorted_items

# Delta Encoding Function
def delta_encode(data):
    if not data:
        return []
    deltas = [data[0]]
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
encoded_deltas, sorted_original_items = delta_encode_dict_values(dict_in)
decoded_values = delta_decode(encoded_deltas)  # Generate decoded values

# Check if the decoded values match the original sorted values
original_sorted_values = [item[1] for item in sorted_original_items]
verification_result = original_sorted_values == decoded_values

print("Verification of Decoded Values Match the Original Sorted Values:", verification_result)

# Calculate differences of consecutive values starting from index 100000
differences = [decoded_values[i] - decoded_values[i - 1] for i in range(100000, len(decoded_values))]

plt.figure(figsize=(14, 7))

# Plot 1: Decoded Values (starting from index 100000)
plt.subplot(2, 1, 1)
plt.plot(range(100000, len(decoded_values)), decoded_values[100000:], 'bo-', label='Decoded Values (Starting from Index 100000)')
plt.title('Codeword Indices vs. Values (rsim_f32) - Starting from Index 100000')
plt.xlabel('Codewords (Index)')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

# Plot 2: Differences Between Consecutive Values (starting from index 100000)
ax2 = plt.subplot(2, 1, 2)
ax2.plot(range(100000, 100000 + len(differences)), differences, 'ro-', label='Differences (Starting from Index 100000)')
ax2.set_yscale('log')  # Set logarithmic scale
plt.title('Codeword Indices vs. Differences Between Consecutive Values (Starting from Index 100000) - Log Scale')
plt.xlabel('Codewords (Index)')
plt.ylabel('Difference (Consecutive Values)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('rsim_f32_adjusted_both_plots.jpg', format='jpg', dpi=300)
plt.show()
