import numpy as np
from scipy.stats import entropy

# Step 1: Generate a binary pattern with equal probability for 0 and 1
# Generate 100 random bits (0s and 1s) with 50% probability for each
high_entropy_pattern = np.random.choice([0, 1], size=100, p=[0.5, 0.5])

# Step 2: Convert the binary pattern to a string format (optional)
binary_pattern_str = ''.join(map(str, high_entropy_pattern))

# Step 3: Calculate the frequency of 0s and 1s
count_zeros = np.sum(high_entropy_pattern == 0)
count_ones = np.sum(high_entropy_pattern == 1)

# Step 4: Compute probabilities of 0s and 1s
total_bits = len(high_entropy_pattern)
p_zeros = count_zeros / total_bits
p_ones = count_ones / total_bits

# Step 5: Create a probability distribution and calculate entropy
probabilities = [p_zeros, p_ones]
entropy_value = entropy(probabilities, base=2)  # Entropy in bits

# Print the binary pattern and entropy result
print("High-Entropy Binary Pattern:", binary_pattern_str)
print(f"Entropy of the binary pattern: {entropy_value:.2f} bits")
