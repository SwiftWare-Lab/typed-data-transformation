import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a dataset where each row represents a binary sequence
data = np.array([
    [1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1]
])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Bit1', 'Bit2', 'Bit3', 'Bit4', 'Bit5'])

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title('Correlation Matrix Between Bit Positions Across Sequences')
plt.show()

# Find pairs with high correlation
threshold = 0.8  # Define threshold for 'high' correlation
high_corr_pairs = np.where((correlation_matrix > threshold) & (np.eye(correlation_matrix.shape[0]) == 0))
high_corr_pairs = list(zip(correlation_matrix.index[high_corr_pairs[0]], correlation_matrix.columns[high_corr_pairs[1]]))

print("High Correlation Pairs:")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}")
