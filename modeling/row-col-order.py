import numpy as np

# Create a 2x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Flatten the array in row-major order (default, 'C')
row_major = A.flatten(order='C')

# Flatten the array in column-major order ('F')
col_major = A.flatten(order='F')

print("Original array:")
print(A)
print("\nFlattened in row-major order ('C'):")
print(row_major)
print("\nFlattened in column-major order ('F'):")
print(col_major)
