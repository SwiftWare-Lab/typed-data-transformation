import numpy as np
import fpzip

# Your original 1D array of float32 values
data = np.array([
    4.997153e-06,
    8.614797e+03,
    2.909218e+24,
    3.023551e+23,
    -3.414799e-38,
    4.109847e-09,
    -5.776737e+11,
    -3.498848e+08,
    2.422167e-39,
    -4.987193e+23,
    6.290306e-35,
    2.030702e+04,
    -1.832967e-24,
    -9.109036e-16,
    -1.253502e-07
], dtype=np.float32)

# Reshape the array to 2D (e.g., shape (1, N))
data_reshaped = data.reshape(1, -1)

# Ensure the array is C-contiguous (important for FPZIP)
data_reshaped = np.ascontiguousarray(data_reshaped)

# Compress the data using FPZIP
compressed_data = fpzip.compress(data_reshaped)

# Decompress the data
decompressed_data = fpzip.decompress(compressed_data)

# Verify if the decompressed data matches the original
if np.allclose(data_reshaped, decompressed_data):
    print("Decompression successful, the data matches the original!")
else:
    print("Decompression failed, the data does not match the original.")

# Calculate sizes and compression ratio
original_size = data_reshaped.nbytes
compressed_size = len(compressed_data)
compression_ratio = original_size / compressed_size

print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compression_ratio:.2f}")
