#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import snappy
import struct
from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import fpzip 
import gzip
import zlib
import zstandard as zstd
import numpy as np
import lz4.frame
import matplotlib.pyplot as plt
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('FingerMovements')
feature_data1 = ts_list[1:11, :, 0].reshape(-1)

# Convert floating-point data to bytes
data_bytes = feature_data1.astype(np.float32).tobytes()

# Calculate the size of the original data
original_size = len(data_bytes)

# Compress the data
compressed_data = snappy.compress(data_bytes)

# Calculate the size of the compressed data
compressed_size = len(compressed_data)

# Calculate the compression ratio
compression_ratio_snappy = original_size / compressed_size


print("Compression_ratio(snappy):", compression_ratio_snappy)

#################################################

class GorillaCompressor:
    def __init__(self):
        self.compressed_data = []
        self.prev_value = None
        self.compressed_size_bits = 0

    def compress(self, value):
        
        if not isinstance(value, float):
            raise ValueError(f"Expected a float, got {type(value)} with value {value}")
        if self.prev_value is None:
            self.compressed_data.append(float_to_bits(value))
            self.compressed_size_bits += 64
        else:
            xor_result = float_to_bits(value) ^ float_to_bits(self.prev_value)
            if xor_result == 0:
                self.compressed_data.append(0)
                self.compressed_size_bits += 1
            else:
                self.compressed_data.append(1)
                self.compressed_data.append(xor_result)
                self.compressed_size_bits += 1 + 64
        self.prev_value = value

    def get_compressed_data(self):
        return self.compressed_data

    def get_compression_ratio(self):
        original_size_bits = len(self.compressed_data) * 64
        return original_size_bits / self.compressed_size_bits

class GorillaDecompressor:
    def __init__(self, compressed_data):
        self.compressed_data = compressed_data
        self.index = 0

    def decompress(self):
        decompressed_data = []
        prev_value = None

        while self.index < len(self.compressed_data):
            if prev_value is None:
                value = bits_to_float(self.compressed_data[self.index])
                decompressed_data.append(value)
                prev_value = value
                self.index += 1
            else:
                indicator = self.compressed_data[self.index]
                self.index += 1
                if indicator == 0:
                    decompressed_data.append(prev_value)
                else:
                    xor_result = self.compressed_data[self.index]
                    self.index += 1
                    value = bits_to_float(float_to_bits(prev_value) ^ xor_result)
                    decompressed_data.append(value)
                    prev_value = value

        return decompressed_data
    
def compress_float_fpzip(input_data, precision=0):
    
    if not isinstance(input_data, np.ndarray) or not np.issubdtype(input_data.dtype, np.floating):
        raise TypeError("FPZIP compression requires input data to be a numpy floating-point array.")
    
    compressed_data = fpzip.compress(input_data, precision=precision)
    compressed_size = len(compressed_data)
    
    
    return compressed_data

def float_to_bits(f):
    return struct.unpack('>Q', struct.pack('>d', f))[0]


##############################################################

# Compress the data
compressed_data = compress_float_fpzip(feature_data1)

compressed_size = len(compressed_data)
original_size = feature_data1.nbytes if isinstance(feature_data1, np.ndarray) else len(data)
compression_ratio_fpzip = original_size / compressed_size if compressed_size else 1
print("compression_ratio(fpzip)",compression_ratio_fpzip)
###################################################
compressor = GorillaCompressor()
if isinstance(feature_data1, np.ndarray):
    for value in np.nditer(feature_data1):
         compressor.compress(float(value)) 

compressed_data = compressor.get_compressed_data()
compressed_size = len(compressed_data)
original_size = feature_data1.nbytes if isinstance(feature_data1, np.ndarray) else len(data)
compression_ratio_Gorrila = original_size / compressed_size if compressed_size else 1
print("compression_ratio(Gorrila)",compression_ratio_Gorrila)
#################################################
feature_data1_contiguous = np.ascontiguousarray(feature_data1)

# Compress the data
compressed_data = zstd.compress(feature_data1_contiguous)

# Calculate original size
original_size = feature_data1_contiguous.nbytes

# Calculate the size of the compressed data
compressed_size = len(compressed_data)

# Calculate the compression ratio
compression_ratio_zstd = original_size / compressed_size if compressed_size else 1

# Print the compression ratio
print("Compression ratio (zstd):", compression_ratio_zstd)
###########################################################
data_bytes = feature_data1.astype(np.float32).tobytes()

# Compress the data
compressed_data = lz4.frame.compress(data_bytes)

# Calculate original size
original_size = len(data_bytes)

# Calculate compressed size
compressed_size = len(compressed_data)

# Calculate compression ratio
compression_ratio = original_size / compressed_size if compressed_size else 1

# Print compression ratio
print("Compression ratio (LZ4):", compression_ratio)

##############################################

compression_ratios = [ideal_compression_ratio_10_16,ideal_compression_ratio_10_8,base_compression_ratio_Dict_10_16,base_compression_ratio_Dict_10_8,compression_ratio_snappy, compression_ratio_fpzip, compression_ratio_Gorrila, compression_ratio_zstd, compression_ratio]

# Compression method names
compression_methods = ["ideal_block_10_16","ideal_block_10_8","base_block_10_16","base_block_10_8","Snappy", "FPZIP", "Gorilla", "zstd", "LZ4"]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(compression_methods, compression_ratios, color='skyblue')
plt.xlabel('Compression Method')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratios for Different Compression Methods')
plt.xticks(rotation=45)
plt.show()
#####################################################################################3

# Loading the dataset
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('FingerMovements')

# Extracting the first 20 samples and feature 0
tensor = ts_list[1:11, :,0]

# Number of samples and timesteps
num_samples = tensor.shape[0]
timesteps = range(tensor.shape[1])

# Setting up the plot
plt.figure(figsize=(14, 8))

# Plotting each sample separately for feature 0
for i in range(num_samples):
    plt.plot(timesteps, tensor[i], label=f'Sample {i + 1}')

# Customizing the plot
plt.title('Trend of Feature 0 for Each Sample')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend()
plt.show()

