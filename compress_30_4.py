
from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import zlib
import numpy as np
import math

def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array
def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

def float_to_bin1(value, m, n):
    """Convert a float value to a binary string."""
    # Convert float value to its IEEE 754 binary representation
    binary_representation = format(struct.unpack('!I', struct.pack('!f', value))[0], '032b')
    # Remove the leading '0b' and zero fill it to the desired length
    return binary_representation.zfill(m * n)

def bin_to_image(b):
    img = []
    #print(b)
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    
    return img
def generate_patterns_from_data(data,m,n):
    
    pattern_set = set()
    
    
    unique_patterns = []
    
    for i in range(0, len(data), m):
        
        for j in range(0,len(data[0]) ,n):
            
            pattern = [data[i + r][j:j + n] for r in range(m)]
                       
            pattern_tuple = tuple(map(tuple, pattern))
                       
            if pattern_tuple not in pattern_set:
                unique_patterns.append(pattern)
                pattern_set.add(pattern_tuple)
              
       
    return unique_patterns

def generate_patterns_from_data_all(data, m, n):
    
    pattern_set = set()
        
    unique_patterns = []
    
    
    for i in range(len(data) - m + 1):
        
        for j in range(len(data[0]) - n + 1):
            # Extract a sub-pattern of size m x n
            pattern = [data[i + r][j:j + n] for r in range(m)]
            
            # Ensure the pattern is of size m x n
            if all(len(row) == n for row in pattern):
                # Convert the sub-pattern to a tuple to use in the set
                pattern_tuple = tuple(map(tuple, pattern))
                
               
                if pattern_tuple not in pattern_set:
                    unique_patterns.append(pattern)
                    pattern_set.add(pattern_tuple)
    
    return unique_patterns


 

def hash_pattern(pattern):
    # Convert the pattern array to a string representation
    return hash(pattern.tostring())

def get_pattern_occurance_non_overlapping(mat, pattern_list,lookup_table):
    #print("mat",mat)
    pattern_occurance = {}
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
    #print(pm,pn)
    matched_binary_representations = [] 
    # Calculate hash values for patterns
    pattern_hashes = [hash_pattern(pattern) for pattern in pattern_list]
    
    # Go over each row in the matrix
    for i in range(0, mat.shape[0], pm):
        #print("i",i)
        #print("mat.shape[0]",mat.shape[0])
        j = 0
        # Go over each column in the matrix
        for j in range(0, mat.shape[1], pn):
        #while j < mat.shape[1]:
           # print("j",j)
            # Check for patterns at (i, j)
            for k, pattern in enumerate(pattern_list):
                #print(k,"pattern",pattern)
                pattern_hash = pattern_hashes[k]
                #pm, pn = pattern.shape[0], pattern.shape[1]
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if hash_pattern(mat[i:i + pm, j:j + pn]) == pattern_hash:
                        pattern_occurance[pattern_hash] = pattern_occurance.get(pattern_hash, 0) + 1
                        compressed_value = lookup_table[tuple(map(tuple, mat[i:i + pm, j:j + pn]))]
                        matched_binary_representations.append(compressed_value)
                        #j += pn
                        break
                else:
                    # Pad mat with zeros
                    slice = mat[i:i + pm, j:j + pn]
                    slice_mat = np.pad(slice, ((0, pm - slice.shape[0]), (0, pn - slice.shape[1])), 'constant')
                    if hash_pattern(slice_mat) == pattern_hash:
                        pattern_occurance[pattern_hash] = pattern_occurance.get(pattern_hash, 0) + 1
                        compressed_value = lookup_table[tuple(map(tuple, mat[i:i + pm, j:j + pn]))]         
                        matched_binary_representations.append(compressed_value)
                        #j += pn
                        break
            j += 1

    # Convert hash codes to pattern occurrences
    pattern_occurance_list = [pattern_occurance.get(pattern_hash, 0) for pattern_hash in pattern_hashes]
    
    return pattern_occurance_list,matched_binary_representations

def calculate_min_bits(patterns_list):
    # Determine the maximum index of the patterns
    max_index = len(patterns_list) - 1

    # Calculate the number of bits needed to represent the maximum index
    min_bits = math.ceil(math.log2(max_index + 1))

    return min_bits

def to_n_bit_binary(index, n):
    # Convert an index to an n-bit binary string
    return f"{index:0{n}b}"

def compress_block_based(mat, m, n):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    # a list to store all patterns
    # Generate patterns based on existing data
    patterns_from_data = generate_patterns_from_data(img_array, m, n)
    pattern_list = np.array(patterns_from_data)
    min_bits=calculate_min_bits(pattern_list)
    lookup_table = {
    tuple(map(tuple, pattern)):to_n_bit_binary(index, min_bits)
    for index, pattern in enumerate(pattern_list)}
     #find_pattern_occurrence(img_array,  my_array,m, n)
    pattern_occurance,matched_binary_representations=get_pattern_occurance_non_overlapping(img_array, pattern_list,lookup_table )

    stats['num_patterns'] = len(pattern_list)
    stats['m'] = m
    stats['n'] = n
    
    # total occurence
    sum_all_occurrences = sum(pattern_occurance)
    stats['total_occurrences'] = sum_all_occurrences
    # get the size of each pattern in bit
    num_nz_pattern_occured = sum(np.array(pattern_occurance) > 0)
    stats['num_nz_patterns'] = num_nz_pattern_occured
    #print(num_nz_pattern_occured)
    size_per_pattern = np.log2(num_nz_pattern_occured)  # in bits
    stats['size_per_pattern'] = size_per_pattern
    # round up to bit
    size_per_pattern_bit_roundup = np.ceil(np.log2(num_nz_pattern_occured))  # in bits
    stats['size_per_pattern_bit_roundup'] = size_per_pattern_bit_roundup
    # round up to the nearest integer of multiple of 8 (byte)
    size_per_pattern_byte_roundup = np.ceil(size_per_pattern / 8) * 8
    stats['size_per_pattern_byte_roundup'] = size_per_pattern_byte_roundup

    # uniform coding size in bit #
    size_uniform_code = size_per_pattern * sum_all_occurrences
    size_uniform_bit_roundup = size_per_pattern_bit_roundup * sum_all_occurrences
    size_unifrom_byte_roundup = size_per_pattern_byte_roundup * sum_all_occurrences
    stats['size_uniform_code'] = size_uniform_code
    stats['size_uniform_bit_roundup'] = size_uniform_bit_roundup
    stats['size_unifrom_byte_roundup'] = size_unifrom_byte_roundup

   
    
    return stats, pattern_occurance,matched_binary_representations,pattern_list
###################################################################################
# Example 
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('FingerMovements')
feature_data1 = ts_list[1:11, :,0].reshape(-1)
m, n = 10,8 # Specify the pattern size
str_array = float_to_bin_array(feature_data1)
img_orig = bin_to_image(str_array)
#print("img",img_orig)
#plot_image(img_orig)
img_array = np.array(img_orig)

# Generate patterns based on existing data
stats8, pattern_occurance,matched_binary_representations,pattern_list=compress_block_based(img_array, m, n)
stats=stats8
ideal_compression_ratio_10_8=stats['original_size']/(stats['total_occurrences']*stats['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict_10_8 = stats['original_size'] / ((stats['total_occurrences'] * stats['size_per_pattern_bit_roundup']) + (stats['num_nz_patterns'] * stats['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict_10_8",base_compression_ratio_Dict_10_8)
print("ideal_compression_ratio_10_8:",ideal_compression_ratio_10_8)
###############################
m, n = 10,16
stats16, pattern_occurance,matched_binary_representations,pattern_list=compress_block_based(img_array, m, n)
stats=stats16
ideal_compression_ratio_10_16=stats['original_size']/(stats['total_occurrences']*stats['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict_10_16 = stats['original_size'] / ((stats['total_occurrences'] * stats['size_per_pattern_bit_roundup']) + (stats['num_nz_patterns'] * stats['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict_10_16",base_compression_ratio_Dict_10_16)
print("ideal_compression_ratio_10_16:",ideal_compression_ratio_10_16)


# In[ ]:




