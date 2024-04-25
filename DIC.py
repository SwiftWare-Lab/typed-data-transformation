#!/usr/bin/env python
# coding: utf-8

# In[136]:


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


 

def get_pattern_occurance_non_overlapping(mat, pattern_list):
   
    pattern_occurance = [0] * len(pattern_list)
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
    # go over each row in the matrix
    i, j = 0, 0
    # while i < mat.shape[0]:  #
    for i in range(0, mat.shape[0], pm):
        # go over each column in the matrix
        j = 0
        while j < mat.shape[1]:  # for j in range(0, mat.shape[1], pn):
            
            for k in range(len(pattern_list)):
                pattern = pattern_list[k]
                
                pm, pn = pattern.shape[0], pattern.shape[1]
                
                # check if the pattern is in the matrix
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if np.array_equal(mat[i:i + pm, j:j + pn], pattern):
                        pattern_occurance[k] += 1
                        j += pn
                        #i += pm
                        break
                else:
                    # pad mat with zeros
                    
                    slice = mat[i:i + pm, j:j + pn]
                    
                    # pad slice with zeros to be a multiple of pattern size
                    slice_mat = np.pad(slice, ((0, pm - slice.shape[0] ), (0, pn - slice.shape[1] )), 'constant')
                    if np.array_equal(slice_mat, pattern):
                        pattern_occurance[k] += 1
                        j += pn
                       
                        break

 
    return pattern_occurance
###################################################################################
# Example 
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')
feature_data1 = ts_list[1:11, 200:800, 0].reshape(-1)
m, n = 10, 8 # Specify the pattern size
str_array = float_to_bin_array(feature_data1)
img_orig = bin_to_image(str_array)
#print("img",img_orig)
#plot_image(img_orig)
img_array = np.array(img_orig)

# Generate patterns based on existing data
patterns_from_data = generate_patterns_from_data(img_array, m, n)
my_array = np.array(patterns_from_data)
#find_pattern_occurrence(img_array,  my_array,m, n)
pattern_occurance=get_pattern_occurance_non_overlapping(img_array,  my_array )
#print("matched_binary_representations",matched_binary_representations)
#print("Patterns based on existing data:")
pattern_list=pattern_occurance
stats = {}
stats['original_size'] = img_array.shape[0] * img_array.shape[1]
stats['num_patterns'] = len(pattern_list)
stats['m'] = m
stats['n'] = n
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
ideal_compression_ratio=stats['original_size']/(stats['total_occurrences']*stats['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict = stats['original_size'] / ((stats['total_occurrences'] * stats['size_per_pattern_bit_roundup']) + (stats['num_nz_patterns'] * stats['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict",base_compression_ratio_Dict)
print("ideal_compression_ratio:",ideal_compression_ratio)


# In[ ]:





# In[ ]:




