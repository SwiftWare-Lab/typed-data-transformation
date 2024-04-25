#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import zlib
import numpy as np
import math

def get_pattern_occurance_non_overlapping(mat, pattern_list):
    # a list to store the occurance of each pattern
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

def generate_patterns(m, n):
    # a list to store all patterns
    pattern_list = []
    # go over all possible patterns
    for i in range(2 ** (m * n)):
        # convert i to binary
        bin_i = bin(i)[2:].zfill(m * n)
        pattern = np.array([int(i) for i in bin_i]).reshape((m, n))
        # add pattern to pattern_list
        pattern_list.append(pattern)
    
    return pattern_list

def generate_patterns_from_data(feature_data, m, n):
    # Get unique values from the feature data
    unique_values = np.unique(feature_data)
    
    
    unique_patterns = []
    
    
    for value in unique_values:
        
        bin_value = float_to_bin(value)
        
        bin_value = bin_value.zfill(m * n)
        # Split the binary representation into chunks of size m * n
        bin_chunks = [bin_value[i:i + m * n] for i in range(0, len(bin_value), m * n)]
        # Convert each chunk into a pattern
        for chunk in bin_chunks:
            pattern = np.array([int(bit) for bit in chunk]).reshape((m, n))
            # Add pattern to unique_patterns
            unique_patterns.append(pattern)
    
    return unique_patterns
# convert a floating point number to a binary string
def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

# convert a binary string to a floating point number
def bin_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]
# convert an array binary strings to a bit map image
def bin_to_image(b):
    img = []
    #print(b)
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    
    return img
def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array


# plot a bit map image
def plot_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()

# it takes a bit representation of a float32 number and estimates the compressed size
def compress_block_based(mat, m, n):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    # a list to store all patterns
    pattern_list1 = generate_patterns(m, n)
    pattern_list = np.array(pattern_list1)
    #pattern_list= generate_patterns_from_data(feature_data, m, n)
    #print("pattern_list",pattern_list[1:20])
    stats['num_patterns'] = len(pattern_list)
    stats['m'] = m
    stats['n'] = n
    # get the occurance of each pattern in the matrix
    pattern_occurance = get_pattern_occurance_non_overlapping(mat, pattern_list)
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

    # non-uniform coding using huffman
    dic_pattern = {}
    
    for i in range(len(pattern_occurance)):
        if pattern_occurance[i] > 0:
            
            dic_pattern[str(pattern_list[i])] = pattern_occurance[i]
    
      
    patterns_list = list(dic_pattern.keys())
   
    cleaned_patterns = []
    for pattern in patterns_list:
    
         cleaned_string = pattern.replace('[', '').replace(']', '').replace('\n', '')
         cleaned_patterns.append(cleaned_string)

            
    # Split the cleaned strings into rows
    rows = [pattern.split() for pattern in cleaned_patterns]

    # Convert the rows to integers
    matrix = [[int(cell) for cell in row] for row in rows]

    # Convert the list of lists to a NumPy array
    patterns_list1 = np.array(matrix)
    #print(patterns_list1)
    #######################################
  
   # print("pattern_list",patterns_list )
    processed_patterns_list = [pattern.replace('[', '').replace(']', '').replace('\n', '').replace(' ', '') for pattern in patterns_list]

    processed_patterns_list = [int(item) for item in processed_patterns_list]
    
    #print("processed_patterns_list",processed_patterns_list)
    binary_representation_list = create_binary_representation(processed_patterns_list)
    
    
    binary_representation_list1 = [int(item) for item in binary_representation_list]
    #print("Binary representation list:", binary_representation_list1)
    processed_patterns_list1 = [np.array(pattern) for pattern in patterns_list]
    matched_binary_representations =find_pattern_occurrence(mat, binary_representation_list, patterns_list1,2,8)
    

    
    return stats, pattern_occurance,matched_binary_representations, patterns_list1,patterns_list,dic_pattern,pattern_occurance

def find_pattern_occurrence(mat, binary_representation_list, pattern_list,pm, pn):
    matched_binary_representations = []  
    
    for i in range(0, mat.shape[0], pm):
        
        j = 0
        while j < mat.shape[1]:
            for k in range(len(pattern_list)):
                pattern_flat = pattern_list[k]
                pattern= np.reshape(pattern_flat, (-1, 2))
                #print("aaaaaaaaaaa",pattern)
                if not isinstance(pattern, np.ndarray):  
                    pattern = np.array(pattern)  
                    
                pm, pn = pattern.shape[0], pattern.shape[1]
                # check if the pattern is in the matrix
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if np.array_equal(mat[i:i + pm, j:j + pn], pattern):
                        matched_binary_representations.append(binary_representation_list[k])
                        j += pn
                        break
                else:
                    # pad mat with zeros
                    slice_mat = np.pad(mat[i:i + pm, j:j + pn], ((0, pm - mat[i:i + pm, j:j + pn].shape[0]), (0, pn - mat[i:i + pm, j:j + pn].shape[1])), 'constant')
                    if np.array_equal(slice_mat, pattern):
                        matched_binary_representations.append(binary_representation_list[k])
                        j += pn
                        break
            else:
                j += 1

                
    return matched_binary_representations
def calculate_min_bits(patterns_list):
    # Determine the maximum index of the patterns
    max_index = len(patterns_list) - 1

    # Calculate the number of bits needed to represent the maximum index
    min_bits = math.ceil(math.log2(max_index + 1))

    return min_bits

def create_binary_representation(patterns_list):
    min_bits = calculate_min_bits(patterns_list)
    binary_representation_list = []

    for i, pattern in enumerate(patterns_list):
        # Convert the index to binary representation
        binary_representation = bin(i)[2:].zfill(min_bits)
        binary_representation_list.append(binary_representation)
    #print("patternlist",patterns_list)
   # print("binary_representation_list",binary_representation_list)
    
    return binary_representation_list



def float_to_exp_mant_arrays(a):
    exp_array = []
    mant_array = []
    for f in a:
        bin_str = float_to_bin(f)
        exp_array.append(bin_str[:8])  # First 8 bits represent the exponent
        mant_array.append(bin_str[8:])  # Remaining bits represent the mantissa
    return exp_array, mant_array
def consecutive_differences(float_list):
    # Calculate the differences between consecutive elements
    differences = [float_list[i + 1] - float_list[i] for i in range(len(float_list) - 1)]
    return differences
#############################full################################
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')



feature_data1 = ts_list[1:10, 400:600, 3].reshape(-1)

str_array = float_to_bin_array(feature_data1)
img_orig = bin_to_image(str_array)
#plot_image(img_orig)
img_array = np.array(img_orig)
m, n =2,8
stats_smooth, list_pattern_smooth,m, patterns_list1,patterns_list,dic_pattern,pattern_occurance = compress_block_based(img_array, m, n)
ideal_compression_ratio=stats_smooth['original_size']/(stats_smooth['total_occurrences']*stats_smooth['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict = stats_smooth['original_size'] / ((stats_smooth['total_occurrences'] * stats_smooth['size_per_pattern_bit_roundup']) + (stats_smooth['num_nz_patterns'] * stats_smooth['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict",base_compression_ratio_Dict)
print("ideal_compression_ratio:",ideal_compression_ratio)
###########################Exponent#################################
exp_array, mant_array = float_to_exp_mant_arrays(feature_data1)

str_array1=exp_array
img_orig1 = bin_to_image(str_array1)
img_array1 = np.array(img_orig1)

m, n =2,4
stats_smooth_e, list_pattern_smooth_e,m_e, patterns_list1_e,patterns_list_e,dic_pattern,pattern_occurance_e =compress_block_based(img_array1, m, n)
ideal_compression_ratio_exp=stats_smooth_e['original_size']/(stats_smooth_e['total_occurrences']
                                                           *stats_smooth_e['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict_e = stats_smooth_e['original_size'] / ((stats_smooth_e['total_occurrences'] * 
                                                                    stats_smooth_e['size_per_pattern_bit_roundup']) + (stats_smooth_e['num_nz_patterns'] * stats_smooth_e['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict_exponent",base_compression_ratio_Dict)
print("ideal_compression_ratio_exponent:",ideal_compression_ratio)

###############################mantssia######################################
str_array=mant_array
#str_array=exp_array
img_orig = bin_to_image(str_array)
#plot_image(img_orig)
img_array = np.array(img_orig)

m, n =2,8

stats_smooth_m, list_pattern_smooth_m,m_m, patterns_list1_m,patterns_list_m,dic_pattern_m,pattern_occurance_m = compress_block_based(img_array, m, n)
ideal_compression_ratio_m=stats_smooth_m['original_size']/(stats_smooth_m['total_occurrences']*stats_smooth_m['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict_m = stats_smooth_m['original_size'] / ((stats_smooth_m['total_occurrences'] * stats_smooth_m['size_per_pattern_bit_roundup']) + (stats_smooth_m['num_nz_patterns'] * stats_smooth_m['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict_Man",base_compression_ratio_Dict_m)
print("ideal_compression_ratio_Man:",ideal_compression_ratio_m)
########################### Calculate the differences between consecutive elements#####################################
diffs=consecutive_differences(feature_data1)
 
str_array_diff = float_to_bin_array(diffs)
img_orig_diff = bin_to_image(str_array_diff)

img_array_diff = np.array(img_orig_diff)
m, n =2,8
stats_smooth_diff, list_pattern_smooth_diff,m, patterns_list1_diff,patterns_list_diff,dic_pattern_diff,pattern_occurance_diff = compress_block_based(img_array_diff, m, n)
ideal_compression_ratio_diff=stats_smooth_diff['original_size']/(stats_smooth_diff['total_occurrences']*stats_smooth_diff['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict_diff = stats_smooth_diff['original_size'] / ((stats_smooth_diff['total_occurrences'] * stats_smooth_diff['size_per_pattern_bit_roundup']) + (stats_smooth_diff['num_nz_patterns'] * stats_smooth_diff['size_per_pattern_bit_roundup'] + 16))
print("base_compression_ratio_Dict_diff",base_compression_ratio_Dict_diff)
print("ideal_compression_ratio_diff:",ideal_compression_ratio_diff)   


# In[ ]:




