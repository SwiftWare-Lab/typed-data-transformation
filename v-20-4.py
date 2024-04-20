#!/usr/bin/env python
# coding: utf-8

# In[42]:



from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import zlib
import numpy as np
import math
import ast

def verify_mining(mat, pat_list, occurence):
    total_nnz, mined_nnz = np.count_nonzero(mat), 0
    for i in range(len(pat_list)):
        mined_nnz += np.count_nonzero(pat_list[i]) * occurence[i]
    #print("total nnz: ", total_nnz, "mined nnz: ", mined_nnz, "error: ", total_nnz - mined_nnz)

# a function to get a matrix and a list of 2D patters and returns the 
#occurance of each pattern in the matrix in a non-overlapping manner
def get_pattern_occurance_non_overlapping(mat, pattern_list):
    # a list to store the occurance of each pattern
    pattern_occurance = [0] * len(pattern_list)
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
    print(pm,pn)
    print(pattern_list[0])
    
    # go over each row in the matrix
    i, j,r,rf = 0, 0,0,0
    # while i < mat.shape[0]:  #
    for i in range(0, mat.shape[0], pm):
        # go over each column in the matrix
        j = 0
        while j < mat.shape[1]:  # for j in range(0, mat.shape[1], pn):
            # check if mat slice is all zero
            #if np.count_nonzero(mat[i:i + pm, j:j + pn]) == 0:
            #    continue
                # go over each pattern in the pattern_list
            for k in range(len(pattern_list)):
                pattern = pattern_list[k]
                
                pm, pn = pattern.shape[0], pattern.shape[1]
                #print(pm,pn)
                # check if the pattern is in the matrix
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if np.array_equal(mat[i:i + pm, j:j + pn], pattern):
                        pattern_occurance[k] += 1
                        j += pn
                        print("pattern",pattern)
                        rf+=1
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
                        print("pattern111",pattern)
                        r+=1
                        #i += pm
                        break

   # verify_mining(mat, pattern_list, pattern_occurance)
    print("r",r)
    print("rf",rf)
    return pattern_occurance


def generate_patterns(m, n):
    # a list to store all patterns
    pattern_list = []
    # go over all possible patterns
    for i in range(2 ** (m * n)):
        # convert i to binary
        bin_i = bin(i)[2:].zfill(m * n)
        
        # skip if all zeros in bin_i
        #if bin_i == '0' * (m * n):
        #    continue
        # convert bin_i to a matrix
        pattern = np.array([int(i) for i in bin_i]).reshape((m, n))
        # add pattern to pattern_list
        pattern_list.append(pattern)
    print(pattern_list[0])
    return pattern_list

# convert a floating point number to a binary string
def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')


# convert a floating point number to a binary string
def xor_float(v1, v2):
    v1_str = format(struct.unpack('!I', struct.pack('!f', v1))[0], '032b')
    v2_str = format(struct.unpack('!I', struct.pack('!f', v2))[0], '032b')
    xor_str = format(int(v1_str, 2) ^ int(v2_str, 2), '032b')
    return xor_str


# convert a binary string to a floating point number
def bin_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]


# convert an array of floating point numbers to an array of binary strings
def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array


# compute all xors of every consecutive float32 number in an array and retuns the array of xors
def compute_xors(a):
    xors = []
    for i in range(len(a) - 1):
        xors.append(xor_float(a[i], a[i + 1]))
    return xors


# convert an array binary strings to a bit map image
def bin_to_image(b):
    img = []
    #print(b)
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    #print("aaaaaaaa")
    #print(img)
    return img


# plot a bit map image
def plot_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


# def generate random array of float32 numbers smoothly increasing from 0 to 1
def generate_smooth_array(n):
    import numpy as np
    a = np.linspace(0, 1, n).astype(np.float32)
    return a

# it takes a bit representation of a float32 number and estimates the compressed size
def compress_block_based(mat, m, n):
    stats = {}
    print("mat",mat)
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    # a list to store all patterns
    pattern_list = generate_patterns(m, n)
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
    print(num_nz_pattern_occured)
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
            #print(pattern_occurance[i])
            dic_pattern[str(pattern_list[i])] = pattern_occurance[i]
    #print("dic_pattern",dic_pattern)
    #print(len(dic_pattern))
    ##################################
    # Get patterns_list from patterns_dict
    patterns_list = list(dic_pattern.keys())
    #patterns_list1 = [pattern.replace(" ", "") for pattern in patterns_list]

    ########################################
    cleaned_patterns = []
    for pattern in patterns_list:
    # Remove the unnecessary characters from the string
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
    

    #print( "m",matched_binary_representations)
    return stats, pattern_occurance,matched_binary_representations, patterns_list1,patterns_list,dic_pattern,pattern_occurance

def find_pattern_occurrence(mat, binary_representation_list, pattern_list,pm, pn):
    matched_binary_representations = []  # Store matched binary representations here
    #pm, pn = 8, 2

    # go over each row in the matrix
    for i in range(0, mat.shape[0], pm):
        # go over each column in the matrix
        j = 0
        while j < mat.shape[1]:
            for k in range(len(pattern_list)):
                pattern_flat = pattern_list[k]
                pattern= np.reshape(pattern_flat, (-1, 2))
                #print("aaaaaaaaaaa",pattern)
                if not isinstance(pattern, np.ndarray):  # Check if pattern is not a NumPy array
                    pattern = np.array(pattern)  # Convert pattern to a NumPy array
                    
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
    print("patternlist",patterns_list)
    print("binary_representation_list",binary_representation_list)
    
    return binary_representation_list

###################################################
def float_to_exp_mant_arrays(a):
    exp_array = []
    mant_array = []
    for f in a:
        bin_str = float_to_bin(f)
        exp_array.append(bin_str[:8])  # First 8 bits represent the exponent
        mant_array.append(bin_str[8:])  # Remaining bits represent the mantissa
    return exp_array, mant_array

# Example usage

################################################
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
datasets = ['Cricket', 'CharacterTrajectories', 'Heartbeat','BasicMotions','FingerMovements']
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')
n_samples, n_timesteps, n_features = ts_list.shape
results = []

feature_data1 = ts_list[0:1, 0:1000, 0].reshape(-1)
exp_array, mant_array = float_to_exp_mant_arrays(feature_data1)
print("exp_array:",exp_array,"mant_array", mant_array )
print(feature_data1)
#feature_data=exp_array
#feature_data=feature_data1
eature_data=mant_array
#str_array = float_to_bin_array(feature_data)
str_array=mant_array
#str_array=exp_array
img_orig = bin_to_image(str_array)
plot_image(img_orig)
img_array = np.array(img_orig)
print("imgarray",img_array)
m, n =2,8
print(img_array.shape[0] * img_array.shape[1])
stats_smooth, list_pattern_smooth,m, patterns_list1,patterns_list,dic_pattern,pattern_occurance = compress_block_based(img_array, m, n)
print(stats_smooth)
        


# In[34]:


feature_data1 = ts_list[0:1, 1:10, 3].reshape(-1)
exp_array, mant_array = float_to_exp_mant_arrays(feature_data1)
print(mant_array)
#print(mant_array.shape[0] * mant_array.shape[1])
img_orig = bin_to_image(mant_array)
img_array = np.array(img_orig)
print(img_array)
print(img_array.shape[0] * img_array.shape[1])


# In[ ]:


Overall {'original_size': 320, 'num_patterns': 256, 
         'm': 2, 'n': 4, 'total_occurrences': 40, 'num_nz_patterns': 26,
         'size_per_pattern': 4.700439718141092, 'size_per_pattern_bit_roundup': 5.0, 
         'size_per_pattern_byte_roundup': 8.0, 'size_uniform_code': 188.01758872564366,
         'size_uniform_bit_roundup': 200.0, 'size_unifrom_byte_roundup': 320.0}

ideal_compression_ratio=1.6
base_compression_ratio_Dict=0.9467455621301775
compression_size(full):200


# In[6]:


exponent(2,4):patternlist [10111011, 11111111]
binary_representation_list ['0', '1']
{'original_size': 80, 'num_patterns': 256, 'm': 2, 'n': 4, 
 'total_occurrences': 10, 'num_nz_patterns': 2, 'size_per_pattern': 1.0, 
 'size_per_pattern_bit_roundup': 1.0, 'size_per_pattern_byte_roundup': 8.0,
 'size_uniform_code': 10.0, 'size_uniform_bit_roundup': 10.0, 'size_unifrom_byte_roundup': 80.0}

ideal_compression_ratio,base_compression_ratio_Dict 8.0 4.0
compression_size(exponent):10


# In[ ]:


mantassia: {'original_size': 240, 'num_patterns': 256, 'm': 2, 'n': 4,
            'total_occurrences': 30, 'num_nz_patterns': 26, 'size_per_pattern': 4.700439718141092,
            'size_per_pattern_bit_roundup': 5.0, 'size_per_pattern_byte_roundup': 8.0, 
            'size_uniform_code': 141.01319154423274, 
            'size_uniform_bit_roundup': 150.0, 'size_unifrom_byte_roundup': 240.0}
ideal_compression_ratio,base_compression_ratio_Dict 1.6 0.8333333333333334
compression_size(mantassia):150
    


# In[43]:


ideal_compression_ratio=stats_smooth['original_size']/(stats_smooth['total_occurrences']*stats_smooth['size_per_pattern_bit_roundup'])
base_compression_ratio_Dict = stats_smooth['original_size'] / ((stats_smooth['total_occurrences'] * stats_smooth['size_per_pattern_bit_roundup']) + (stats_smooth['num_nz_patterns'] * stats_smooth['size_per_pattern_bit_roundup'] + 16))


# In[44]:


print("ideal_compression_ratio,base_compression_ratio_Dict",ideal_compression_ratio,base_compression_ratio_Dict)


# In[25]:


import fpzip
import sys
data=feature_data1.astype(np.float32)
# Get the size of the original data in bytes
#original_size = sys.getsizeof(data)
original_size=img_array.shape[0] * img_array.shape[1]
# Compress data
compressed_data = fpzip.compress(data,0)

# Get the size of the compressed data in bytes
#compressed_size = sys.getsizeof(compressed_data)
compressed_size = len(compressed_data)
compressed_size_bits = compressed_size * 8
# Calculate compression ratio
compression_ratio = original_size / compressed_size

print("Original size:", original_size, "bits")
print("Compressed size(fpzip):", compressed_size, "bits")
print("Compression ratio(fpzip):", compression_ratio)


# In[ ]:


Original size: 320 
Compressed size(fpzip): 56 
compression_size(mantassia):150 
compression_size(exponent):10    
compression_size(full):200    


# In[65]:


import matplotlib.pyplot as plt

# Data
compression_methods = ['Original', 'fpzip', 'Mantissa', 'Exponent', 'Full(Exponent+Mantissa)']
sizes = [320, 56, 150, 10, 200]  # Sizes in bytes

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(compression_methods, sizes, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel('Compression Method')
plt.ylabel('Size (bytes)')
plt.title('Comparison of Compression Sizes')
plt.ylim(0, max(sizes) * 1.2)  # Adjust ylim to leave some space
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:


Compression ratio(fpzip): 5.714285714285714
ideal_compression_ratio(mantassia):1.6
base_compression_ratio_Dict(mantassia):0.8333333333333334  
ideal_compression_ratio(Exponent):8.0
base_compression_ratio_Dict (Exponent): 4.0
ideal_compression_ratio(full)=1.6
base_compression_ratio_Dict(full)=0.9467455621301775    
    


# In[ ]:


#1000


# In[ ]:


{'original_size': 24000, 'num_patterns': 65536, 'm': 2, 'n': 8, 'total_occurrences': 1500, 
 'num_nz_patterns': 909, 'size_per_pattern': 9.828136484194108, 'size_per_pattern_bit_roundup': 10.0,
 'size_per_pattern_byte_roundup': 16.0, 'size_uniform_code': 14742.204726291162,
 'size_uniform_bit_roundup': 15000.0, 'size_unifrom_byte_roundup': 24000.0}

pattern [[0 1 1 0 1 0 0 1]
 [0 1 1 0 1 0 0 1]] 8*2

