#!/usr/bin/env python
# coding: utf-8

# In[66]:


from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import zlib
import numpy as np
from huffman_coding import huffman_codes
import time


def verify_mining(mat, pat_list, occurence):
    total_nnz, mined_nnz = np.count_nonzero(mat), 0
    for i in range(len(pat_list)):
        mined_nnz += np.count_nonzero(pat_list[i]) * occurence[i]
    print("total nnz: ", total_nnz, "mined nnz: ", mined_nnz, "error: ", total_nnz - mined_nnz)


def get_pattern_occurance_non_overlapping1(mat, pattern_list):
    # a list to store the occurance of each pattern
    pattern_occurance = [0] * len(pattern_list)
    print("sizePatternList:",len(pattern_occurance))
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
   
    print("pm,pn",pm,pn)

    # go over each row in the matrix
    i, j = 0, 0
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
                        #i += pm
                        break

    #print(("pattern_occurance",pattern_occurance))
    #print("pattern_list",pattern_list)
    verify_mining(mat, pattern_list, pattern_occurance)
    return pattern_occurance
def get_pattern_occurance_non_overlapping(mat, pattern_list):
    # A dictionary to store the occurrence count and locations of each pattern
    pattern_occurrences = {}

    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
   
    for i in range(0, mat.shape[0], pm):
        for j in range(0, mat.shape[1], pn):
            pattern_found = False
            # Iterate over each pattern in the pattern_list
            for k, pattern in enumerate(pattern_list):
                # Check if the pattern fits in the current slice of the matrix
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if np.array_equal(mat[i:i + pm, j:j + pn], pattern):
                        # If the pattern is found, record the occurrence count and location
                        if k not in pattern_occurrences:
                            pattern_occurrences[k] = {"Occurrence Count": 0, "Locations": []}
                        pattern_occurrences[k]["Occurrence Count"] += 1
                        pattern_occurrences[k]["Locations"].append((i, j))
                        pattern_found = True
                        break
                else:
                    # Pad the slice with zeros to match the pattern size
                    slice_mat = np.pad(mat[i:i + pm, j:j + pn], ((0, pm - slice.shape[0]), (0, pn - slice.shape[1])), 'constant')
                    if np.array_equal(slice_mat, pattern):
                        # If the pattern is found, record the occurrence count and location
                        if k not in pattern_occurrences:
                            pattern_occurrences[k] = {"Occurrence Count": 0, "Locations": []}
                        pattern_occurrences[k]["Occurrence Count"] += 1
                        pattern_occurrences[k]["Locations"].append((i, j))
                        pattern_found = True
                        break
            # If no pattern is found in the current slice, append an empty list
            if not pattern_found:
                for pattern_idx in range(len(pattern_list)):
                    if pattern_idx not in pattern_occurrences:
                        pattern_occurrences[pattern_idx] = {"Occurrence Count": 0, "Locations": []}
                    pattern_occurrences[pattern_idx]["Locations"].append([])

    #verify_mining(mat, pattern_list, [occurrence_data["Occurrence Count"] for occurrence_data in pattern_occurrences.values()])
    print(pattern_occurrences)
    return pattern_occurrences


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
    
    print(pattern_list[1])
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

def compress_block_based(mat, m, n):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    print(mat.shape[0], mat.shape[1])
    # a list to store all patterns
    pattern_list = generate_patterns(m, n)
    print(len(pattern_list))
    stats['num_patterns'] = len(pattern_list)
    stats['m'] = m
    stats['n'] = n
    # get the occurrence of each pattern in the matrix
    pattern_occurrences = get_pattern_occurance_non_overlapping(mat, pattern_list)
    print("pattern_occurrences", pattern_occurrences)
    print("pattern_occurrences-len", len(pattern_occurrences))
    # total occurrence
    sum_all_occurrences = sum([data["Occurrence Count"] for data in pattern_occurrences.values()])
    stats['total_occurrences'] = sum_all_occurrences
    # get the size of each pattern in bit
    num_nz_pattern_occurred = len(pattern_occurrences)
    stats['num_nz_patterns'] = num_nz_pattern_occurred
    # print(num_nz_pattern_occured)
    size_per_pattern = np.log2(num_nz_pattern_occurred)  # in bits
    stats['size_per_pattern'] = size_per_pattern
    # round up to bit
    size_per_pattern_bit_roundup = np.ceil(np.log2(num_nz_pattern_occurred))  # in bits
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

    
    return  pattern_occurrences, pattern_list

def decompress_block_based(pattern_occurrences, pattern_list, original_size):
    reconstructed_matrix = np.zeros(original_size)
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]
    for pattern_idx, data in pattern_occurrences.items():
        if data["Occurrence Count"] > 0:
            pattern = pattern_list[pattern_idx]
            occurrence_count = data["Occurrence Count"]
            for location in data["Locations"]:
                if location:  # If the pattern is found in the location
                    i, j = location
                    reconstructed_matrix[i:i+pm, j:j+pn] = pattern
    return reconstructed_matrix


UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
datasets = ['Cricket', 'CharacterTrajectories', 'Heartbeat','BasicMotions','FingerMovements']
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')
n_samples, n_timesteps, n_features = ts_list.shape
results = []

feature_data = ts_list[:,: , 1].reshape(-1)
print(len(feature_data))
print(feature_data)
str_array = float_to_bin_array(feature_data)
img_orig = bin_to_image(str_array)
plot_image(img_orig)
img_array = np.array(img_orig)
m, n = 16, 2
# Compress the data
start_time = time.time()
pattern_occurance,pattern_list = compress_block_based(img_array, m, n)
compress_time = time.time() - start_time
print("Compression time:", compress_time, "seconds")


original_size = (129276, 32) 
start_time = time.time()
reconstructed_image_array = decompress_block_based(pattern_occurance, pattern_list, img_array.shape)
decompress_time = time.time() - start_time
print("Decompression time:", decompress_time, "seconds")

comparison_result = np.array_equal(img_array, reconstructed_image_array)

# Check if the arrays are equal
if comparison_result:
    print("The original image array and the reconstructed image array are equal.")
else:
    print("The original image array and the reconstructed image array are not equal.")


# In[50]:


reconstructed_image_array 


# In[51]:


img_array


# In[21]:


stats


# In[54]:


import sys

# Calculate the size of pattern_occurrence in bytes
size_pattern_occurrence = sys.getsizeof(pattern_occurance)

# Calculate the size of pattern_list in bytes
size_pattern_list = sys.getsizeof(pattern_list)

# Calculate the size of img_array in bytes
size_img_array = img_array.nbytes

# Print the sizes
print("Size of pattern_occurrence:", size_pattern_occurrence, "bytes")
print("Size of pattern_list:", size_pattern_list, "bytes")
print("Size of img_array:", size_img_array, "bytes")


# In[65]:


UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
datasets = ['Cricket', 'CharacterTrajectories', 'Heartbeat','BasicMotions','FingerMovements']
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')
n_samples, n_timesteps, n_features = ts_list.shape
results = []
print(ts_list.shape)
feature_data = ts_list[:,:, 1].reshape(-1)
print(len(feature_data))

str_array = float_to_bin_array(feature_data)
img_orig = bin_to_image(str_array)
print(img_orig)
plot_image(img_orig)
img_array = np.array(img_orig)


# In[64]:


def bin_to_image(b):
    
    for i in range(min(len(b), 100)):  # Iterate over the rows of the image
        for j in range(min(len(b[0]), 32)):  # Iterate over the columns of the image
            img[i, j] = int(b[i][j])  # Assign the binary value to the corresponding pixel
    return img
plot_image(img_orig)


# In[ ]:




