#!/usr/bin/env python
# coding: utf-8

# In[24]:


from tslearn.datasets import UCR_UEA_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import zlib
import numpy as np
import math
import multiprocessing
import hashlib
from math import ceil

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
    #print(data,m,n)
    
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
    # Serialize the pattern data along with its shape to ensure unique hashes for different patterns or sizes
    pattern_bytes = pattern.tostring() + str(pattern.shape).encode()
    return hashlib.sha256(pattern_bytes).hexdigest()

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
               # print(k,"pattern",pattern)
                pattern_hash = pattern_hashes[k]
                
                pm, pn = pattern.shape[0], pattern.shape[1]
               # print("i + pm",i + pm ,"mat.shape[0]", mat.shape[0]) 
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if hash_pattern(mat[i:i + pm, j:j + pn]) == pattern_hash:
                        pattern_occurance[pattern_hash] = pattern_occurance.get(pattern_hash, 0) + 1
                        print("mat[i:i + pm, j:j + pn]",mat[i:i + pm, j:j + pn])
                        compressed_value = lookup_table[tuple(map(tuple, mat[i:i + pm, j:j + pn]))]
                        print("compressed_value",compressed_value)
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
   # print(max_index)

    # Calculate the number of bits needed to represent the maximum index
    min_bits = math.ceil(math.log2(max_index + 1))

    #print(min_bits)
    return min_bits

def to_n_bit_binary(index, n):
    # Convert an index to an n-bit binary string
    return f"{index:0{n}b}"


def compress_block_based(mat, m, n,pattern_list,lookup_table):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    # a list to store all patterns
    # Generate patterns based on existing data
    
    
     #find_pattern_occurrence(img_array,  my_array,m, n)
    pattern_occurance,matched_binary_representations=get_pattern_occurance_non_overlapping(mat, pattern_list,lookup_table )

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

   
    
    return stats, pattern_occurance,matched_binary_representations,pattern_list,lookup_table




def reverse_lookup_table(lookup_table):
    # This will ensure the last inserted item for each hash will be used (if collisions occur, consider revising the hash function or logic)
    reverse_table = {}
    for k, v in lookup_table.items():
        if v in reverse_table:
            # If collision, we replace it, but you should ideally check here or log it to handle issues
            print(f"Collision detected for hash {v}!")
        reverse_table[v] = k
    return reverse_table

def decompress_data(binary_representations, reverse_table, img_shape, m, n, num_cols):
    # Calculate the number of blocks that should be filled based on image dimensions and block size
    expected_blocks = (ceil(img_shape[0] / m) * ceil(img_shape[1] / n))
    print(f"Expected number of blocks: {expected_blocks}, Actual number of blocks: {len(binary_representations)}")

    # Initialize an empty matrix for the reconstructed image
    reconstructed_img = np.zeros(img_shape, dtype=int)
    block_index = 0

    # Iterate over the image in steps of block size
    for i in range(0, img_shape[0], m):
        for j in range(0, img_shape[1], n):
            # Check if there are no more blocks to process
            print("binary_representations",binary_representations)
            print(len(binary_representations))
            if block_index >= len(binary_representations):
                print(f"Warning: Block index {block_index} out of range, stopping decompression.")
                return reconstructed_img  # Return the partially reconstructed image

            # Fetch the binary code and look up the corresponding pattern
            binary = binary_representations[block_index]
            print("binary",binary)
            if binary in reverse_table:
                pattern = np.array(reverse_table[binary])
            else:
                pattern = np.zeros((m, n))  # Use zero matrix if no pattern is found
                print(f"Warning: No pattern found for binary code {binary}")

            # Calculate the actual size of the block to be placed, handling edge cases
            actual_m = min(m, img_shape[0] - i)
            
            actual_n = min(n, img_shape[1] - j)
           

            # Place the pattern block in the reconstructed image
            #reconstructed_img[i:i+actual_m, j:j+actual_n] = pattern[:actual_m, :actual_n]
            print(pattern)
            reconstructed_img[i:i+actual_m, j:j+actual_n] = pattern

            # Increment the block index
            block_index += 1

    # Print a message if not all blocks were used (useful for debugging)
    if block_index < len(binary_representations):
        print(f"Warning: Not all blocks were used in decompression. Unused blocks from index {block_index}")

    return reconstructed_img
def process_and_compress_by_chunks(data, chunk_size, m, n, pattern_list):
    
    
    total_rows = data.shape[0]
    compressed_data_all = []
    total_original_size = 0
    total_compressed_size = 0
    min_bits=calculate_min_bits(pattern_list)
    lookup_table = {
    tuple(map(tuple, pattern)):to_n_bit_binary(index, min_bits)
    for index, pattern in enumerate(pattern_list)}
    
    for start_row in range(0, total_rows, chunk_size):
        #print("start_row",start_row)
        end_row = min(start_row + chunk_size, total_rows)
        # Select the chunk of data
        data_chunk = data[start_row:end_row]
        #print("data_chunk",data_chunk)
        
        # Convert float data to binary image
        str_array = float_to_bin_array(data_chunk.flatten())
        img_orig = bin_to_image(str_array)
        img_array = np.array(img_orig)
        total_original_size += img_array.size 
        #print("img_array",img_array)
        # Compress the chunk
        stats, pattern_occurance, matched_binary_representations, _, lookup_table = compress_block_based(img_array, m, n,pattern_list,lookup_table)
        
        # Compute compressed size for the chunk
        min_bits = calculate_min_bits(np.array(pattern_list))
        compressed_size_chunk = len(matched_binary_representations) * min_bits
        total_compressed_size += compressed_size_chunk
        compressed_data_all.extend(matched_binary_representations)
        
    # Calculate compression ratio
    if total_compressed_size > 0:
        compression_ratio = total_original_size / total_compressed_size
    else:
        compression_ratio = 0

    #print("compressed_data_all",compressed_data_all)    
    return {
        'compressed_data_all': compressed_data_all,
        'total_original_size': total_original_size,
        'total_compressed_size': total_compressed_size,
        'matched_binary_representations': compressed_data_all,
        'lookup_table':lookup_table
    }

# Function to calculate compression ratios
def calculate_compression_ratios(result, pattern_list, min_bits):
    ideal_ratio = result['total_original_size'] / result['total_compressed_size']
    base_ratio = result['total_original_size'] / (result['total_compressed_size'] + (pattern_list.shape[0] * pattern_list[0].shape[1] + 16))
    #print((pattern_list.shape[0] * pattern_list[0].shape[1] + 16))
    #print(result['total_compressed_size'])
    lookup_ratio = result['total_original_size'] / (result['total_compressed_size'] + (pattern_list.shape[0] * pattern_list[0].shape[1] + len(pattern_list) * min_bits + 16))
   # print("ideal_ratio",ideal_ratio)
    return ideal_ratio, base_ratio, lookup_ratio

# Function to process a single feature
def process_feature(dataset_name, feature_idx, feature_data,feature_data1):
    # Convert float data to binary image
    # Apply padding to the data to ensure it is divisible by chunk_size
    
    
    
    str_array = float_to_bin_array(feature_data)
    
    img_orig = bin_to_image(str_array)
    img_array = np.array(img_orig)
    
    # Generate patterns
    m, n = 10, 16
    patterns_from_data = generate_patterns_from_data(img_array, m, n)
    pattern_list = np.array(patterns_from_data)
    #print("len(pattern_list)",len(pattern_list))
    # Compress the data
   
    
    result = process_and_compress_by_chunks(feature_data1, chunk_size, m, n, pattern_list)
    
    # Calculate compression ratios
    min_bits = calculate_min_bits(pattern_list)
    ratios = calculate_compression_ratios(result, pattern_list, min_bits)
    
    return dataset_name, feature_idx, ratios,result
########################
def apply_padding(data, chunk_size):
    
    # Calculate the padding needed to make rows divisible by chunk_size
    rows_needed = chunk_size - (data.shape[0] % chunk_size) if data.shape[0] % chunk_size != 0 else 0
    print("rows_needed",rows_needed)
    # Pad along the first dimension only if necessary
    if rows_needed > 0:
        data = np.pad(data, ((0, rows_needed), (0, 0)), mode='constant', constant_values=0)
    return data
def check_table_consistency(lookup_table, reverse_lookup_table):
    # Check if reversing the reverse lookup table gets back the original keys of the lookup table
    reverse_of_reverse = {v: k for k, v in reverse_lookup_table.items()}
    
    # Compare keys from the original lookup table and the reversed reverse lookup table
    if set(lookup_table.keys()) == set(reverse_of_reverse.keys()):
        print("The keys match: Lookup table is consistent with the reverse lookup table.")
        return True
    else:
        print("Keys do not match: There might be a mismatch or error in table creation.")
        return False

# Get a list of all datasets from UCR and AEON repositories
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
UCR_UEA_datasets_list.remove('InsectWingbeat')
UCR_UEA_datasets_list.remove('AtrialFibrillation')
AEON_datasets_list = ['AEON datasets list here']

# Combine both lists
#datasets = [{'name': name, 'type': 'UCR_UEA'} for name in UCR_UEA_datasets_list] + [{'name': name, 'type': 'AEON'} for name in AEON_datasets_list]
chunk_size = 10
datasets =[
 {'name': 'BasicMotions', 'type': 'UCR_UEA'}      ] 

# Prepare a list to store results
results = []

# Multiprocessing pool
max_processes = 6  
pool = multiprocessing.Pool(processes=max_processes)
for dataset_info in datasets:
    dataset_name = dataset_info['name']
    dataset_type = dataset_info['type']
    
    try:
        if dataset_type == 'UCR_UEA':
            ts_data, _, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
            ts_data=ts_data[0:20, 0:1,:]
            
        else:
            # Load dataset from AEON repository
            pass  # Load AEON dataset
        
        # Check if dataset is loaded successfully
        if ts_data is None:
            print(f"Warning: Dataset '{dataset_name}' could not be loaded.")
            continue
        
        # Get dimensions of the loaded dataset
        n_samples, n_timesteps, n_features = ts_data.shape
        
        # Iterate over features sequentially
        for feature_index in range(n_features):
            #feature_data1= apply_padding(ts_data[:, :, feature_index] , chunk_size)
            feature_data = ts_data[:, :, feature_index].reshape(-1)
            feature_data1 = ts_data[:, :, feature_index] 
                    
            
            # Process each feature
            result = process_feature(dataset_name, feature_index, feature_data, feature_data1)
            #####################################################
            # Unpacking the tuple
            (dataset_name, feature_index, compression_ratios, compression_details) = result

            # Accessing elements from the tuple
            print("Dataset Name:", dataset_name)
            print("Feature Index:", feature_index)
            print("Compression Ratios:", compression_ratios)
            # Accessing dictionary items
            compressed_data_all = compression_details['compressed_data_all']
            total_original_size = compression_details['total_original_size']
            total_compressed_size = compression_details['total_compressed_size']
            matched_binary_representations = compression_details['matched_binary_representations']
            lookup_table = compression_details['lookup_table']
            m, n =10, 16
    
            str_array = float_to_bin_array(feature_data)
    
            img_orig = bin_to_image(str_array)
            img_array = np.array(img_orig)
            print("img_array",img_array)
    
    
            reverse_table = reverse_lookup_table(lookup_table)
            #print("reverse_table",reverse_table)

            decompressed_data = decompress_data(matched_binary_representations, reverse_table, img_array.shape, m, n, img_array.shape[1] // n)
            #print("decompressed_data",decompressed_data)
            check_table_consistency(lookup_table, reverse_table)
            
            if np.array_equal(decompressed_data, img_array):
                   print("The decompressed data is identical to the original data.")
            else:
                    print("The decompressed data is not identical to the original data.")
                    differences = np.where(decompressed_data != img_array)
                    different_indices = list(zip(differences[0], differences[1]))
                    if len(different_indices) > 10:  # limit to the first 10 differences
                        different_indices = different_indices[:10]
                        print("Differences found at indices (showing up to 10):")
                        for index in different_indices:
                              print(f"At index {index}, original is {img_array[index]}, decompressed is {decompressed_data[index]}")

           

            results.append(result)
            # Handle result if necessary
            print("ts_data",dataset_name)
    except Exception as e:
        print(f"Error processing dataset '{dataset_name}': {e}")
        continue



#results.append(pool.apply_async(process_feature, args=(dataset_name, feature_index, feature_data,feature_data1)))

# Close the pool and wait for all processes to finish
pool.close()
pool.join()

df = pd.DataFrame({
    'Dataset Name': [result[0] for result in results],
    'Feature Index': [result[1] for result in results],
    'Ideal Ratio': [result[2][0] for result in results],
    'Base Ratio': [result[2][1] for result in results],
    'Lookup Ratio': [result[2][2] for result in results],
})

file_path = '/home/jamalids/Documents/my_dataframe1.csv'

# Save the DataFrame to CSV
df.to_csv(file_path, index=False)  # Set `index=False` if you do not want to include row indices in the CSV
################################################
# Unpacking the tuple
(dataset_name, feature_index, compression_ratios, compression_details) = result

# Accessing elements from the tuple
print("Dataset Name:", dataset_name)
print("Feature Index:", feature_index)
print("Compression Ratios:", compression_ratios)

# Accessing dictionary items
compressed_data_all = compression_details['compressed_data_all']
total_original_size = compression_details['total_original_size']
total_compressed_size = compression_details['total_compressed_size']
matched_binary_representations = compression_details['matched_binary_representations']
lookup_table = compression_details['lookup_table']

# Print some of the details
print("Compressed Data All:", compressed_data_all)
print("Total Original Size:", total_original_size)
print("Total Compressed Size:", total_compressed_size)
print("Matched Binary Representations:", matched_binary_representations)



# In[23]:


# Unpacking the tuple
(dataset_name, feature_index, compression_ratios, compression_details) = result

# Accessing elements from the tuple
print("Dataset Name:", dataset_name)
print("Feature Index:", feature_index)
print("Compression Ratios:", compression_ratios)

# Accessing dictionary items
compressed_data_all = compression_details['compressed_data_all']
total_original_size = compression_details['total_original_size']
total_compressed_size = compression_details['total_compressed_size']
matched_binary_representations = compression_details['matched_binary_representations']
lookup_table = compression_details['lookup_table']

# Print some of the details
print("Compressed Data All:", compressed_data_all)
print("Total Original Size:", total_original_size)
print("Total Compressed Size:", total_compressed_size)
print("Matched Binary Representations:", matched_binary_representations)


# In[ ]:




