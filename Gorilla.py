#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
from scipy.stats import entropy
from io import BytesIO
import gzip
import zlib  
from typing import Optional, Literal
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets
import zlib
import blosc
from aeon.datasets.tsc_data_lists import univariate2015, univariate, multivariate, univariate_equal_length, multivariate_equal_length
from aeon.datasets import load_classification
import zstandard as zstd
import seaborn as sns
from scipy.stats import entropy
from io import BytesIO
import gzip
from io import BytesIO
from skimage.metrics import peak_signal_noise_ratio as psnr
import fpzip  
import struct

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

def float_to_bits(f):
    return struct.unpack('>Q', struct.pack('>d', f))[0]

def bits_to_float(b):
    return struct.unpack('>d', struct.pack('>Q', b))[0]
############################################################################
def nsfl_compress(data, decimal_places=2):
    """Compress floating-point data by truncating to a fixed number of decimal places."""
    compressed_data = [float(Decimal(str(d)).quantize(Decimal(f'1.{"0"*decimal_places}'), 
                                                      rounding=ROUND_DOWN)) for d in data]
    return compressed_data

def nsvl_compress(data, threshold=1.0):
    """Compress floating-point data with variable precision based on value's magnitude."""
    compressed_data = []
    for d in data:
        if abs(d) < threshold:
            # Smaller values get truncated more
            compressed_data.append(round(d, 2))
        else:
            # Larger values retain more precision
            compressed_data.append(round(d, 3))
    return compressed_data


def compress_nsv1(data):
    compressed = []
    count = 0
    for value in data:
        if value == 0:
            count += 1
        else:
            if count > 0:
                compressed.append((0, count))  # Tuple (value, count)
                count = 0
            compressed.append(value)  # Directly store non-zero values
    if count > 0:  # Handle trailing zeros
        compressed.append((0, count))
    return compressed

def decompress_nsv(compressed):
    decompressed = []
    for item in compressed:
        if isinstance(item, tuple):
            decompressed.extend([0] * item[1])  # Extend by zeros
        else:
            decompressed.append(item)
    return decompressed

def detect_data_type(series):
    if pd.api.types.is_integer_dtype(series):
        unique_ratio = series.nunique() / len(series)
        return 'Categorical' if unique_ratio < 0.05 else 'Integer'
    elif pd.api.types.is_float_dtype(series):
        return 'Float'
    else:
        return 'Other'




def compress_float_fpzip(input_data, precision=8):
    
    if not isinstance(input_data, np.ndarray) or not np.issubdtype(input_data.dtype, np.floating):
        raise TypeError("FPZIP compression requires input data to be a numpy floating-point array.")
    
    compressed_data = fpzip.compress(input_data, precision=precision)
    compressed_size = len(compressed_data)
    
    
    return compressed_data


def quantize_data(data, precision=np.float16):
   
    return data.astype(precision)

def quantized_gzip_compression(data, num_levels):
    
    if isinstance(data, bytes):
        raise ValueError("Expected numeric data, received bytes.")
    quantized_data = quantize_data(data)
    data_bytes = quantized_data.tobytes()  
    return gzip_compression(data_bytes)

def gzip_compression(data_bytes):
    
    out = BytesIO()  
    with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(data_bytes)
    return out.getvalue()  

def rle_compression(data_bytes):
    
    if not data_bytes:
        return bytes()
    result = []
    current_byte = data_bytes[0]
    count = 1
    for byte in data_bytes[1:]:
        if byte == current_byte:
            count += 1
        else:
            result.extend((current_byte, count))
            current_byte = byte
            count = 1
    result.extend((current_byte, count))
    
    return bytes(result)

def delta_encoding(data_bytes):
    """Apply delta encoding to a sequence of bytes."""
    deltas = np.diff(np.frombuffer(data_bytes, dtype=np.uint8), prepend=0)
    return deltas.tobytes()

def delta_gzip_compression(data_bytes):
    """Combine delta encoding with gzip compression."""
    delta_encoded = delta_encoding(data_bytes)
    return gzip_compression(delta_encoded)

def blosc_compression(data_bytes):
    """Compress data using blosc."""
    return blosc.compress(data_bytes)

def apply_compression(data, method):
    compressor = GorillaCompressor()
    if method == 'fpzip':
        # Ensure data is a floating-point NumPy array
        if isinstance(data, np.ndarray):
            if not np.issubdtype(data.dtype, np.floating):
                #  convert to a floating-point type if not already
                data = data.astype(np.float32)
        elif isinstance(data, bytes):
            
            raise ValueError("Data for FPZIP compression must not be in bytes format.")
        else:
            # convert other types to a floating-point NumPy array
            try:
                data = np.array(data, dtype=np.float32)
            except Exception as e:
                raise TypeError(f"Failed to convert input to a floating-point numpy array: {e}")

        compressed_data = compress_float_fpzip(data, precision=32)
    else:
        
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise TypeError("Unsupported data type for compression.")

        
        if method == 'gzip':
            compressed_data = gzip_compression(data_bytes)
        elif method == 'nsv':
            compressed_data = compress_nsv(data_bytes)
        
        elif method == 'Gorilla':
        # Check if data is a NumPy array and iterate through its elements
            if isinstance(data, np.ndarray):
                for value in np.nditer(data):
                     compressor.compress(float(value))  # Convert each value to float before compression
            else:
                compressor.compress(data)  # Assume data is already a single float value

            compressed_data = compressor.get_compressed_data()
        else:
            raise ValueError(f"Compression method '{method}' not recognized.")

    compressed_size = len(compressed_data)
    original_size = data.nbytes if isinstance(data, np.ndarray) else len(data)
    compression_ratio = original_size / compressed_size if compressed_size else 1
    print(original_size)
    return compressed_size, compression_ratio

def calculate_metrics_and_select_compression(ts_data, dataset_name, dataset_type):
    if dataset_type == 'UCR_UEA':
        n_samples, n_timesteps, n_features = ts_data.shape
        
    elif dataset_type == 'AEON':
        #n_samples, n_features,n_timesteps = ts_data.shape
        ts_data1=ts_data
        ts_data = np.transpose(ts_data, (0, 2, 1))
        n_samples, n_timesteps, n_features = ts_data.shape
    
    results = []

    for feature_index in range(n_features):
        feature_data = ts_data[:, :, feature_index].reshape(-1)
        data_bytes = feature_data.tobytes()
        
        series = pd.Series(feature_data)
        data_type = detect_data_type(series)
        
        if data_type in ['Integer', 'Float']:
            variance = np.var(feature_data, ddof=1)
            hist, _ = np.histogram(feature_data, bins='auto', density=True)
            feature_entropy = entropy(hist)
        else:
            variance, feature_entropy = np.nan, np.nan
        
        compression_method = select_compression_method(data_type, variance, feature_entropy)
        print(compression_method)
        #compression_method='fpzip'
        # Apply the selected compression method
        if compression_method == 'Gorilla':
    
               compressed_size, compression_ratio = apply_compression(feature_data, compression_method)
        else:
        
               compressed_size ,compression_ratio = apply_compression(data_bytes, compression_method)
        original_size = len(data_bytes)
        
        
        results.append({
            'Dataset Name': dataset_name,
            'Dataset Type': dataset_type,
            'Feature Index': feature_index,
            'Data Type': data_type,
            'Compression Method': compression_method,
            'Original Size (bytes)': original_size,
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Variance': variance,
            'Entropy': feature_entropy,
        })
    
    # Calculate sum of compression ratios and sizes for comparison
    total_original_size = sum(item['Original Size (bytes)'] for item in results)
    total_compressed_size = sum(item['Compressed Size (bytes)'] for item in results)
    overall_compression_ratio = total_original_size / total_compressed_size
    
    summary = {
        'Dataset Name': dataset_name,
        'Dataset Type': dataset_type,
        'Original Size (bytes)': total_original_size,
        'Compressed Size (bytes)': total_compressed_size,
        'Compression Ratio': overall_compression_ratio,
        
    }
    
    
    return pd.DataFrame(results), pd.DataFrame([summary])
###############based on feature and time step #################
def calculate_metrics_and_select_compression_time_feathure(ts_data, dataset_name, dataset_type):
    if dataset_type == 'UCR_UEA':
        n_samples, n_timesteps, n_features = ts_data.shape
    elif dataset_type == 'AEON':
        ts_data = np.transpose(ts_data, (0, 2, 1))  # Assuming you want n_samples, n_features, n_timesteps
        n_samples, n_timesteps, n_features = ts_data.shape
    
    results = []

    for timestep_index in range(n_timesteps):
        for feature_index in range(n_features):
            feature_data = ts_data[:, timestep_index, feature_index].reshape(-1)
            data_bytes = feature_data.tobytes()
            
            series = pd.Series(feature_data)
            data_type = detect_data_type(series)
            
            if data_type in ['Integer', 'Float']:
                variance = np.var(feature_data, ddof=1)
                hist, _ = np.histogram(feature_data, bins='auto', density=True)
                feature_entropy = entropy(hist)
            else:
                variance, feature_entropy = np.nan, np.nan
            
            compression_method = select_compression_method(data_type, variance, feature_entropy)
            
            # Apply the selected compression method
            if compression_method == 'Gorilla':
                compressed_size, compression_ratio = apply_compression(feature_data, compression_method)
            else:
                compressed_size, compression_ratio = apply_compression(data_bytes, compression_method)
            original_size = len(data_bytes)
            results.append({
            'Dataset Name': dataset_name,
            'Dataset Type': dataset_type,
            'Feature Index': feature_index,
            'Timestep Index': timestep_index,    
            'Data Type': data_type,
            'Compression Method': compression_method,
            'Original Size (bytes)': original_size,
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Variance': variance,
            'Entropy': feature_entropy,
        })
            
    # Calculate sum of compression ratios and sizes for comparison
    # Calculate sum of compression ratios and sizes for comparison
    total_original_size = sum(item['Original Size (bytes)'] for item in results)
    total_compressed_size = sum(item['Compressed Size (bytes)'] for item in results)
    overall_compression_ratio = total_original_size / total_compressed_size
    
    summary = {
        'Dataset Name': dataset_name,
        'Dataset Type': dataset_type,
        'Original Size (bytes)': total_original_size,
        'Compressed Size (bytes)': total_compressed_size,
        'Compression Ratio': overall_compression_ratio,
        
    }
    
    
    return pd.DataFrame(results), pd.DataFrame([summary])

           
###########################timesteps
def calculate_metrics_and_select_compression_time(ts_data, dataset_name, dataset_type):
    if dataset_type == 'UCR_UEA':
        n_samples, n_timesteps, n_features = ts_data.shape
    elif dataset_type == 'AEON':
        ts_data = np.transpose(ts_data, (0, 2, 1))  # Assuming you want n_samples, n_features, n_timesteps
        n_samples, n_timesteps, n_features = ts_data.shape
    
    results = []

    for timestep_index in range(n_timesteps):
        
            feature_data = ts_data[:, timestep_index, :].reshape(-1)
            data_bytes = feature_data.tobytes()
            
            series = pd.Series(feature_data)
            data_type = detect_data_type(series)
            
            if data_type in ['Integer', 'Float']:
                variance = np.var(feature_data, ddof=1)
                hist, _ = np.histogram(feature_data, bins='auto', density=True)
                feature_entropy = entropy(hist)
            else:
                variance, feature_entropy = np.nan, np.nan
            
            compression_method = select_compression_method(data_type, variance, feature_entropy)
            
            # Apply the selected compression method
            if compression_method == 'Gorilla':
                compressed_size, compression_ratio = apply_compression(feature_data, compression_method)
            else:
                compressed_size, compression_ratio = apply_compression(data_bytes, compression_method)
            original_size = len(data_bytes)
            results.append({
            'Dataset Name': dataset_name,
            'Dataset Type': dataset_type,
            'Timestep Index': timestep_index,    
            'Data Type': data_type,
            'Compression Method': compression_method,
            'Original Size (bytes)': original_size,
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Variance': variance,
            'Entropy': feature_entropy,
        })
            
    # Calculate sum of compression ratios and sizes for comparison
    # Calculate sum of compression ratios and sizes for comparison
    total_original_size = sum(item['Original Size (bytes)'] for item in results)
    total_compressed_size = sum(item['Compressed Size (bytes)'] for item in results)
    overall_compression_ratio = total_original_size / total_compressed_size
    
    summary = {
        'Dataset Name': dataset_name,
        'Dataset Type': dataset_type,
        'Original Size (bytes)': total_original_size,
        'Compressed Size (bytes)': total_compressed_size,
        'Compression Ratio': overall_compression_ratio,
        
    }
    
    
    return pd.DataFrame(results), pd.DataFrame([summary])
###########################

def select_compression_method(data_type, variance, entropy):
    
    if data_type == 'Integer':
        if variance < 0.2:
            return 'delta+gzip'
        else:
            return 'gzip'
    elif data_type == 'Float':
        if variance > 0.5 or entropy > 0.5:
            return 'Gorilla'
        else:
            return 'blosc'
    elif data_type == 'Categorical':
        if entropy > 1.0:
            return 'gzip'
        else:
            return 'RLE'
    return 'none'
def calculate_overall_metrics_and_compress(ts_data, dataset_name, dataset_type):
    
    # Flatten the entire dataset
    flattened_data = ts_data.reshape(-1)
    flattened_data1=flattened_data
    
    
   
    variance = np.var(flattened_data, ddof=1)
    hist, _ = np.histogram(flattened_data, bins='auto', density=True)
    overall_entropy = entropy(hist)
    
    # Convert flattened data to bytes for compression
    data_bytes = flattened_data.tobytes()

    # Detect  data type in flattened data
    series = pd.Series(flattened_data)
    data_type = detect_data_type(series)

    
    compression_method = select_compression_method(data_type, variance, overall_entropy)
    #compression_method='fpzip'
    if compression_method == 'Gorilla':
        
        compressed_size, compression_ratio = apply_compression(flattened_data, compression_method)
    else:
        compressed_size ,compression_ratio = apply_compression(data_bytes, compression_method)
    
    #compressed_size, compression_ratio = apply_compression(data_bytes, compression_method)

    original_size = len(data_bytes)

    
    results = {
        'Dataset Name': dataset_name,
        'Dataset Type': dataset_type,
        'Data Type': data_type,
        'Compression Method': compression_method,
        'Original Size (bytes)': original_size,
        'Compressed Size (bytes)': compressed_size,
        'Compression Ratio': compression_ratio,
        'Variance': variance,
        'Entropy': overall_entropy
    }

    return pd.DataFrame([results])


def pipeline(
        dataset_type: Literal['UCR_UEA', 'AEON'],
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None
        ) -> pd.DataFrame:
    print(f'Processing dataset: {dataset_name if dataset_name else dataset_path}')

    
    if dataset_path:
        df = pd.read_csv(dataset_path, header=None)
        ts_list = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values
    elif dataset_type == 'UCR_UEA':
        ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
        ts_list1 = ts_list
    elif dataset_type == 'AEON':
        ts_list, y_true = load_classification(dataset_name)
        ts_list1 = ts_list
        ts_list = np.array(ts_list).flatten()
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    
    element_type_description = "Unknown data structure or type"
    
    
    if hasattr(ts_list1, 'ndim'):
        if ts_list1.ndim == 3:
            element_type_description = f"3D array with element type {type(ts_list1[0][0][0])}"
        elif ts_list1.ndim == 2:
            element_type_description = f"2D array with element type {type(ts_list1[0][0])}"
        elif ts_list1.ndim == 1:
            element_type_description = f"1D array with element type {type(ts_list1[0])}"
    elif isinstance(ts_list1, list) and len(ts_list1) > 0:
        if isinstance(ts_list1[0], list):
            element_type_description = f"2D list with element type {type(ts_list1[0][0])}"
        else:
            element_type_description = f"1D list with element type {type(ts_list1[0])}"

    print(f"Type of the dataset: {type(ts_list1)}, Structure and element type: {element_type_description}")
    if ts_list1 is None:
        print(f"Error: Failed to load dataset {dataset_name if dataset_name else dataset_path}.")
        return None
    


    compression_results_df ,compression_results_Summary= calculate_metrics_and_select_compression(ts_list1,dataset_name,dataset_type)
    compression_results_overall =calculate_overall_metrics_and_compress(ts_list1,dataset_name,dataset_type)
    compression_results_df_T ,compression_results_Summary_T=calculate_metrics_and_select_compression_time_feathure(ts_list1,dataset_name,dataset_type)
    compression_results_df_TS,compression_results_Summary_TS=calculate_metrics_and_select_compression_time(ts_list1,dataset_name,dataset_type)
    return compression_results_df,compression_results_Summary,compression_results_overall,compression_results_df_T ,compression_results_Summary_T,compression_results_df_TS,compression_results_Summary_TS
def main():
    UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
    UCR_UEA_datasets_list.remove('InsectWingbeat')
    AEON_datasets_list = list(multivariate)
    AEON_datasets_list.remove('InsectWingbeat')
    AEON_datasets_list.remove('CharacterTrajectories')
    AEON_datasets_list.remove('SpokenArabicDigits')
    AEON_datasets_list.remove('JapaneseVowels')

    #datasets = ['Cricket', 'CharacterTrajectories', 'Heartbeat','BasicMotions','FingerMovements']
    datasets = [{'name': name, 'type': 'UCR_UEA'} for name in UCR_UEA_datasets_list] + [{'name': name, 'type': 'AEON'}
                                                                                        for name in AEON_datasets_list]
               
        
    #dataset_type = 'UCR_UEA'
    #datasets = [{'name': name, 'type': 'UCR_UEA'} for name in UCR_UEA_datasets_list] 
    results = []
    results_sum = []
    result_overal = []
    results_df_T=[]
    results_Summary_T=[]
    results_df_TS=[]
    results_Summary_TS=[]
    
    
    for dataset in datasets:
        print(f"Processing {dataset}")
        pipeline_result = pipeline(
            dataset_name=dataset['name'],
            dataset_type=dataset['type']
        )
        
        if pipeline_result is not None:
        # Use different variable names for the outputs of the pipeline function
            result, result_sum_current, result_overal_current, df_T_current,summary_T_current,df_TS_current, summary_TS_current= pipeline_result
            
        
            results.append(result)
            results_sum.append(result_sum_current)
            result_overal.append(result_overal_current)
        
             # Append the current results to the lists correctly
            results_df_T.append(df_T_current)
            results_Summary_T.append(summary_T_current)
            results_df_TS.append(df_TS_current)
            results_Summary_TS.append(summary_TS_current)
    else:
        print(f"Skipping dataset {dataset} due to errors in processing.")

    return results, results_sum, result_overal,results_df_T,results_Summary_T,results_df_TS,results_Summary_TS

if __name__ == '__main__':
    results, results_sum, result_overal,results_T ,results_sum_T,results_TS ,results_sum_TS= main()
    
    
    df_results = pd.concat(results, ignore_index=True)
    df_results.to_csv('/home/jamalids/Downloads/output/df_results.csv', index=False)


    df_results_sum = pd.concat(results_sum, ignore_index=True)
    df_results_sum.to_csv('/home/jamalids/Downloads/output/df_results_sum.csv', index=False)

    df_result_overal = pd.concat(result_overal, ignore_index=True)
    df_result_overal.to_csv('/home/jamalids/Downloads/output/df_result_overal.csv', index=False)
    
    df_results = pd.concat(results, ignore_index=True)
    df_results.to_csv('/home/jamalids/Downloads/output/df_results.csv', index=False)

    df_results_T = pd.concat(results_T, ignore_index=True)
    df_results_T.to_csv('/home/jamalids/Downloads/output/df_results_T.csv', index=False)

    df_results_sum_T = pd.concat(results_sum_T, ignore_index=True)
    df_results_sum_T.to_csv('/home/jamalids/Downloads/output/df_results_sum_T.csv', index=False)
    
    df_results_TS = pd.concat(results_TS, ignore_index=True)
    df_results_TS.to_csv('/home/jamalids/Downloads/output/df_results_TS.csv', index=False)

    df_results_sum_TS = pd.concat(results_sum_TS, ignore_index=True)
    df_results_sum_TS.to_csv('/home/jamalids/Downloads/output/df_results_sum_TS.csv', index=False)


    
    df1=df_results_sum
    df2=df_result_overal 
    df3=df_results_sum_T
    df4=df_results_sum_TS
# Add a column to indicate the source DataFrame
    df1['Source'] = 'Compressed  Based on each signal'
    df2['Source'] = 'Compressed  Based on all signal'
    df3['Source'] = 'Compressed  Based on each signal and each timestep'
    df4['Source'] = 'Compressed  Based on  each timestep'
    

# Concatenate the two DataFrames
    
    df_combined = pd.concat([df1[['Dataset Name', 'Compressed Size (bytes)', 'Source','Dataset Type','Compression Ratio']],
                         df2[['Dataset Name', 'Compressed Size (bytes)', 'Source','Dataset Type','Compression Ratio']],
                             df3[['Dataset Name', 'Compressed Size (bytes)', 'Source','Dataset Type','Compression Ratio']],
                                df4[['Dataset Name', 'Compressed Size (bytes)', 'Source','Dataset Type','Compression Ratio']]])
   
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_combined, x='Dataset Name', y='Compressed Size (bytes)', hue='Source')
    plt.title('Comparison of Compressed Sizes Between DF1 and DF2 and DF3')
    plt.xlabel('Dataset Name')
    plt.xticks(rotation=45, fontsize=12, ha='right')  # Horizontal alignment set to 'right'
    plt.ylabel('Compressed Size (bytes)')
    plt.xticks(rotation=45)
    plt.legend(title='Source')
    plt.tight_layout()
    
    
    plt.savefig('/home/jamalids/Downloads/output/compressed_sizes_comparison.png')
    plt.show()
    df_combined['Source and Dataset Type'] = df_combined['Source'] + " - " + df_combined['Dataset Type']

    plt.figure(figsize=(26, 20))  # Adjust figure size as needed

    sns.barplot(data=df_combined, x='Dataset Name', y='Compression Ratio', 
            hue='Source and Dataset Type', dodge=True)

    plt.title('Comparison of Compression Ratios Across Datasets', fontsize=16)
    plt.xlabel('Dataset Name', fontsize=16)
    plt.ylabel('Compression Ratio', fontsize=16)

    # Adjust the rotation and alignment of x-tick labels
    plt.xticks(rotation=45, fontsize=18, ha='right')  # Horizontal alignment set to 'right'

    plt.yticks(fontsize=24)
    plt.legend(title='Source and Dataset Type', bbox_to_anchor=(1.05, 1), loc='upper right', fontsize=18)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('/home/jamalids/Downloads/output/compression_ratios_comparison_combined.png', dpi=300)
    plt.show()




# In[6]:


# Filter the DataFrame for rows where 'Dataset Type' is 'UCR_UEA'
df_filtered = df_combined[df_combined['Dataset Type'] == 'UCR_UEA']

# Assuming df_filtered is your DataFrame prepared for plotting

plt.figure(figsize=(16, 10))  # Adjust figure size as needed

sns.barplot(data=df_filtered, x='Dataset Name', y='Compression Ratio', hue='Source', dodge=True)

plt.title('Comparison of Compression Ratios for UCR/UEA Datasets', fontsize=16)
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Compression Ratio', fontsize=14)

# Adjust the rotation and alignment of x-tick labels
plt.xticks(rotation=45, fontsize=12, ha='right')  # Horizontal alignment set to 'right'

plt.yticks(fontsize=12)
plt.legend(title='Compression Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()

# Save and show the plot as before
plt.savefig('UCR_UEA_compression_ratios_comparison_adjusted.png', dpi=300)
plt.show()


# In[7]:


# Filter the DataFrame for rows where 'Dataset Type' is 'UCR_UEA'
df_filtered = df_combined[df_combined['Dataset Type'] == 'AEON']

# Assuming df_filtered is your DataFrame prepared for plotting

plt.figure(figsize=(16, 10))  # Adjust figure size as needed

sns.barplot(data=df_filtered, x='Dataset Name', y='Compression Ratio', hue='Source', dodge=True)

plt.title('Comparison of Compression Ratios for AEON Datasets', fontsize=16)
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Compression Ratio', fontsize=14)

# Adjust the rotation and alignment of x-tick labels
plt.xticks(rotation=45, fontsize=12, ha='right')  # Horizontal alignment set to 'right'

plt.yticks(fontsize=12)
plt.legend(title='Compression Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()

# Save and show the plot as before
plt.savefig('UCR_UEA_compression_ratios_comparison_adjusted.png', dpi=300)
plt.show()


# In[ ]:




