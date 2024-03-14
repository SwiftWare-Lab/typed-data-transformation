
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

def detect_data_type(series):
    if pd.api.types.is_integer_dtype(series):
        unique_ratio = series.nunique() / len(series)
        return 'Categorical' if unique_ratio < 0.05 else 'Integer'
    elif pd.api.types.is_float_dtype(series):
        return 'Float'
    else:
        return 'Other'




def compress_float_fpzip(input_data, precision=32):
    
    if not isinstance(input_data, np.ndarray) or not np.issubdtype(input_data.dtype, np.floating):
        raise TypeError("FPZIP compression requires input data to be a numpy floating-point array.")
    
    compressed_data = fpzip.compress(input_data, precision=precision)
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
        
        else:
            raise ValueError(f"Compression method '{method}' not recognized.")

    compressed_size = len(compressed_data)
    original_size = data.nbytes if isinstance(data, np.ndarray) else len(data)
    compression_ratio = original_size / compressed_size if compressed_size else 1
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
        if compression_method == 'fpzip':
    
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

def select_compression_method(data_type, variance, entropy):
    
    if data_type == 'Integer':
        if variance < 0.2:
            return 'delta+gzip'
        else:
            return 'gzip'
    elif data_type == 'Float':
        if variance > 0.5 or entropy > 0.5:
            return 'fpzip'
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
    compression_method='fpzip'
    if compression_method == 'fpzip':
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
    
    return compression_results_df,compression_results_Summary,compression_results_overall
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
    
    for dataset in datasets:
        print(f"Processing {dataset}")
        pipeline_result = pipeline(
            dataset_name=dataset['name'],
            dataset_type=dataset['type']
        )
        
        if pipeline_result is not None:
            result, results_sum_current, result_overal_current = pipeline_result
            results.append(result)
            results_sum.append(results_sum_current)
            result_overal.append(result_overal_current)
        else:
            print(f"Skipping dataset {dataset} due to errors in processing.")
    
    return results, results_sum, result_overal

if __name__ == '__main__':
    results, results_sum, result_overal = main()
    
    
    df_results = pd.concat(results, ignore_index=True)
    df_results.to_csv('/home/jamalids/Downloads/output/df_results.csv', index=False)


    df_results_sum = pd.concat(results_sum, ignore_index=True)
    df_results_sum.to_csv('/home/jamalids/Downloads/output/df_results_sum.csv', index=False)

    df_result_overal = pd.concat(result_overal, ignore_index=True)
    df_result_overal.to_csv('/home/jamalids/Downloads/output/df_result_overal.csv', index=False)
    df1=df_results_sum
    df2=df_result_overal 
# Add a column to indicate the source DataFrame
    df1['Source'] = 'Compressed size Based on each signal'
    df2['Source'] = 'Compressed size Based on all signal'

# Concatenate the two DataFrames
    
    df_combined = pd.concat([df1[['Dataset Name', 'Compressed Size (bytes)', 'Source']],
                         df2[['Dataset Name', 'Compressed Size (bytes)', 'Source']]])
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_combined, x='Dataset Name', y='Compressed Size (bytes)', hue='Source')
    plt.title('Comparison of Compressed Sizes Between DF1 and DF2')
    plt.xlabel('Dataset Name')
    plt.xticks(rotation=45, fontsize=12, ha='right')  # Horizontal alignment set to 'right'
    plt.ylabel('Compressed Size (bytes)')
    plt.xticks(rotation=45)
    plt.legend(title='Source')
    plt.tight_layout()
    
    
    plt.savefig('/home/jamalids/Downloads/output/compressed_sizes_comparison.png')
    plt.show()



# In[ ]:


############### COMPARE 

def analyze_dataset(data):
    if data is None:
        return {'data_type': 'Dataset could not be loaded'}
    
    analysis = {}
    if isinstance(data, np.ndarray):
        analysis['shape'] = data.shape
        analysis['data_type'] = data.dtype.name
        if np.issubdtype(data.dtype, np.number):
            analysis['mean'] = np.nanmean(data)
            analysis['std_dev'] = np.nanstd(data)
            analysis['min'] = np.nanmin(data)
            analysis['max'] = np.nanmax(data)
            analysis['percent_missing'] = np.mean(np.isnan(data)) * 100
    else:
        analysis['data_type'] = 'non-numeric or non-array data'
    
    return analysis

# Given two datasets, this function compares their analyses
def compare_dataset_analyses(analysis1, analysis2):
    differences = {}
    for key in analysis1:
        if analysis1[key] != analysis2[key]:
            differences[key] = (analysis1[key], analysis2[key])
    return differences
#dataset1 = np.random.rand(100, 10)  # Mock data for UCR/UEA dataset
#dataset2 = np.random.rand(100, 10) * 100  # Mock data for AEON dataset scaled differently

dataset1, y_true, _, _ = UCR_UEA_datasets().load_dataset('FaceDetection')

dataset2, y_true = load_classification('FaceDetection')
dataset2 = np.transpose(dataset2, (0, 2, 1))
analysis1 = analyze_dataset(dataset1)
analysis2 = analyze_dataset(dataset2)

# Compare the analyses
differences = compare_dataset_analyses(analysis1, analysis2)
differences


# In[ ]:


#PLOT 
df_combined = pd.concat([df1, df2])

df_combined['Source and Dataset Type'] = df_combined['Source'] + " - " + df_combined['Dataset Type']

plt.figure(figsize=(16, 10))  #

sns.barplot(data=df_combined, x='Dataset Name', y='Compression Ratio', 
            hue='Source and Dataset Type', dodge=True)

plt.title('Comparison of Compression Ratios Across Datasets', fontsize=16)
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Compression Ratio', fontsize=14)

plt.xticks(rotation=45, fontsize=12, ha='right')  

plt.yticks(fontsize=12)
plt.legend(title='Source and Dataset Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()

# Save and show the plot
plt.savefig('compression_ratios_comparison_combined.png', dpi=300)
plt.show()


# In[ ]:



df_combined = pd.concat([df1, df2])

# Filter the DataFrame for rows where 'Dataset Type'
df_filtered = df_combined[df_combined['Dataset Type'] == 'AEON']


plt.figure(figsize=(16, 10))  

sns.barplot(data=df_filtered, x='Dataset Name', y='Compression Ratio', hue='Source', dodge=True)

plt.title('Comparison of Compression Ratios forAEON Datasets', fontsize=16)
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Compression Ratio', fontsize=14)

plt.xticks(rotation=45, fontsize=12, ha='right')  

plt.yticks(fontsize=12)
plt.legend(title='Compression Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()


plt.savefig('AEON_compression_ratios_comparison_adjusted.png', dpi=300)
plt.show()

