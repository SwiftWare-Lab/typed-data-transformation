
from typing import Optional, Literal
import os
import pickle
import numpy as np
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets
import zlib
import blosc
from aeon.datasets.tsc_data_lists import univariate, multivariate
from aeon.datasets import load_classification
import zstandard as zstd


def run_length_encode(data, MAX_POINT_NUMBER=2):
    
    # Round data to the specified precision 
    if np.issubdtype(data.dtype, np.floating):
        data = np.round(data, MAX_POINT_NUMBER)
    
    # Flatten the data 
    if data.ndim != 1:
        data = data.flatten()
    
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.append((data[i - 1], count))
            count = 1
    encoded_data.append((data[-1], count))
    
    return encoded_data



def delta_encode_multivariate(ts_tensor):
    if ts_tensor.ndim == 3:  
        
        avg_initial = np.mean(ts_tensor[:, 0, :], axis=0, keepdims=True)
        
        deltas = np.diff(ts_tensor, axis=1, prepend=np.expand_dims(avg_initial, axis=1))
    elif ts_tensor.ndim == 2:  
        
        avg_initial = np.mean(ts_tensor[0, :], keepdims=True)
        # Use the average value for delta encoding
        deltas = np.diff(ts_tensor, axis=0, prepend=np.expand_dims(avg_initial, axis=0))
    else:
        raise ValueError("Unsupported tensor shape")
    
    
    original_size = ts_tensor.nbytes  
    compressed_size = deltas.nbytes 
    
    # Calculate the compression ratio
    compression_ratio = original_size / max(compressed_size, 1)  # Avoid division by zero
    
    return deltas, compression_ratio


def xor_encode(data):
    if data.ndim != 1:
        data = data.flatten()
    encoded_data = [data[0]]
    for i in range(1, len(data)):
        xor_result = np.bitwise_xor(np.float32(data[i]).view(np.int32), np.float32(data[i-1]).view(np.int32))
        encoded_data.append(xor_result)
    return np.array(encoded_data)



def split_dataset_by_dtype(ts_list):
   
    
    int_indices = []
    float_indices = []
    
    # Assuming the last dimension represents features
    for feature_idx in range(ts_list.shape[2]):
        # Check dtype of the first sample and time step for this feature
        dtype = ts_list[0, 0,feature_idx].dtype
        
        if np.issubdtype(dtype, np.integer):
            int_indices.append(feature_idx)
        elif np.issubdtype(dtype, np.floating):
            float_indices.append(feature_idx)
    
    # Use the collected indices to split the dataset
    int_data = ts_list[:, :, int_indices] if int_indices else np.array([]).reshape(ts_list.shape[0], ts_list.shape[1], 0)
    float_data = ts_list[:, :, float_indices] if float_indices else np.array([]).reshape(ts_list.shape[0], ts_list.shape[1], 0)
    
    return int_data, float_data



def featurewise_compression(ts_list, dataset_name):
  
    int_data, float_data = split_dataset_by_dtype(ts_list)
    
    
    df1 = pd.DataFrame()
    results_list = []

    if int_data.size > 0:
        
        X_train = int_data
        num_samples, channels, sequence_length = X_train.shape
        
        original_size_bytes = len(pickle.dumps(X_train))

        # Apply Delta Encoding
        delta_encoded_data, delta_ratio = delta_encode_multivariate(X_train)

        # Calculate size after Delta Encoding for comparison
        delta_encoded_size_bytes = len(pickle.dumps(delta_encoded_data))

        # RLE Compression
        compressed_data_RLE = run_length_encode(delta_encoded_data.flatten())
        compressed_size_RLE_bytes = len(pickle.dumps(compressed_data_RLE))
        rle_ratio = original_size_bytes / max(compressed_size_RLE_bytes, 1)

        # zlib Compression
        compressed_data_zlib = zlib.compress(pickle.dumps(delta_encoded_data))
        compressed_size_zlib_bytes = len(compressed_data_zlib)
        zlib_ratio = original_size_bytes / max(compressed_size_zlib_bytes, 1)

        # blosc Compression
        compressed_data_blosc = blosc.compress(pickle.dumps(delta_encoded_data), typesize=8)
        compressed_size_blosc_bytes = len(compressed_data_blosc)
        blosc_ratio = original_size_bytes / max(compressed_size_blosc_bytes, 1)

        # zstandard Compression
        cctx = zstd.ZstdCompressor()
        compressed_data_zstd = cctx.compress(pickle.dumps(delta_encoded_data))
        compressed_size_zstd_bytes = len(compressed_data_zstd)
        zstd_ratio = original_size_bytes / max(compressed_size_zstd_bytes, 1)
        
        
        results_list.append({
            'Dataset': dataset_name,
            'Data Type': 'Integer',
            
            'Original Size (bytes)': original_size_bytes,
            'Compressed Size (bytes)Delta+RLE': compressed_size_RLE_bytes,
            'Compressed Size (bytes)Delta+zlib': compressed_size_zlib_bytes,
            'Compression Ratio': rle_ratio,
            'Compression Ratio1': zlib_ratio
        })

    if float_data.size > 0:
        # Process floating-point data
        original_size_bytes = len(pickle.dumps(float_data))
        compressed_data_XOR = xor_encode(float_data.flatten())
        compressed_size_XOR_bytes = len(pickle.dumps(compressed_data_XOR))
        xor_ratio = original_size_bytes / max(compressed_size_XOR_bytes, 1)

        # RLE Compression
        compressed_data_RLE = run_length_encode(compressed_data_XOR )
        compressed_size_RLE_bytes = len(pickle.dumps(compressed_data_RLE))
        rle_ratio = original_size_bytes / max(compressed_size_RLE_bytes, 1)

        # zlib Compression for XOR encoded data
        compressed_data_zlib1 = zlib.compress(pickle.dumps(float_data))
        compressed_size_zlib_bytes1 = len(compressed_data_zlib1)
        zlib_ratio1 = original_size_bytes / max(compressed_size_zlib_bytes1, 1)


        
        results_list.append({
            'Dataset': dataset_name,
            'Data Type': 'Floating-point',
            
            'Original Size (bytes)': original_size_bytes,
            'Compressed Size (bytes)XOR + RLE':compressed_size_RLE_bytes,
            'Compressed Size (bytes)ZLib': compressed_size_zlib_bytes1,
            'Compression Ratio RLE': rle_ratio,
            'Compression Ratio1 Zlib': zlib_ratio1
        })


    return pd.DataFrame(results_list)



def pipeline(dataset_type: Literal['UCR_UEA', 'AEON'], dataset_name: Optional[str] = None):
    print(f'Processing dataset: {dataset_name}')
    
    # Load dataset
    if dataset_type == 'UCR_UEA':
        ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
    elif dataset_type == 'AEON':
        ts_list, y_true = load_classification(dataset_name)
    else:
        print(f"Failed to load dataset {dataset_name}. Invalid dataset type: {dataset_type}")
        return
    
   
    results_df = featurewise_compression(ts_list,dataset_name)
    output_path = './output/result3_combined.csv'
    if not os.path.exists('./output'):
          os.makedirs('./output')
    results_df.to_csv(output_path, index=False)
    return results_df


def main():
    datasets = [{'name': 'MotorImagery', 'type': 'AEON'}]
    all_results = []  
    
    for dataset in datasets:
        results_df = pipeline(dataset_type=dataset['type'], dataset_name=dataset['name'])
        all_results.append(results_df)  
    
    return all_results  

if __name__ == '__main__':
    all_results = main()  
   
    for results_df in all_results:
        print(results_df)






