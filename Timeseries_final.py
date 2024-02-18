
from typing import Optional, Literal
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score
from tslearn.datasets import UCR_UEA_datasets
import zlib
import blosc 
from aeon.datasets.tsc_data_lists import univariate2015, univariate, multivariate, univariate_equal_length, multivariate_equal_length
from aeon.datasets import load_classification
import zstandard as zstd

def pipeline(
        dataset_type: Literal['UCR_UEA', 'AEON'],
        model_type: Literal['Hierarchical', 'KMeans', 'Spectral'] = 'Hierarchical',
        
        batch_size: int = 500,
        p: int = 1,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        transform_type: Optional[Literal['std', 'minmax', 'robust']] = None
) -> pd.DataFrame:
    print(f'Processing dataset: {dataset_name if dataset_name else dataset_path}')

    # Load dataset based on dataset_type or dataset_path
    if dataset_path:
        df = pd.read_csv(dataset_path, header=None)
        ts_list = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values
    elif dataset_type == 'UCR_UEA':
        ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
        ts_list1=ts_list
    elif dataset_type == 'AEON':
        ts_list, y_true = load_classification(dataset_name) 
        ts_list1=ts_list
        ts_list = np.array(ts_list).flatten()
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    # Ensure dataset was loaded
    if ts_list is None or len(ts_list) == 0:
        print(f"Failed to load dataset: {dataset_name if dataset_name else dataset_path}")
        return pd.DataFrame()
    n_clusters = len(set(y_true))
    # calculating sparsity
    non_zeros = np.count_nonzero(ts_list)
    no_elements = np.prod(ts_list.shape)
    sparsity = 1 - (non_zeros / no_elements)

    # measure values smaller than 1e-5
    threshold = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sparsity_values = []
    for t in threshold:
        no_small_values = np.count_nonzero(np.abs(ts_list.flatten()) < t)
        sparsity_values.append(no_small_values / no_elements)
        print('Sparsity with values smaller than {}: {}'.format(t, sparsity_values[-1]))

    # normalize ts_list
    ts_list_norm = (ts_list - np.min(ts_list)) / (np.max(ts_list) - np.min(ts_list))
    print('Normalized Sparsity: {}'.format(np.count_nonzero(ts_list_norm) / no_elements))

    # measure the total number of unique values in ts_list
    unique_values = np.unique(ts_list)
    print('Number of unique values: {} out of {}'.format(len(unique_values), no_elements))

    # the number of most frequent values
    unique, counts = np.unique(ts_list, return_counts=True)
    most_frequent = unique[np.argmax(counts)]
    print('Most frequent value: {} with {} occurrences'.format(most_frequent, np.max(counts)))

    # plot the frequency of values
    plt.hist(ts_list.flatten(), bins=100)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency of Values')
    plt.savefig(os.path.join('plot', dataset_name + dataset_type + '_frequency_bins.png'))
    plt.close()

    # max and mean and min
    max_value = np.max(ts_list)
    mean_value = np.mean(ts_list)
    min_value = np.min(ts_list)
    print('Max: {}, Mean: {}, Min: {}'.format(max_value, mean_value, min_value))

    
    element_type_description = "Unknown" 

   # Check the structure and adjust the message accordingly
    if hasattr(ts_list1, 'ndim'):  
        if ts_list1.ndim == 3:
            element_type_description = f"3D array with element type {type(ts_list1[0][0][0])}"
        elif ts_list1.ndim == 2:
            element_type_description = f"2D array with element type {type(ts_list1[0][0])}"
        elif ts_list1.ndim == 1:
            element_type_description = f"1D array with element type {type(ts_list1[0])}"
    else:
          # Handle the case where ts_list1 does not have 'ndim'
        if isinstance(ts_list1, list) and len(ts_list1) > 0:
            if isinstance(ts_list1[0], list):
                # Assuming a 2D list structure
                element_type_description = f"2D list with element type {type(ts_list1[0][0])}"
            else:
               # Assuming a 1D list
               element_type_description = f"1D list with element type {type(ts_list1[0])}"

        print(f"Type of the dataset: {type(ts_list1)}, Structure and element type: {element_type_description}")


    # compress tensor using zlib
    compressed_zlib = zlib.compress(pickle.dumps(ts_list1))
    original_size_zlib = ts_list1.size * ts_list1.itemsize
    print('Zlib Orig size {}, Compressed size: {}'.format(original_size_zlib, len(compressed_zlib)))
    uncomp_zlib = pickle.loads(zlib.decompress(compressed_zlib))
    print('Zlib Uncompressed size: {}'.format(uncomp_zlib.size * uncomp_zlib.itemsize))  

    # compress tensor using Blosc
    compressed_blosc = blosc.compress(pickle.dumps(ts_list1))
    original_size_blosc = ts_list1.size * ts_list1.itemsize
    print('Blosc Orig size {}, Compressed size: {}'.format(original_size_blosc, len(compressed_blosc)))
    uncomp_blosc = pickle.loads(blosc.decompress(compressed_blosc))
    print('Blosc Uncompressed size: {}'.format(uncomp_blosc.size * uncomp_blosc.itemsize))  
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(pickle.dumps(ts_list))
    print('zstd Compressed size: {}'.format(len(compressed)))
    

    result_dict = {
        'Dataset': [dataset_name],
        'Repo':[dataset_type],
        'Shape': [ts_list1.shape],
        'Structure and element type':[element_type_description],
        'Num of clusters': [n_clusters],
        'Sparsity': [1 - np.count_nonzero(ts_list) / ts_list.size],
        'Sparsity with values < 1e-5': [sparsity_values[0]],
        'Sparsity with values < 1e-4': [sparsity_values[1]],
        'Sparsity with values < 1e-3': [sparsity_values[2]],
        'Sparsity with values < 1e-2': [sparsity_values[3]],
        'Sparsity with values < 1e-1': [sparsity_values[4]],
        'Max': [np.max(ts_list)],
        'Mean': [np.mean(ts_list)],
        'Min': [np.min(ts_list)],
        'Number of unique values': [len(unique_values)],
        'Most frequent value': [most_frequent],
        'Zlib Original size': [original_size_zlib],
        'Zlib Compressed size': [len(compressed_zlib)],
        'Zlib Uncompressed size': [uncomp_zlib.size * uncomp_zlib.itemsize],
        'Blosc Original size': [original_size_blosc],
        'Blosc Compressed size': [len(compressed_blosc)],
        'Blosc Uncompressed size': [uncomp_blosc.size * uncomp_blosc.itemsize],
        'zstd Compressed size':[len(compressed)],
    }


    result_df = pd.DataFrame(result_dict)
    return result_df
    
    
def main():
    
    UCR_UEA_datasets_list =  UCR_UEA_datasets().list_multivariate_datasets()
    UCR_UEA_datasets_list.remove('InsectWingbeat')
    AEON_datasets_list = list(multivariate)
    AEON_datasets_list.remove('InsectWingbeat')
    AEON_datasets_list.remove('CharacterTrajectories')
    AEON_datasets_list.remove('SpokenArabicDigits')
    AEON_datasets_list.remove('JapaneseVowels')
    
    

    # Prepare datasets info with types
    datasets = [{'name': name, 'type': 'UCR_UEA'} for name in UCR_UEA_datasets_list] +                [{'name': name, 'type': 'AEON'} for name in AEON_datasets_list]


    results = []
    for dataset in datasets:
        print(dataset)
        result = pipeline(
            dataset_name=dataset['name'],
            dataset_type=dataset['type'],
            transform_type='minmax',
            model_type='Hierarchical',
            
            batch_size=500,
            p=4
        )
        results.append(result)

    # Combine results and save to CSV
    if results:
        all_results = pd.concat(results, ignore_index=True)
        output_path = './output/results_combined.csv'
        if not os.path.exists('./output'):
            os.makedirs('./output')
        all_results.to_csv(output_path, index=False)
        print(f'Results saved to {output_path}')
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()

