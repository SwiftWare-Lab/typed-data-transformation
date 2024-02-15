from typing import List, Literal, Optional
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

def pipeline(
        dataset_name: str,
        transform_type: Optional[Literal['std', 'minmax', 'robust']],
        model_type: Literal['Hierarchical', 'KMeans', 'Spectral'],
        train_size: float = 0,
        batch_size: int = 500,
        p: int = 1
) -> pd.DataFrame:
    # Simple consistency check
    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    print('Read ucr dataset: ', dataset_name)
    ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)

    if y_true is None:
        print(f"Failed to load dataset: {dataset_name}")
        return pd.DataFrame()

    n_clusters = len(set(y_true))  

    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))
       

    # extract statistics of ts_list
    # measure sparsity of ts_list
    non_zeros = np.count_nonzero(ts_list)
    no_elements = ts_list.size
    sparsity = 1 - (non_zeros / no_elements)
    print('Sparsity: {}'.format(sparsity))

    # measure values smaller than 1e-5
    threshold = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sparsity_values = []
    for t in threshold:
        no_small_values = np.count_nonzero(np.abs(ts_list) < t)
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
    plt.savefig(os.path.join('plot', dataset_name + '_frequency_bins.png'))
    plt.close()

    # max and mean and min
    max_value = np.max(ts_list)
    mean_value = np.mean(ts_list)
    min_value = np.min(ts_list)
    print('Max: {}, Mean: {}, Min: {}'.format(max_value, mean_value, min_value))

    # print the type of the dataset
    print('Type of the dataset: {} and value type {}'.format(type(ts_list), type(ts_list[0][0][0])))

    # compress tensor using zlib
    compressed_zlib = zlib.compress(pickle.dumps(ts_list))
    original_size_zlib = ts_list.size * ts_list.itemsize
    print('Zlib Orig size {}, Compressed size: {}'.format(original_size_zlib, len(compressed_zlib)))
    uncomp_zlib = pickle.loads(zlib.decompress(compressed_zlib))
    print('Zlib Uncompressed size: {}'.format(uncomp_zlib.size * uncomp_zlib.itemsize))  

    # compress tensor using Blosc
    compressed_blosc = blosc.compress(pickle.dumps(ts_list))
    original_size_blosc = ts_list.size * ts_list.itemsize
    print('Blosc Orig size {}, Compressed size: {}'.format(original_size_blosc, len(compressed_blosc)))
    uncomp_blosc = pickle.loads(blosc.decompress(compressed_blosc))
    print('Blosc Uncompressed size: {}'.format(uncomp_blosc.size * uncomp_blosc.itemsize))  

    result_dict = {
        'Dataset': [dataset_name],
        'Shape': [ts_list.shape],
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
    }


    result_df = pd.DataFrame(result_dict)
    return result_df

if __name__ == '__main__':
    datasets = [
        'Libras',
        'AtrialFibrillation',
        'BasicMotions',
        'RacketSports',
        'ERing',
        'Epilepsy',
        'PenDigits',
        'StandWalkJump',
        'UWaveGestureLibrary',
        'Handwriting',
        'ArticularyWordRecognition',
        'HandMovementDirection',
        'LSST',
        'Cricket',
        'EthanolConcentration',
        'SelfRegulationSCP1',
        'SelfRegulationSCP2',
        'PhonemeSpectra'
    ]
    if not os.path.exists('./plot'):
               os.makedirs('./plot')
               
    if not os.path.exists('./output'):
               os.makedirs('./output')           
    results = []
    for dataset in datasets:
        result = pipeline(
            dataset_name=dataset,
            transform_type='minmax',
            model_type='Hierarchical',
            train_size=0.3,
            batch_size=500,
            p=4,
        )
        results.append(result)
    
    all_results = pd.concat(results, ignore_index=True)
    # Save DataFrame to CSV
    filename = "results.csv"
    csv_save_path = os.path.join('./output', filename)
    all_results.to_csv(csv_save_path, index=False)






