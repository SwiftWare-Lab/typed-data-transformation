from typing import List, Literal, Optional
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score

from t2f.data.dataset import read_ucr_datasets
from t2f.extraction.extractor import feature_extraction
from t2f.selection.selection import feature_selection
from t2f.model.clustering import ClusterWrapper


def pipeline(
        files: List[str],
        transform_type: Optional[Literal['std', 'minmax', 'robust']],
        model_type: Literal['Hierarchical', 'KMeans', 'Spectral'],
        train_size: float = 0,
        batch_size: int = 500,
        p: int = 1
) -> None:
    # Simple consistency check
    if [x for x in files if not os.path.isfile(x)]:
        raise ValueError('At least time-series path don\'t exist')
    if train_size < 0 or train_size > 1:
        raise ValueError('Train size must be between 0 and 1')

    print('Read ucr datasets: ', files)
    ts_list, y_true = read_ucr_datasets(paths=files)
    n_clusters = len(set(y_true))  # Get number of clusters to find

    print('Dataset shape: {}, Num of clusters: {}'.format(ts_list.shape, n_clusters))
    # extract name of the dataset
    dataset_name = files[0].split('/')[-1].split('_')[0]
    plt_directory = "plots"

    # extract statistics of ts_list
    # measure sparsity of ts_list
    non_zeros = np.count_nonzero(ts_list)
    no_elements = ts_list.size
    sparsity = 1 - (non_zeros / no_elements)
    print('Sparsity: {}'.format(sparsity))

    # measure values smaller than 1e-5
    threshold = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for t in threshold:
        no_small_values = np.count_nonzero(np.abs(ts_list) < t)
        print('sparsity with smaller than {}: {}'.format(t, no_small_values / no_elements))

    # normalise ts_list
    ts_list_norm = (ts_list - np.min(ts_list)) / (np.max(ts_list) - np.min(ts_list))
    print('Normalised Sparsity: {}'.format(np.count_nonzero(ts_list_norm) / no_elements))
    # TODO: any other normalizazion methods?

    # measure the total number of unique values in ts_list
    unique_values = np.unique(ts_list)
    print('Number of unique values: {} out of {}'.format(len(unique_values), no_elements))

    # the number of most frequent values
    unique, counts = np.unique(ts_list, return_counts=True)
    most_frequent = unique[np.argmax(counts)]
    print('Most frequent value: {} with {} occurences'.format(most_frequent, np.max(counts)))

    # plot the frequency of values
    import matplotlib.pyplot as plt
    plt.hist(ts_list.flatten(), bins=100)
    plt.show()
    # sacve the plot with dataset name
    # make the plots path
    if not os.path.exists(plt_directory):
        os.makedirs(plt_directory)
    plt.savefig(os.path.join(plt_directory, dataset_name+'frequencybins.png'))
    plt.close()

    # max and mean and min
    max_value = np.max(ts_list)
    mean_value = np.mean(ts_list)
    min_value = np.min(ts_list)
    print('Max: {}, Mean: {}, Min: {}'.format(max_value, mean_value, min_value))

    # print the type of the dataset
    print('Type of the dataset: {} and value type {}'.format(type(ts_list), type(ts_list[0][0][0])))


    # compress tensor using zlib
    import zlib
    compressed = zlib.compress(pickle.dumps(ts_list))
    original_size = ts_list.size * ts_list.itemsize
    print('Orig size {}, Compressed size: {}'.format(original_size, len(compressed)))
    uncomp = pickle.loads(zlib.decompress(compressed))
    print('Uncompressed size: {}'.format(uncomp.size * uncomp.itemsize)) # sanity check
    # TODO find some libraries like this



if __name__ == '__main__':
    pipeline(
        files=['data/BasicMotions/BasicMotions_TRAIN.txt', 'data/BasicMotions/BasicMotions_TEST.txt'],
        transform_type='minmax',
        model_type='Hierarchical',
        train_size=0.3,
        batch_size=500,
        p=4,
    )
    print('Hello World!')
