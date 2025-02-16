
import numpy as np
from pandas import value_counts
from scipy.stats import entropy
import math
from collections import Counter
import sys
import pandas as pd



def calculate_kth_order_entropy(sequence, k):
    """
    NOTE: not sure if this woeks correctly, GPY generated
    Calculate the kth-order Shannon entropy of a sequence.
    Parameters:
        sequence (list or array-like): The input sequence of symbols.
        k (int): The order of entropy (context length).

    Returns:
        float: The computed kth-order entropy.
    """
    n = len(sequence)

    if k >= n:
        raise ValueError("k must be less than the length of the sequence.")

    # Extract k-grams and (k+1)-grams
    k_grams = [tuple(sequence[i:i + k]) for i in range(n - k)]
    k_plus_1_grams = [tuple(sequence[i:i + k + 1]) for i in range(n - k)]

    # Compute frequency counts
    k_gram_counts = Counter(k_grams)
    k_plus_1_gram_counts = Counter(k_plus_1_grams)

    # Calculate probabilities
    total_k_grams = sum(k_gram_counts.values())
    total_k_plus_1_grams = sum(k_plus_1_gram_counts.values())

    # Normalize counts to probabilities
    k_gram_probs = {gram: count / total_k_grams for gram, count in k_gram_counts.items()}
    k_plus_1_gram_probs = {gram: count / total_k_plus_1_grams for gram, count in k_plus_1_gram_counts.items()}

    # Compute kth-order entropy
    kth_order_entropy = 0.0
    for k_plus_1_gram, p_k_plus_1 in k_plus_1_gram_probs.items():
        k_gram = k_plus_1_gram[:-1]  # Extract the context (k-gram)
        p_k = k_gram_probs[k_gram]  # Probability of the context (k-gram)
        p_conditional = p_k_plus_1 / p_k  # Compute P(s | s1, ..., sk)
        kth_order_entropy -= p_k_plus_1 * np.log2(p_conditional)
    return kth_order_entropy






def compute_probablity(data):
    """
    Computes the probability of each element in a 1D array of floating-point data.

    Parameters:
        data (array-like): Input array of real numbers.

    Returns:
        array: The computed probabilities.
    """
    # Ensure the data is a numpy array
    data = np.asarray(data)

    # num of bins is equal to number of unique elements in the data
    uniq_vals = np.unique(data)
    num_bins = len(uniq_vals)
    value_counts = np.zeros(num_bins)
    for i, val in enumerate(uniq_vals):
        value_counts[i] = np.sum(data == val)

    # count the number of occurrences of each unique element
    #value_counts = np.bincount(data)

    assert np.sum(value_counts) == len(data)

    return value_counts, value_counts / len(data)


def compute_entropy(data):
    """
    Computes the entropy of a 1D array of floating-point data.

    Parameters:
        data (array-like): Input array of real numbers.

    Returns:
        float: The computed entropy.
    """
    # Ensure the data is a numpy array
    data = np.asarray(data)

    value_counts, probabilities = compute_probablity(data)
    #print(value_counts)

    assert np.abs(np.sum(probabilities) - 1 ) < 1e-8
    # Compute Shannon entropy
    entropy_manual = -np.sum(probabilities * np.log2(probabilities))
    entropy_scipy = entropy(np.array(probabilities), base=2)

    return entropy_manual, entropy_scipy


def calculate_kth_order_entropy_paper(sequence, k, modified=True):
    """
    NOTE. from this paper: https://people.unipmn.it/manzini/papers/bwjacm2.pdf
    Calculate the kth-order Shannon entropy of a sequence.
    Parameters:
        sequence (list or array-like): The input sequence of symbols.
        k (int): The order of entropy (context length).

    Returns:
        float: The computed kth-order entropy.
    """
    n = len(sequence)

    if k >= n:
        raise ValueError("k must be less than the length of the sequence.")
    k_grams = [tuple(sequence[i:i + k]) for i in range(n - k)]
    # unique k-grams
    unique_k_grams = set(k_grams)
    # compute the location of occurence of k-gram
    k_th_order = 0.0
    for k_gram in unique_k_grams:
        # for every sequence of k symbols
        #k_gram = tuple(sequence[i:i + k])
        # find the location of the k-gram
        loc = [j for j in range(n - k) if tuple(sequence[j:j + k]) == k_gram]
        # all values in loc + k + 1 are the next symbols
        next_symbols = [sequence[j + k] for j in loc]
        # compute the entropy of the next symbol
        entropy_i, entropy_i_sci = compute_entropy(next_symbols)
        len_i = len(next_symbols)
        if entropy_i == 0 and modified:
            entropy_i_sci= (1 + np.log2(n))/n
        k_th_order += len_i * entropy_i_sci
    # Normalize the entropy
    k_th_order = 1/n * k_th_order
    return k_th_order


def compute_kth_order_entropy_optimized(sequence, k, modified=False):
    """
    Optimized computation of k-th order entropy.
    NOTE. from this paper: https://people.unipmn.it/manzini/papers/bwjacm2.pdf
    Args:
        sequence (list): The input sequence.
        k (int): The k value for k-grams.
        modified (bool): Whether to apply the modified entropy adjustment.

    Returns:
        float: The normalized k-th order entropy.
    """
    n = len(sequence)

    if k >= n:
        raise ValueError("k must be less than the length of the sequence.")

    # Step 1: Precompute all k-grams and their locations
    k_gram_map = {}  # Dictionary to store locations of each k-gram
    for i in range(n - k):
        k_gram = tuple(sequence[i:i + k])
        if k_gram not in k_gram_map:
            k_gram_map[k_gram] = []  # Initialize with an empty list
        k_gram_map[k_gram].append(i)

    # Step 2: Compute the entropy for each k-gram's next symbols
    k_th_order = 0.0
    for k_gram, locations in k_gram_map.items():
        # Gather next symbols based on locations
        next_symbols = [sequence[j + k] for j in locations]

        # Compute the entropy of these next symbols
        entropy_i, entropy_i_sci = compute_entropy(next_symbols)
        len_i = len(next_symbols)

        # Apply modified entropy adjustment if needed
        if entropy_i == 0 and modified:
            entropy_i_sci = (1 + np.log2(n)) / n

        # Contribution of this k-gram's next symbol entropy to total entropy
        k_th_order += len_i * entropy_i_sci

    # Normalize the computed entropy
    k_th_order /= n
    return k_th_order




# compute entropy for an array that is casted to w-width type
def compute_entropy_w(data, w):
    """
    Computes the entropy of a 1D array of floating-point data casted to a fixed-width integer type.

    Parameters:
        data (array-like): Input array of real numbers.
        w (int): The fixed-width integer type to cast the data.

    Returns:
        float: The computed entropy.
    """
    # Ensure the data is a numpy array
    data = np.asarray(data)

    # Cast the data to the fixed-width integer type
    if w == 8:
        data_casted = data.view(dtype=np.uint8)
    elif w == 16:
        data_casted = data.view(dtype=np.uint16)
    elif w == 32:
        data_casted = data.view(dtype=np.uint32)
    elif w == 64:
        data_casted = data.view(dtype=np.uint64)
    else:
        raise ValueError("Invalid fixed-width integer type.")


    # make sure casting is lossless
    assert np.allclose(data, data_casted.view(dtype=data.dtype))

    # Compute the entropy of the casted data
    manual_entropy, entropy_scipy = compute_entropy(data_casted)

    return manual_entropy, entropy_scipy


def tdt_transform(data):
    """
    Transpose-Downsample-Transpose transform for a 1D array of floating-point data.

    Parameters:
        data (array-like): Input array of real numbers casted as uint8 (bytes).

    Returns:
        array: The transformed data.
    """
    # Ensure the data is a numpy array
    data = np.asarray(data)

    # Transpose the data
    #cast_data = data.view(dtype=np.uint8)
    bytes0 = data[0::4]
    bytes1 = data[1::4]
    bytes2 = data[2::4]
    bytes3 = data[3::4]

    data_transformed = np.zeros(len(data), dtype=np.uint8)
    p_len = len(data) // 4
    data_transformed[0:p_len:1] = bytes0
    data_transformed[p_len:2*p_len:1] = bytes1
    data_transformed[2*p_len:3*p_len:1] = bytes2
    data_transformed[3*p_len:4*p_len:1] = bytes3
    return data_transformed


# calc_test_k = compute_kth_order_entropy_optimized("mississippi", 1, False)
# print(f"Test k-th order entropy: {calc_test_k}")

chunk_size = -1
contig_order = False
# generate random floating point data aray and measure entropy for different fixed-width integer types
data_size = 50000 #65000
data = np.random.rand(data_size)
dataset_name = "random"
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    # open tsv file with df
    data = pd.read_csv(dataset_path, sep='\t')
if len(sys.argv) > 2:
    chunk_size = int(sys.argv[3])
if chunk_size == -1:
    chunk_size = 65000


# convert it to float32 for testing # TODO remove this once transformation supports float64
data = np.float32(data)
bit_width = data.dtype.itemsize * 8
data_int = data.view(dtype=np.uint8)

num_blocs = len(data_int) // chunk_size

num_entropies = 5
kth_order_ent = {'0': [], '1':[], '2':[], '3':[], '4':[]}
orig_entropy = {'8':[], '16': [], '32': [], '64': []}

tdt_kth_order_ent = {'0': [], '1':[], '2':[], '3':[], '4':[]}
tdt_orig_entropy = {'8':[], '16': [], '32': [], '64': []}

for i in range(num_blocs):
    data_block = data_int[i*chunk_size:(i+1)*chunk_size]
    for k in range(num_entropies):
        kth_order_ent[str(k)].append(compute_kth_order_entropy_optimized(data_block, k))
        print(f"{k}-th order entropy for block {i}: {kth_order_ent[str(k)][-1]}")

    for w in [8, 16, 32, 64]:
        manual_entropy, entropy_scipy = compute_entropy_w(data, w)
        orig_entropy[str(w)].append(manual_entropy)
        print(f"Entropy for {w}-bit fixed-width integers: {manual_entropy} (SciPy: {entropy_scipy})")

    tdt_data = tdt_transform(data_block) # this is done assuming float32, TODO fix this
    tdt_data_int = tdt_data.view(dtype=np.uint8)
    for w in [8, 16, 32, 64]:
        manual_entropy, entropy_scipy = compute_entropy_w(tdt_data, w)
        tdt_orig_entropy[str(w)].append(manual_entropy)
        print(f"Entropy for TDT-transformed {w}-bit fixed-width integers: {manual_entropy} (SciPy: {entropy_scipy})")

    for k in range(num_entropies):
        kth_order_ent[str(k)].append(compute_kth_order_entropy_optimized(tdt_data_int, k))
        print(f"{k}-th order entropy for TDT-transformed block {i}: {kth_order_ent[str(k)][-1]}")




