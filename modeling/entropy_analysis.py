
import numpy as np
from pandas import value_counts
from scipy.stats import entropy
import math
from collections import Counter
import sys
import pandas as pd
import matplotlib.pyplot as plt



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


def compute_kth_order_entropy_optimized(sequence, k, modified=True):
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
    # TODO: make it work for float64 or others
    # Ensure the data is a numpy array
    data = np.asarray(data)

    # Transpose the data
    #cast_data = data.view(dtype=np.uint8)
    bytes0 = data[0::4]
    bytes1 = data[1::4]
    bytes2 = data[2::4]
    bytes3 = data[3::4]

    entropy_info = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
    for k_level in range(6):
        for bytes in [bytes0, bytes1, bytes2, bytes3]:
            entropy = calculate_kth_order_entropy_paper(bytes, k_level)
            entropy_info[str(k_level)].append(entropy)
            #entropy_per_byte.append(compute_entropy(bytes)[0])

    data_transformed = np.zeros(len(data), dtype=np.uint8)
    p_len = len(data) // 4
    data_transformed[0:p_len:1] = bytes0
    data_transformed[p_len:2*p_len:1] = bytes1
    data_transformed[2*p_len:3*p_len:1] = bytes2
    data_transformed[3*p_len:4*p_len:1] = bytes3
    return data_transformed, entropy_info


def plot(kth_order_ent, orig_entropy, tdt_kth_order_ent, tdt_orig_entropy, output_path, metadata):
    """
    Plot the entropy values for different fixed-width integer types and k-th order entropy.

    Parameters:
        kth_order_ent (dict): The k-th order entropy values.
        orig_entropy (dict): The original entropy values.
        tdt_kth_order_ent (dict): The TDT-transformed k-th order entropy values.
        tdt_orig_entropy (dict): The TDT-transformed original entropy values.
        byte_cluster_entropy (list): The byte cluster entropy values.
    """
    # plot 1x3 grid
    fig, axs = plt.subplots(1, 4, figsize=(25, 10))
    # compute max entropy across all blocks and all arrays and set all y-axis to this value
    max_entropy = 0
    for k, values in kth_order_ent.items():
        max_entropy = max(max_entropy, max(values))
    for w, values in orig_entropy.items():
        max_entropy = max(max_entropy, max(values))
    for k, values in tdt_kth_order_ent.items():
        max_entropy = max(max_entropy, max(values))
    for w, values in tdt_orig_entropy.items():
        max_entropy = max(max_entropy, max(values))

    # Set the same y-axis limit for all plots
    for ax in axs:
        ax.set_ylim(0, max_entropy)
        ax.set_yticks(np.arange(0, max_entropy, 0.3))
        ax.grid(True)


    # Plot (0,0): Block number vs original and k-th order entropy
    axs[0].set_title(f"Original and K-th Order Entropy for {metadata['dataset']} block size: {metadata['chunk_size']}")
    axs[0].set_xlabel("Block Number")
    axs[0].set_ylabel("Entropy")
    for k, values in kth_order_ent.items():
        axs[0].plot(range(len(values)), values, label=f"K-th Order (k={k})")
    #axs[0].plot(range(len(orig_entropy)), orig_entropy, label="Original Entropy", linestyle="--")
    axs[0].legend()

    axs[1].set_title(f"Original Entropy for different alphabet sizes for {metadata['dataset']} block size: {metadata['chunk_size']}")
    axs[1].set_xlabel("Block Number")
    axs[1].set_ylabel("Entropy")
    # do the same for original entropy
    for w, values in orig_entropy.items():
        axs[1].plot(range(len(values)), values, label=f"{w}-bit Entropy", linestyle="--")
    axs[1].legend()


    # Plot (0,2): Block number vs transformed k-th order and original entropy
    axs[2].set_title(f"TDT Transformed Entropies for {metadata['dataset']} block size: {metadata['chunk_size']}")
    axs[2].set_xlabel("Block Number")
    axs[2].set_ylabel("Entropy")
    for k, values in tdt_kth_order_ent.items():
        axs[2].plot(range(len(values)), values, label=f"TDT K-th Order (k={k})")
    #axs[2].plot(range(len(tdt_orig_entropy)), tdt_orig_entropy, label="TDT Original Entropy", linestyle="--")
    axs[2].legend()

    # Plot (0,1): Block number vs k-th order entropy for each component
    axs[3].set_title(f"TDT Transformed Entropies for different alphabet sizes for {metadata['dataset']} block size: {metadata['chunk_size']}")
    axs[3].set_xlabel("Block Number")
    axs[1].set_ylabel("Entropy")
    for w, values in tdt_orig_entropy.items():
        axs[3].plot(range(len(values)), values, label=f"TDT {w}-bit Entropy", linestyle="--")
    axs[1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_tensor(byte_cluster_entropy, output_path, ds_name):
    """
    Create a 3D tensor (block_id, k, component) and plot it per block_id.

    Parameters:
        byte_cluster_entropy (list[dict]): List of byte cluster entropies across blocks.
                    Each entry is a dictionary where keys are k-th orders and values are component lists.
        output_path (str): File path to save the plot.
    """
    # Step 1: Build the 3D tensor
    block_ids = len(byte_cluster_entropy)
    k_values = set()
    max_components = 0

    # Determine the range of `k` and maximum component size
    for block in byte_cluster_entropy:
        k_values.update(block.keys())
        for values in block.values():
            max_components = max(max_components, len(values))
    k_values = sorted(k_values)

    # Create a tensor with dimensions (block_id, k, component)
    tensor = np.zeros((block_ids, len(k_values), max_components))

    for i, block in enumerate(byte_cluster_entropy):  # Over blocks
        for k_index, k in enumerate(k_values):  # Over k-th orders
            if k in block:
                values = block[k]
                tensor[i, k_index, :len(values)] = values

    # Define line styles for each k
    line_styles = ['-', '--', '-.', ':']
    if len(k_values) > len(line_styles):  # Cycle styles if there are more `k` values than styles
        line_styles = line_styles * (len(k_values) // len(line_styles) + 1)

    # Step 2: Plot the tensor with block number on the x-axis
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_title(f"Entropy Per Block ID for dataset for {metadata['dataset']} block size: {metadata['chunk_size']}")
    ax.set_xlabel("Block Number")
    ax.set_ylabel("Entropy")


    for k_index, k in enumerate(k_values):  # Over k-th orders
        for component in range(max_components):  # Over components
            entropy_values = tensor[:, k_index, component]  # Select (block_id, k, component)
            if np.any(entropy_values):  # Only plot non-zero values
                ax.plot(
                    range(block_ids),
                    entropy_values,
                    linestyle=line_styles[k_index],
                    label=f"K={k}, Component={component}"
                )

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()




# calc_test_k = compute_kth_order_entropy_optimized("mississippi", 1, False)
# print(f"Test k-th order entropy: {calc_test_k}")

if __name__ == "__main__":
    chunk_size = -1
    contig_order = False
    # generate random floating point data aray and measure entropy for different fixed-width integer types
    data_size = 5000 #65000
    data = np.random.rand(data_size)
    dataset_name = "random"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        dataset_name = dataset_path.split('/')[-1].split('.')[0]
        # open tsv file with df
        data = pd.read_csv(dataset_path, sep='\t')
        sliced_data = data.values[:, 1].astype(np.float32)  # TODO: change this to general
        # flatten the data
        data = sliced_data.flatten()
    if len(sys.argv) > 2:
        chunk_size = int(sys.argv[3])
    if chunk_size == -1:
        chunk_size = 1000 #TODO a larger block size

    # we only needd a few blocks
    data_size = chunk_size * 10
    data = data[:data_size]

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

    byte_cluster_entropy = []

    for i in range(num_blocs):
        data_block = data_int[i*chunk_size:(i+1)*chunk_size]
        for k in range(num_entropies):
            kth_order_ent[str(k)].append(compute_kth_order_entropy_optimized(data_block, k))
            print(f"{k}-th order entropy for block {i}: {kth_order_ent[str(k)][-1]}")

        for w in [8, 16, 32, 64]:
            manual_entropy, entropy_scipy = compute_entropy_w(data, w)
            orig_entropy[str(w)].append(manual_entropy)
            print(f"Entropy for {w}-bit fixed-width integers: {manual_entropy} (SciPy: {entropy_scipy})")

        tdt_data, entropy_per_cluster = tdt_transform(data_block) # this is done assuming float32, TODO fix this
        byte_cluster_entropy.append(entropy_per_cluster)
        tdt_data_int = tdt_data.view(dtype=np.uint8)
        for w in [8, 16, 32, 64]:
            manual_entropy, entropy_scipy = compute_entropy_w(tdt_data, w)
            tdt_orig_entropy[str(w)].append(manual_entropy)
            print(f"Entropy for TDT-transformed {w}-bit fixed-width integers: {manual_entropy} (SciPy: {entropy_scipy})")

        for k in range(num_entropies):
            tdt_kth_order_ent[str(k)].append(compute_kth_order_entropy_optimized(tdt_data_int, k))
            print(f"{k}-th order entropy for TDT-transformed block {i}: {kth_order_ent[str(k)][-1]}")

    # plot the entropy values
    metadata = {'dataset': dataset_name, 'chunk_size': chunk_size}
    output_path = f"{dataset_name}_entropy_plot.png"
    plot(kth_order_ent, orig_entropy, tdt_kth_order_ent, tdt_orig_entropy, output_path, metadata)
    plot_tensor(byte_cluster_entropy, f"{dataset_name}_byte_cluster_entropy.png", metadata)




