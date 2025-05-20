
import numpy as np
import math
from collections import Counter
import scipy.stats as stats
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from enum import Enum

from entropy_analysis import calculate_kth_order_entropy_paper


class CompressionTool(Enum):
    ZSTD = "zstd"
    ZLIB = "zlib"
    BZ2 = "bz2"
    SNAPPY = "snappy"
    FASTLZ = "fastlz"
    HUFFMAN = "huffman"
    RLE = "rle"

compression_tool = "fastlz" #"huffman"

def calculate_entropy(probabilities):
    """Calculates the entropy of a probability distribution."""
    probabilities = probabilities[probabilities > 0]  # Avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_multivariate_mutual_information(datasets):
    """
    Calculates the multivariate mutual information (total correlation)
    of multiple uint8 arrays.

    Args:
        datasets: A list or array of NumPy arrays of dtype uint8.

    Returns:
        The multivariate mutual information.
    """

    # 1. Ensure all data arrays have the same shape
    if not all(dataset.shape == datasets[0].shape for dataset in datasets):
        raise ValueError("All input arrays must have the same shape.")

    # 2. Flatten the arrays for easier handling
    flattened_datasets = [dataset.flatten() for dataset in datasets]

    # 3. Calculate individual entropies
    individual_entropies = [
        calculate_entropy(np.histogram(flat_data, bins=256, range=(0, 256))[0] / len(flat_data))
        for flat_data in flattened_datasets
    ]

    # 4. Calculate joint entropy
    #    - Stack the flattened arrays to get combinations
    stacked_data = np.stack(flattened_datasets, axis=-1)
    #    - Use unique to efficiently count occurrences of each unique combination
    unique_combinations, counts = np.unique(stacked_data.view(dtype=[('', 'u1')] * len(datasets)), return_counts=True)
    joint_probabilities = counts / len(flattened_datasets[0])  # Normalize to get probabilities
    joint_entropy = calculate_entropy(joint_probabilities)

    # 5. Calculate multivariate mutual information
    mmi = sum(individual_entropies) - joint_entropy
    return mmi, joint_entropy


def compute_interaction(datasets):
    num_vars = len(datasets)
    joint_entropy_dict = {}
    interaction_info = 0
    for i in range(1, num_vars+1):
        # all combination of i from num_vars
        combinations = itertools.combinations(range(num_vars), i)
        for comb in combinations:
            # get the datasets for the combination
            subset = [datasets[j] for j in comb]
            # calculate the mutual information
            mmi, joint_entropy = calculate_multivariate_mutual_information(subset)
            #print(f"Mutual Information for combination {comb}: {mmi}, Joint Entropy: {joint_entropy}")
            interaction_info += (mmi *  pow(-1, i - 1))  # (-1)^(i-1) * mmi
            joint_entropy_dict[comb] = joint_entropy, mmi
    return interaction_info, joint_entropy_dict


def conditional_entropy(X, conditional_vars_indices, target_var_index):
    """
    Calculates the conditional entropy H(X[target_var_index] | X[conditional_vars_indices]).

    Args:
        X: A list or numpy array of numpy arrays, where each inner array represents the observed
           values of a random variable.  All inner arrays must have the same length.
        conditional_vars_indices: A list of integers, indicating the indices of the variables
                                  on which the entropy is conditioned.
        target_var_index: An integer, indicating the index of the variable for which the
                          conditional entropy is calculated.

    Returns:
        float: The conditional entropy H(X[target_var_index] | X[conditional_vars_indices]).
               Returns 0 if any input array is empty or if there are invalid indices.
    """
    # Basic input validation
    if not isinstance(X, (list, np.ndarray)) or len(X) == 0:
        return 0.0
    if not all(isinstance(arr, np.ndarray) for arr in X):
        return 0.0
    if not all(len(arr) == len(X[0]) for arr in X) or len(X[0]) == 0:
        return 0.0
    if not isinstance(conditional_vars_indices, list) or not all(isinstance(i, int) and 0 <= i < len(X) for i in conditional_vars_indices):
        return 0.0
    if not isinstance(target_var_index, int) or not 0 <= target_var_index < len(X):
        return 0.0
    if target_var_index in conditional_vars_indices:
        return 0.0 # Target variable cannot be in the conditioning set.

    n = len(X[0])  # Number of data points
    num_vars = len(X)

    # Create a list of the variables to condition on
    conditioning_vars = [X[i] for i in conditional_vars_indices]

    # Calculate joint probabilities of the conditioning variables
    unique_conditioning_values, counts_conditioning_values = np.unique(
        list(zip(*conditioning_vars)), axis=0, return_counts=True )
    p_conditioning_values = counts_conditioning_values / n

    conditional_entropy = 0.0
    for i, conditioning_value_tuple in enumerate(unique_conditioning_values):
        # Find indices where the conditioning variables match the current combination
        indices = np.where(np.all(np.array(conditioning_vars).T == conditioning_value_tuple, axis=1))[0]

        X_target_values_given_conditioning = X[target_var_index][indices]

        # Calculate p(x_target | x_conditioning)
        unique_X_target_given_conditioning, counts_X_target_given_conditioning = np.unique(
            X_target_values_given_conditioning, return_counts=True
        )
        p_X_target_given_conditioning = counts_X_target_given_conditioning / len(X_target_values_given_conditioning)

        # Calculate the entropy of the target variable given the conditioning variables
        entropy_X_target_given_conditioning = -np.sum(p_X_target_given_conditioning * np.log2(p_X_target_given_conditioning))

        # Accumulate the conditional entropy
        conditional_entropy += p_conditioning_values[i] * entropy_X_target_given_conditioning

    return conditional_entropy


def get_conditional_entropies(datasets):
    """
    Calculate the conditional entropies of each variable in the dataset given all other variables.

    Args:
        datasets: A list or array of NumPy arrays of dtype uint8.

    Returns:
        A list of conditional entropies.
    """
    num_vars = len(datasets)
    conditional_entropies = []
    for v in range(num_vars):
        dict_c = {}
        dict_c[v] = {}
        for i in range(1, num_vars + 1):
            # all combination of i from num_vars
            combinations = itertools.combinations(range(num_vars), i)
            for comb in combinations:
                if v in comb:
                    dict_c[v][comb] = 0
                    continue
                # Get the conditional entropy of variable i given all other variables
                cond_entropy = conditional_entropy(datasets, [j for j in comb if j != v], v)
                dict_c[v][comb] = cond_entropy
        conditional_entropies.append(dict_c)
    return conditional_entropies


def cross_relative_entropy(dataset1, dataset2):
    # Count occurrences of each element
    counts1 = Counter(dataset1)
    counts2 = Counter(dataset2)
    total1 = len(dataset1)
    total2 = len(dataset2)
    # max size of two counts
    max_size = max(len(counts1), len(counts2))
    pk, qk = np.zeros(max_size), np.zeros(max_size)
    for i in range(len(counts1.values())):
        pk[i] = counts1[i] / total1
    for i in range(len(counts2.values())):
        qk[i] = counts2[i] / total2

    # Calculate probabilities and cross entropy
    relative_entropy = stats.entropy(pk, qk)
    #kl_diver = scipy.special.kl_divergence(pk, qk)
    cross_entropy = stats.entropy(pk) + relative_entropy
    return relative_entropy, cross_entropy


def compress_with_zstd(data, level=3):
    if compression_tool == CompressionTool.HUFFMAN:
        # Use Huffman coding for compression
        from compression_tools import huffman_compress
        compressed = huffman_compress(data)
    elif compression_tool == CompressionTool.FASTLZ:
        # Use FastLZ for compression
        from compression_tools import fastlz_compress
        compressed = fastlz_compress(data)
    else:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=level)
        # if data is not contiguous, make it contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        compressed = cctx.compress(data)
    # comp ratio
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio


def compute_entropy(stream):
    # Count occurrences of each element
    counts = Counter(stream)
    total = len(stream)
    # Calculate probabilities and entropy
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def get_compression_ratio_stats(components, merging_indices, original_dataset=None):
    """
    Calculate the compression ratio after merging components.

    Parameters:
    - components (list): List of components.
    - merging_indices (list): Indices of components to merge.

    Returns:
    - float: Compression ratio after merging.
    """
    merged_comp, merged_comp_entropy, merged_k2_entropy = [], [], []
    for m_indices in merging_indices:
        comp = np.concatenate([components[i] for i in m_indices], axis=0)
        merged_comp.append(comp)
    compressed_size_merged = 0
    for comp in merged_comp:
        comp = np.ascontiguousarray(comp)
        compressed, comp_ratio_tmp = compress_with_zstd(comp)
        compressed_size_merged += len(compressed)
        entropy_comp = compute_entropy(comp)
        merged_comp_entropy.append(entropy_comp)
        entropy_k2 = calculate_kth_order_entropy_paper(comp, 1)
        merged_k2_entropy.append(entropy_k2)
    relative_entropy, merged_cross_entropy = np.zeros((len(merged_comp), len(merged_comp))), np.zeros((len(merged_comp), len(merged_comp)))
    for i in range(len(merged_comp)):
        for j in range(i + 1, len(merged_comp)):
            relative_entropy[i][j], merged_cross_entropy[i][j] = cross_relative_entropy(merged_comp[i], merged_comp[j])
            merged_cross_entropy[j][i] = merged_cross_entropy[i][j]
            relative_entropy[j][i] = relative_entropy[i][j]

    reordered_data = np.concatenate(merged_comp, axis=0)
    compressed_reordered, compressed_size_reo = compress_with_zstd(reordered_data)
    entropy_k2_reorder = calculate_kth_order_entropy_paper(reordered_data, 1)
    # flatten the components array
    if original_dataset is None:
        original_dataset = np.concatenate(components, axis=0)
    comp_original, cr_original = compress_with_zstd(original_dataset)
    decomp_cr_all = len(original_dataset.tobytes()) / compressed_size_merged
    comp_ratio_reo = len(original_dataset.tobytes()) / len(compressed_reordered)
    return comp_ratio_reo, decomp_cr_all, cr_original, merged_comp_entropy, merged_cross_entropy, relative_entropy, merged_k2_entropy, entropy_k2_reorder

def generate_partitions(elements):
    """
    Generate all possible partitions (clusterings) of a set.

    Args:
        elements (list): The set to partition.

    Returns:
        list: A list of all partitions, where each partition is a list of subsets.
    """
    if len(elements) == 0:
        return [[]]

    # Recursive generation of partitions
    first, rest = elements[0], elements[1:]
    partitions = []
    for smaller_partition in generate_partitions(rest):
        # Add the first element to each subset in the smaller partition
        for i in range(len(smaller_partition)):
            new_partition = smaller_partition[:i] + [[first] + smaller_partition[i]] + smaller_partition[i + 1:]
            partitions.append(new_partition)
        # Or create a new subset with the first element
        partitions.append([[first]] + smaller_partition)
    return partitions


def all_possible_merging(comp_array, ds_name, original_dataset=None):
    bit_width = len(comp_array)
    interaction_info, joint_entropy_dict = compute_interaction(comp_array)
    print(f"Interaction Information: {interaction_info}")
    # print(f"Joint Entropy: {joint_entropy_dict}")
    # Calculate conditional entropies
    conditional_entropies = get_conditional_entropies(comp_array)
    # for row in conditional_entropies:
    #     for key, value in row.items():
    #         print(f"Conditional Entropy for variable {key}: {value}")
    #for i in range(1, bit_width+1):
    elements = list(range(0, bit_width ))
    stat_dict = []
    partitions = generate_partitions(elements)
    for comb in partitions:
        dicts_or_stat_item = {}
        dicts_or_stat_item["Name"], dicts_or_stat_item["tool"] = ds_name, compression_tool
        cr_reordered, decomp_cr, orignal_cr, entropy_combined, cross_ent_combined, rel_ent_combined, k2_entropy_combined, k2_entropy_reorder = get_compression_ratio_stats(comp_array, comb, original_dataset)
        #print(f"Compression Ratio for combination {comb}: reorded: {cr_reordered}, Decompsed CR: {decomp_cr} vs original: {orignal_cr}")
        print(f"Entropy of the merged components: {entropy_combined}")
        # print(f"Cross Entropy of the merged components \n {cross_ent_combined}\n\n")
        # print(f"Relative Entropy of the merged components \n {rel_ent_combined}\n\n")
        # print(f"K2 Entropy of the merged components \n {k2_entropy_combined}\n\n")
        # print(f"K2 Entropy of the reordered components \n {k2_entropy_reorder}\n\n")
        joint_entropy_sum, mmi_sum = 0, 0
        for cluster in comb:
            # convert it to tuple
            cluster = tuple(cluster)
            if cluster in joint_entropy_dict:
                joint_entropy_sum += joint_entropy_dict[cluster][0]
                mmi_sum += joint_entropy_dict[cluster][1]
            else:
                # print(f"Cluster {cluster} not found in joint_entropy_dict")
                pass

        dicts_or_stat_item["joint entropy"] = joint_entropy_sum
        dicts_or_stat_item["mmi"] = mmi_sum
        dicts_or_stat_item["clustering"] = comb
        dicts_or_stat_item["reordered cr"] = cr_reordered
        dicts_or_stat_item["decomposed cr"] = decomp_cr
        dicts_or_stat_item["original cr"] = orignal_cr
        dicts_or_stat_item["entropy"], dicts_or_stat_item["cross entropy"], dicts_or_stat_item["relative entropy"] = entropy_combined, cross_ent_combined, rel_ent_combined
        dicts_or_stat_item["k2 entropy"] = k2_entropy_combined
        dicts_or_stat_item["k2 entropy reorder"] = k2_entropy_reorder
        stat_dict.append(dicts_or_stat_item)


    return stat_dict, interaction_info, joint_entropy_dict, conditional_entropies



def generate_byte_stream(size, entropy):
    """
    Generate a byte stream of a given size with a specified entropy.

    Parameters:
    - size (int): The size of the byte stream to generate.
    - entropy (float): The desired entropy (0 to 8, as entropy for bytes is max 8 bits).

    Returns:
    - bytes: A byte stream of the specified size and entropy.
    """
    if entropy < 0 or entropy > 65:
        raise ValueError("Entropy must be between 0 and 8.")

    # Calculate the number of unique byte values needed
    num_symbols = int(2 ** entropy)

    # Create probabilities for the symbols
    probabilities = np.ones(num_symbols) / num_symbols

    # Generate the byte stream
    #np.random.seed(int(time.time()))
    symbols = np.random.choice(range(num_symbols), size=size, p=probabilities)

    # cast it as int8
    if entropy <= 8:
        symbols = symbols.astype(np.uint8)
    elif entropy <= 16:
        symbols = symbols.astype(np.uint16)
    elif entropy <= 32:
        symbols = symbols.astype(np.uint32)
    elif entropy <= 64:
        symbols = symbols.astype(np.uint64)
    else:
        raise ValueError("Entropy too high for byte stream generation.")
    # random shuffle the symbols
    #np.random.shuffle(symbols)
    return symbols


def generate_float_stream(size, entropy_per_byte_array):
    num_components = len(entropy_per_byte_array)
    entropy_array = np.zeros(num_components)
    byte_width = 4
    if byte_width > 8:
        raise ValueError("Bit width exceeds 64 bits.")
    byte_array = []

    if num_components == 4:
        packed_array = np.zeros(size * num_components, dtype=np.uint8)
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[0]))
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[1]))
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[2]))
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[3]))
        entropy_array[0] = compute_entropy(byte_array[0])
        entropy_array[1] = compute_entropy(byte_array[1])
        entropy_array[2] = compute_entropy(byte_array[2])
        entropy_array[3] = compute_entropy(byte_array[3])
        for i in range(0, size*num_components, num_components):
            packed_array[i] = byte_array[0][i//num_components]
            packed_array[i + 1] = byte_array[1][i//num_components]
            packed_array[i + 2] = byte_array[2][i//num_components]
            packed_array[i + 3] = byte_array[3][i//num_components]

    if num_components == 2:
        packed_array = np.zeros(size * num_components, dtype=np.uint16)
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[0]))
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[1]))
        #print(f"Entropy of the generated float stream 1 : {compute_entropy(byte_array[0])} \n\n")
        #print(f"Entropy of the generated float stream 2 : {compute_entropy(byte_array[1])} \n\n")
        entropy_array[0] = compute_entropy(byte_array[0])
        entropy_array[1] = compute_entropy(byte_array[1])
        for i in range(0, size*num_components, num_components):
            packed_array[i] = byte_array[0][i]
            packed_array[i + 1] = byte_array[1][i]

    if num_components == 1:
        packed_array = np.zeros(size, dtype=np.uint32)
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[0]))
        entropy_array[0] = compute_entropy(byte_array[0])
        for i in range(0, size, num_components):
            packed_array[i] = byte_array[0][i]
    return packed_array, byte_array, entropy_array


def process_dictionary(dicts_or_stat, interaction_info, joint_entropy_dict, conditional_entropies):
    # plot average entropy vs compression ratio

    stat_dict_df = pd.DataFrame(dicts_or_stat)
    ds_name = stat_dict_df["Name"][0]
    combined_ent = stat_dict_df["entropy"]
    combined_entk2 = stat_dict_df["k2 entropy"]
    CR_original = stat_dict_df["original cr"]
    CR_reordered = stat_dict_df["reordered cr"]
    CR_decomp = stat_dict_df["decomposed cr"]
    joint_entropy = stat_dict_df["joint entropy"]
    mmi = stat_dict_df["mmi"]

    comp_Tool = stat_dict_df["tool"][0]
    average_ent, std_ent, std_ent2 = [], [], []
    for v in combined_ent:
        average_ent.append(np.mean(v))
        std_ent.append(np.std(v))
    for v in combined_entk2:
        average_ent.append(np.mean(v))
        std_ent2.append(np.std(v))
    # create a dataframe
    label1, label2 = "STD Entropy", "STD Entropy K2"
    label1, label2 = "joint entropy", "mmi"
    # df = pd.DataFrame({
    #     "STD Entropy": std_ent,
    #     "STD Entropy K2": std_ent2,
    #     "Compression Ratio": CR_decomp,
    # })
    df = pd.DataFrame({
        label1: joint_entropy,
        label2: mmi,
        "Compression Ratio": CR_decomp,
    })
    # plot the data
    #sns.scatterplot(data=df, x="STD Entropy", y="Compression Ratio")
    # create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plot the first subplot
    sns.scatterplot(data=df, x=label1, y="Compression Ratio", ax=ax1)
    ax1.set_title(f"{ds_name} : {comp_Tool}")
    ax1.set_xlabel(label1)
    ax1.set_ylabel("Decomposed Compression Ratio")
    # add a regression line
    sns.regplot(data=df, x=label1, y="Compression Ratio", scatter=False, color='red', ax=ax1)
    # show the r2
    r2 = stats.linregress(df[label1], df["Compression Ratio"]).rvalue ** 2
    print(f"R2 for STD Entropy vs Compression Ratio: {r2}")
    ax1.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax1.transAxes, fontsize=10, verticalalignment='top', color='red')

    # plot the second subplot
    sns.scatterplot(data=df, x=label2, y="Compression Ratio", ax=ax2)
    ax2.set_title(f"{ds_name} : {comp_Tool}")
    ax2.set_xlabel(label2)
    ax2.set_ylabel("Decomposed Compression Ratio")
    sns.regplot(data=df, x=label2, y="Compression Ratio", scatter=False, color='blue', ax=ax2)
    # show the r2
    r2 = stats.linregress(df[label2], df["Compression Ratio"]).rvalue ** 2
    ax2.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax2.transAxes, fontsize=10, verticalalignment='top', color='blue')
    print(f"R2 for STD Entropy K2 vs Compression Ratio: {r2}")
    plt.show()


def get_best_compression_clustering(stat_dict):
    """
    Get the best compression clustering based on the compression ratio.

    Args:
        stat_dict: A list of dictionaries containing the clustering information.

    Returns:
        The best clustering based on the compression ratio.
    """
    best_cr = 0
    best_clustering = {}
    for item in stat_dict:
        if item["decomposed cr"] > best_cr:
            best_cr = item["decomposed cr"]
            best_clustering = item
    return best_clustering


def all_possible_ent(sampling=0):
    all_possible = []
    # all possible combination of 4 variables between 1 to 7
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                for l in range(1, 8):
                    all_possible.append([i, j, k, l])
    if sampling > 0:
        # generate a random array of integers between 1 and sampling
        rand_int = np.random.randint(1, len(all_possible), size=(sampling))
        # get the random samples
        all_possible = np.array([all_possible[i] for i in rand_int])
        #all_possible = np.random.choice(all_possible, sampling, replace=False)
    return all_possible

best_clustering, all_stats = [], []
data_set_name = ""
compression_tool = CompressionTool.HUFFMAN
# generate a list of entropies arrays
entropy_array = all_possible_ent(10) #[[2, 2, 2, 2], [6, 2, 2, 3], [2, 6, 2, 6], [2, 2, 6, 2], [5, 1, 2, 6], [4, 4, 4, 4], [7, 3, 4, 1]]
for entropies in entropy_array:
    if data_set_name == "":
        #entropies = [7, 2, 1, 4]
        float_stream, comp_array, comp_entropy_array = generate_float_stream(
                1*1024, entropies)
        data_set_name = "Synthetic"+str(entropies)
        string = False
        if string:
            from string_float import load_20newsgroups_dataset, decompose_strings
            dataset = load_20newsgroups_dataset()
            merged_dataset = ""
            for item in dataset:
                merged_dataset = merged_dataset + item
            float_stream = np.frombuffer(merged_dataset[:4*1024].encode(), dtype=np.uint8)
            b0, b1, b2, b3 = decompose_strings(float_stream)
            # convert list to uint8 array
            comp_array = []
            comp_array.append(np.array(b0))
            comp_array.append(np.array(b1))
            comp_array.append(np.array(b2))
            comp_array.append(np.array(b3))
        is_float = False
        if is_float:
            from utils import generate_smooth_array
            comp_array = []
            symbols = generate_smooth_array(len(float_stream) // 4)
            size, num_components = len(float_stream) // 4, 4
            # cast symbols as uint8
            float_stream = symbols.view(np.uint8)
            comp_array.append(np.array(float_stream[0: 4 * size: 4]))
            comp_array.append(np.array(float_stream[1: 4 * size: 4]))
            comp_array.append(np.array(float_stream[2: 4 * size: 4]))
            comp_array.append(np.array(float_stream[3: 4 * size: 4]))
    else:
        # TODO: load the dataset and get the entropy of each component
        pass

    dicts_or_stat, interaction_info, joint_entropy_dict, conditional_entropies = all_possible_merging(comp_array, data_set_name, float_stream)
    best_cluster = get_best_compression_clustering(dicts_or_stat)
    all_stats.append(dicts_or_stat)

    best_clustering.append(best_cluster)
    process_dictionary(dicts_or_stat, interaction_info, joint_entropy_dict, conditional_entropies)
    data_set_name = ""
    # best_df = pd.DataFrame(best_clustering)
    # best_df.to_csv("best_clustering.csv", index=False)
    # all_df = pd.DataFrame(all_stats)
    # all_df.to_csv("all_clustering.csv", index=False)
#process_dictionary(best_clustering, None, None, None)
