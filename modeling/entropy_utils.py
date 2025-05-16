
import numpy as np
import math
from collections import Counter
import scipy.stats as stats
import itertools



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
            joint_entropy_dict[comb] = joint_entropy
    return interaction_info, joint_entropy_dict

def cross_entropy(dataset1, dataset2):
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
    cross_entropy = stats.entropy(pk, qk)
    #kl_diver = scipy.special.kl_divergence(pk, qk)
    return cross_entropy


def compress_with_zstd(data, level=3):
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


def get_compression_ratio(components, merging_indices, original_dataset=None):
    """
    Calculate the compression ratio after merging components.

    Parameters:
    - components (list): List of components.
    - merging_indices (list): Indices of components to merge.

    Returns:
    - float: Compression ratio after merging.
    """
    merged_comp, merged_comp_entropy = [], []
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
    merged_cross_entropy = np.zeros((len(merged_comp), len(merged_comp)))
    for i in range(len(merged_comp)):
        for j in range(i + 1, len(merged_comp)):
            merged_cross_entropy[i][j] = cross_entropy(merged_comp[i], merged_comp[j])
            merged_cross_entropy[j][i] = merged_cross_entropy[i][j]

    reordered_data = np.concatenate(merged_comp, axis=0)
    compressed_reordered, compressed_size_reo = compress_with_zstd(reordered_data)
    # flatten the components array
    if original_dataset is None:
        original_dataset = np.concatenate(components, axis=0)
    comp_original, cr_original = compress_with_zstd(original_dataset)
    decomp_cr_all = len(original_dataset.tobytes()) / compressed_size_merged
    comp_ratio_reo = len(original_dataset.tobytes()) / len(compressed_reordered)
    return comp_ratio_reo, decomp_cr_all, cr_original, merged_comp_entropy, merged_cross_entropy

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


def all_possible_merging(comp_array, original_dataset=None):
    bit_width = len(comp_array)
    for i in range(1, bit_width+1):
        elements = list(range(0, bit_width ))

        # Generate all combinations of 3 elements
        #combinations_of_3 = list(itertools.combinations(elements, i))
        partitions = generate_partitions(elements)
        for comb in partitions:
            cr_reordered, decomp_cr, rignal_cr, entropy_combined, cross_ent_combined = get_compression_ratio(comp_array, comb, original_dataset)
            print(f"Compression Ratio for combination {comb}: reorded: {cr_reordered}, Decompsed CR: {decomp_cr} vs original: {rignal_cr}")
            print(f"Entropy of the merged components: {entropy_combined}")
            print(f"Cross Entropy of the merged components: {cross_ent_combined}\n\n")


    interaction_info, joint_entropy_dict = compute_interaction(comp_array)
    print(f"Interaction Information: {interaction_info}")
    print(f"Joint Entropy: {joint_entropy_dict}")



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

data_set_name = ""
if data_set_name == "":
    entropies = [7, 2, 1, 7]
    float_stream, comp_array, comp_entropy_array = generate_float_stream(
            2 * 1024 * 1024, entropies)
else:
    # TODO: load the dataset and get the entropy of each component
    pass

all_possible_merging(comp_array, float_stream)