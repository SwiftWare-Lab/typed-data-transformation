import numpy as np
import pandas as pd
import scipy.stats as stats

import math
from collections import Counter
import time

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
    return cross_entropy

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
        for i in range(0, size, num_components):
            packed_array[i] = byte_array[0][i]
            packed_array[i + 1] = byte_array[1][i]
            packed_array[i + 2] = byte_array[2][i]
            packed_array[i + 3] = byte_array[3][i]

    if num_components == 2:
        packed_array = np.zeros(size * num_components, dtype=np.uint16)
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[0]))
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[1]))
        #print(f"Entropy of the generated float stream 1 : {compute_entropy(byte_array[0])} \n\n")
        #print(f"Entropy of the generated float stream 2 : {compute_entropy(byte_array[1])} \n\n")
        entropy_array[0] = compute_entropy(byte_array[0])
        entropy_array[1] = compute_entropy(byte_array[1])
        for i in range(0, size, num_components):
            packed_array[i] = byte_array[0][i]
            packed_array[i + 1] = byte_array[1][i]

    if num_components == 1:
        packed_array = np.zeros(size, dtype=np.uint32)
        byte_array.append(generate_byte_stream(size, entropy_per_byte_array[0]))
        entropy_array[0] = compute_entropy(byte_array[0])
        for i in range(0, size, num_components):
            packed_array[i] = byte_array[0][i]
    return packed_array, byte_array, entropy_array


def compress_with_zstd(data, level=3):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    # comp ratio
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio

def plot_time_series(data, title):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style of seaborn
    sns.set(style="darkgrid")

    # Create a time series plot
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


def get_compression_ratio(components, merging_indices):
    """
    Calculate the compression ratio after merging components.

    Parameters:
    - components (list): List of components.
    - merging_indices (list): Indices of components to merge.

    Returns:
    - float: Compression ratio after merging.
    """
    merged_component = np.concatenate([components[i] for i in merging_indices], axis=0)
    compressed, comp_ratio = compress_with_zstd(merged_component)
    entropy_merged = compute_entropy(merged_component)
    decomposed_size = 0
    for c in components:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
    # flatten the components array
    float_stream = np.concatenate(components, axis=0)
    decomp_cr = len(float_stream.tobytes()) / decomposed_size
    return comp_ratio, decomp_cr, entropy_merged

def check_mutual_information():
    float_stream, comp_array, comp_entropy_array = generate_float_stream(
        2 * 1024 * 1024, [2, 2, 4, 2])
    float_stream2, comp_array2, comp_entropy_array2 = generate_float_stream(
        2 * 1024 * 1024, [7, 7, 7, 7])
    cross_table = np.zeros((len(comp_array), len(comp_array)))
    comp_data, original_cr = compress_with_zstd(float_stream)
    entropy_combined = compute_entropy(float_stream)
    entropy_vals = np.zeros((len(comp_array)))
    for i in range(len(comp_array)):
        data = comp_array[i]
        entropy_vals[i] = compute_entropy(data)
    print(" Original compression ratio: ", original_cr)

    print(f" ---ratio : {np.sum(entropy_vals)/entropy_combined} ---")
    for i in range(len(comp_array)):
        for j in range(i + 1, len(comp_array)):
            data1 = comp_array[i]
            data2 = comp_array[j]
            e1 = compute_entropy(data1)
            e2 = compute_entropy(data2)
            # Calculate mutual information
            mi = cross_entropy(data1, data2)
            cross_table[i][j] = mi
            cr, decomp_cr, e_merged = get_compression_ratio(comp_array, [i, j])
            e_sum = e_merged
            for k in range(len(comp_array)):
                if k != i and k != j:
                    e_sum += compute_entropy(comp_array[k])
            print(f" ---ratio : {e_sum / entropy_combined} ---")
            print(f"Mutual Information between component {i} and {j}: {mi}  -> {cr} vs {decomp_cr} vs {original_cr}")



check_mutual_information()
exit(1)


log_array = []

# for j in range(1, 17):
#     for i in range(1, 17):
#         dict_log = {}
#         float_stream, comp_array, comp_entropy_array = generate_float_stream(2*1024*1024, [17-i, 17-j])
#
#         # print(f"Generated float stream of size {len(float_stream)} with entropy .")
#         # print(f"Entropy of the generated float stream: {compute_entropy(float_stream)} \n\n")
#         decomposed_size = 0
#         entropy_per_byte_array = []
#         for c in comp_array:
#             decompressed, comp_ratio = compress_with_zstd(c)
#             decomposed_size += len(decompressed)
#             entropy_per_byte_array.append(compute_entropy(c))
#         decomp_cr = len(float_stream.tobytes()) / decomposed_size
#         orig_cop, original_cr = compress_with_zstd(float_stream)
#         entropy_bytes = compute_entropy(float_stream)
#         # cast it as int32
#         float_stream_i32 = float_stream.view(dtype=np.uint32)
#         #orig_cop32, original_cr32 = compress_with_zstd(float_stream_i32)
#         col_ord = np.concatenate(comp_array, axis=0)
#         transformed_data, transform_cr = compress_with_zstd(col_ord)
#         print(f"Entropy-orig: {entropy_bytes}  Entropy-32: {compute_entropy(float_stream_i32)}, array: {entropy_per_byte_array}, std: {np.std(entropy_per_byte_array)}")
#         print(f"CR decomp:  {decomp_cr}, order: {transform_cr}, original: {original_cr}\n\n")
#         dict_log["entropy"] = str(entropy_per_byte_array)
#         dict_log["entropy bytes"] = str(entropy_bytes)
#         dict_log["entropy int32"] = str(compute_entropy(float_stream_i32))
#         dict_log["decomp cr"] = str(decomp_cr)
#         dict_log["transform cr"] = str(transform_cr)
#         dict_log["original cr"] = str(original_cr)
#         dict_log["original size"] = str(len(float_stream))
#         log_array.append(dict_log)
#     # store as pandas df
#     df = pd.DataFrame(log_array)
#     df.to_csv(f"entropy_log.csv", index=False)
#
#
# exit(1)


# for l in range(1, 9):
#     for k in range(1, 9):
#         for j in range(1, 9):
#             for i in range(1, 9):
#                 dict_log = {}
#                 float_stream, comp_array, comp_entropy_array = generate_float_stream(2*1024*1024, [i, j, k, l])
#
#                 # print(f"Generated float stream of size {len(float_stream)} with entropy .")
#                 # print(f"Entropy of the generated float stream: {compute_entropy(float_stream)} \n\n")
#                 decomposed_size = 0
#                 entropy_per_byte_array = []
#                 for c in comp_array:
#                     decompressed, comp_ratio = compress_with_zstd(c)
#                     decomposed_size += len(decompressed)
#                     entropy_per_byte_array.append(compute_entropy(c))
#                 decomp_cr = len(float_stream.tobytes()) / decomposed_size
#                 orig_cop, original_cr = compress_with_zstd(float_stream)
#                 entropy_bytes = compute_entropy(float_stream)
#                 # cast it as int32
#                 float_stream_i32 = float_stream.view(dtype=np.uint32)
#                 #orig_cop32, original_cr32 = compress_with_zstd(float_stream_i32)
#                 col_ord = np.concatenate(comp_array, axis=0)
#                 transformed_data, transform_cr = compress_with_zstd(col_ord)
#                 print(f"Entropy-orig: {entropy_bytes}  Entropy-32: {compute_entropy(float_stream_i32)}, array: {entropy_per_byte_array}, std: {np.std(entropy_per_byte_array)}")
#                 print(f"CR decomp:  {decomp_cr}, order: {transform_cr}, original: {original_cr}\n\n")
#                 dict_log["entropy"] = str(entropy_per_byte_array)
#                 dict_log["entropy bytes"] = str(entropy_bytes)
#                 dict_log["entropy int32"] = str(compute_entropy(float_stream_i32))
#                 dict_log["decomp cr"] = str(decomp_cr)
#                 dict_log["transform cr"] = str(transform_cr)
#                 dict_log["original cr"] = str(original_cr)
#                 dict_log["original size"] = str(len(float_stream))
#                 log_array.append(dict_log)
#             # store as pandas df
#             df = pd.DataFrame(log_array)
#             df.to_csv(f"entropy_log.csv", index=False)


        #plot_time_series(float_stream_i32, f"randome data{i}")

#
#
#
#
# for i in range(1, 8):
#     float_stream, comp_array, comp_entropy_array = generate_float_stream(2*1024*1024, [7, 7, i, 8-i])
#     # print(f"Generated float stream of size {len(float_stream)} with entropy .")
#     # print(f"Entropy of the generated float stream: {compute_entropy(float_stream)} \n\n")
#     decomposed_size = 0
#     entropy_per_byte_array = []
#     for c in comp_array:
#         decompressed, comp_ratio = compress_with_zstd(c)
#         decomposed_size += len(decompressed)
#         entropy_per_byte_array.append(compute_entropy(c))
#     decomp_cr = len(float_stream.tobytes()) / decomposed_size
#     orig_cop, original_cr = compress_with_zstd(float_stream)
#     entropy_bytes = compute_entropy(float_stream)
#     # cast it as int32
#     float_stream_i32 = float_stream.view(dtype=np.uint32)
#     #orig_cop32, original_cr32 = compress_with_zstd(float_stream_i32)
#     print(f"Entropy-orig: {entropy_bytes}  Entropy-32: {compute_entropy(float_stream_i32)}, array: {entropy_per_byte_array}, std: {np.std(entropy_per_byte_array)}")
#     print(f"CR decomp:  {decomp_cr}, original: {original_cr}\n\n")

# print(log_array)
# exit(1)
log_dict = {}
for i in range(1, 31):
    print(f" -------------- Entropy: {i} --------------")
    log_dict
    float_stream, comp_array, comp_entropy_array = generate_float_stream(2*1024*1024, [i])
    print(f"Generated float stream of size {len(float_stream)} with entropy .")
    print(f"Entropy of the generated float stream: {compute_entropy(float_stream)}")
    # cast it as int32
    float_stream_i32 = float_stream.view(dtype=np.uint32)
    print(f"Entropy of the generated float stream: {compute_entropy(float_stream_i32)}")
    # compress the data
    comp, comp_ratio = compress_with_zstd(float_stream)
    print(f"CR: {comp_ratio}")
    # make each a component
    byte_array = float_stream_i32.view(dtype=np.uint8)
    comp_array = []
    comp_array.append(byte_array[0: len(byte_array) : 4])
    comp_array.append(byte_array[1: len(byte_array) : 4])
    comp_array.append(byte_array[2: len(byte_array) : 4])
    comp_array.append(byte_array[3: len(byte_array) : 4])
    comp_array_np = []
    comp_array_np.append(np.array(comp_array[0]))
    comp_array_np.append(np.array(comp_array[1]))
    comp_array_np.append(np.array(comp_array[2]))
    comp_array_np.append(np.array(comp_array[3]))
    # flatten the array
    comp_array_flattened = np.concatenate(comp_array_np, axis=0)
    transformed_data, transform_cr = compress_with_zstd(comp_array_flattened)
    print(f"CR transformed: {transform_cr}")

    # compress each component
    decomposed_size = 0
    entropy_array = []
    for c in comp_array_np:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
        #print(f"entropy of: {compute_entropy(c)}")
        entropy_array.append(compute_entropy(c))
    decomp_cr = len(float_stream.tobytes()) / decomposed_size

    # compute standard deviation of entropy_array
    entropy_array = np.array(entropy_array)
    std_entropy = np.std(entropy_array)
    print(f"Entropy array: {entropy_array} with std: {std_entropy} ")
    print(f"-> CR dec: {decomp_cr} \n")

    new_last_byte = np.concatenate( (comp_array_np[2], comp_array_np[3]), axis=0)

    comp_array_3 = []
    comp_array_3.append(comp_array_np[0])
    comp_array_3.append(comp_array_np[1])
    comp_array_3.append(new_last_byte)


    comp_array3_flattened = np.concatenate(comp_array_3, axis=0)
    transformed_data3, transform_cr3 = compress_with_zstd(comp_array3_flattened)
    print(f"CR transformed3: {transform_cr3}")

    decomposed_size = 0
    entropy_array = []
    for c in comp_array_3:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
        #print(f"entropy of: {compute_entropy(c)}")
        entropy_array.append(compute_entropy(c))
    decomp_cr = len(float_stream.tobytes()) / decomposed_size
    # compute standard deviation of entropy_array
    entropy_array = np.array(entropy_array)
    std_entropy = np.std(entropy_array)
    print(f"Entropy array 3: {entropy_array} with std3: {std_entropy} ")
    print(f"-> CR dec 3:  {decomp_cr} \n")



    interleaved = np.zeros( (len(comp_array_np[2])*2), dtype=np.uint8)
    interleaved[0::2] = comp_array_np[2]
    interleaved[1::2] = comp_array_np[3]
    comp_array_3_inter = []
    comp_array_3_inter.append(comp_array_np[0])
    comp_array_3_inter.append(comp_array_np[1])
    comp_array_3_inter.append(interleaved)
    comp_array3_inter_flattened = np.concatenate(comp_array_3_inter, axis=0)
    transformed_data3_inter, transform_cr3_inter = compress_with_zstd(comp_array3_inter_flattened)
    print(f"CR transformed3 inter: {transform_cr3_inter}")
    decomposed_size = 0
    entropy_array = []
    for c in comp_array_3_inter:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
        #print(f"entropy of: {compute_entropy(c)}")
        entropy_array.append(compute_entropy(c))
    decomp_cr = len(float_stream.tobytes()) / decomposed_size
    # compute standard deviation of entropy_array
    entropy_array = np.array(entropy_array)
    std_entropy = np.std(entropy_array)
    print(f"Entropy array 3 inter: {entropy_array} with std3 inter: {std_entropy} ")
    print(f"-> CR dec 3 inter:  {decomp_cr} \n")



    first_two_bytes = np.concatenate(comp_array_np[0:2], axis=0)
    comp_array_3_inter_h = []
    comp_array_3_inter_h.append(first_two_bytes)
    comp_array_3_inter_h.append(comp_array_np[2])
    comp_array_3_inter_h.append(comp_array_np[3])
    comp_array3_inter_h_flattened = np.concatenate(comp_array_3_inter_h, axis=0)
    transformed_data3_inter_h, transform_cr3_inter_h = compress_with_zstd(comp_array3_inter_h_flattened)
    print(f"CR transformed3 h: {transform_cr3_inter_h}")
    decomposed_size = 0
    entropy_array = []
    for c in comp_array_3_inter_h:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
        #print(f"entropy of: {compute_entropy(c)}")
        entropy_array.append(compute_entropy(c))
    decomp_cr = len(float_stream.tobytes()) / decomposed_size
    # compute standard deviation of entropy_array
    entropy_array = np.array(entropy_array)
    std_entropy = np.std(entropy_array)
    print(f"Entropy array 3 high: {entropy_array} with std3 inter h: {std_entropy} ")
    print(f"-> CR dec 3 high:  {decomp_cr} \n")


    interleaved_h = np.zeros((len(comp_array_np[0]) * 2), dtype=np.uint8)
    interleaved_h[0::2] = comp_array_np[0]
    interleaved_h[1::2] = comp_array_np[1]
    comp_array_3_inter_h = []
    comp_array_3_inter_h.append(interleaved_h)
    comp_array_3_inter_h.append(comp_array_np[2])
    comp_array_3_inter_h.append(comp_array_np[3])
    comp_array3_inter_h_flattened = np.concatenate(comp_array_3_inter_h, axis=0)
    transformed_data3_inter_h, transform_cr3_inter_h = compress_with_zstd(comp_array3_inter_h_flattened)
    print(f"CR transformed3 inter h: {transform_cr3_inter_h}")
    decomposed_size = 0
    entropy_array = []
    for c in comp_array_3_inter_h:
        decompressed, comp_ratio = compress_with_zstd(c)
        decomposed_size += len(decompressed)
        #print(f"entropy of: {compute_entropy(c)}")
        entropy_array.append(compute_entropy(c))
    decomp_cr = len(float_stream.tobytes()) / decomposed_size
    # compute standard deviation of entropy_array
    entropy_array = np.array(entropy_array)
    std_entropy = np.std(entropy_array)
    print(f"Entropy array 3 inter h: {entropy_array} with std3 inter h: {std_entropy} ")
    print(f"-> CR dec 3 inter h:  {decomp_cr} \n")



exit(1)



