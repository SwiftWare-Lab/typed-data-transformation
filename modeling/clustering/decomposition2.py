import sys
import os
import numpy as np
import itertools
import pandas as pd
from modeling.xor_based import encode_xor_floats
from modeling.utils import tuple_to_string, compute_entropy, list_to_string, find_max_consecutive_similar_values
from modeling.compression_tools import (
    zstd_comp, zlib_comp, bz2_comp, snappy_comp, fastlz_compress,
    rle_compress, huffman_compress,
    blosc_comp, blosc_comp_bit
)

def possible_sum(m):
    ### compute all possible set of integers that sum to m
    possible_sets = []
    for i in range(1, m+1):
        if i == m:
            possible_sets.append([i])
        else:
            for j in possible_sum(m-i):
                possible_sets.append([i] + j)
    return possible_sets


def find_uniqe_sets(possible_sets):
    # sort all sets
    possible_sets_sorted = [sorted(x) for x in possible_sets]
    # remove duplicates
    unique_sets = list(set([tuple(x) for x in possible_sets_sorted]))
    return unique_sets


def merge_order_with_decomposition(order, decomposition):
    # merge order with decomposition
    cur_len = 0
    # empty set
    merged_order = set()
    for i in range(len(decomposition)):
        comp_len = decomposition[i]
        cur_comp = order[cur_len:cur_len+comp_len]
        merged_order.add(cur_comp)
        cur_len += comp_len
    return merged_order


def find_all_combinations(all_possible_consecutive_comp, m, contiguous=True):
    # find all combinations of decomposition in a naive way
    byte_loc = np.arange(0, m)
    if contiguous:
        # make a tuple of size m
        all_permutations = [tuple(range(0, m))]
    else:
        # get all m! permutations of byte_loc
        all_permutations = list(itertools.permutations(byte_loc))
    # for every composition, apply all permutations
    all_decomposition = []
    for composition in all_possible_consecutive_comp:
        for permutation in all_permutations:
            cur_comp = merge_order_with_decomposition(permutation, composition)
            all_decomposition.append(cur_comp)
    # remove duplicates
    all_perm_length = len(all_decomposition)
    all_decomposition = list(set([tuple(x) for x in all_decomposition]))
    return all_decomposition, all_perm_length


def compress_data(data_set_list, compress_method, order='F'):
    # view data_set as bytes
    compressed_data, compressed_size = [], 0
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_comp = compress_method(data_set_bytes)
        compressed_data.append(compressed_comp)
        compressed_size += len(compressed_comp)
    return compressed_data, compressed_size


def transform_data(data_set_list, order='C'):
    # view data_set as bytes
    compressed_data = []
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_data.append(data_set_bytes)
    # flatten the list
    compressed_data = np.concatenate(compressed_data, axis=0)
    return compressed_data


def analyze_data(data_set_list, data_set_word):
    # view data_set as bytes
    entropy_list, WE, tot_size, entropy_word = [], 0, 0, 0
    max_rep_lst, uniq_ratio_lst = [], []
    for cmp in data_set_list:
        tot_size += len(cmp.flatten().tobytes())
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten('F').tobytes(), dtype=np.byte)
        # entropy of the compressed data
        entropy = compute_entropy(data_set_bytes)
        entropy_list.append(entropy)
        max_rep = find_max_consecutive_similar_values(data_set_bytes)
        max_rep_lst.append(max_rep)
        uniq_ratio_lst.append(len(set(data_set_bytes))/len(data_set_bytes))
        WE += entropy * (len(data_set_bytes) / tot_size)
    entropy_word = compute_entropy(data_set_word)
    return entropy_list, WE, entropy_word, max_rep_lst, uniq_ratio_lst


def test_decomposition(data_set, dataset_name, comp_tool_dict={}, given_decomp=None, m=4, chuck_no=-1, contig_order=True, out_log_dir=''):
    type_byte = np.uint8
    if not given_decomp:
        all_4, len_4 = find_all_combinations(possible_sum(m), m, contig_order)
    else:
        all_4, len_4 = given_decomp, len(given_decomp)
    # view data_set as bytes
    data_set_bytes = data_set.view(type_byte)
    len_bytes = len(data_set_bytes)
    # 2D byte array of m x len(data_set) bytes for easy access
    comps = np.zeros((m, len(data_set)), dtype=type_byte)
    for i in range(m):
        comps[i] = data_set_bytes[i:len_bytes:m]
    # create new decomposed data set
    stat_array = []
    for idx, decomp in enumerate(all_4):
        stats = {'dataset name': dataset_name, 'original size': len_bytes, 'type width': m,
                 'Dimension': len(data_set), 'decomposition': tuple_to_string(decomp), 'chunk no': chuck_no}
        comp_list, comp_list_bytes = [], []
        for cur_comp in decomp:
            cur_comp_data = np.zeros((len(cur_comp), len(data_set)), dtype=type_byte)
            for i in range(len(cur_comp)):
                cur_comp_data[i] = comps[cur_comp[i]]
            comp_list.append(cur_comp_data)
        reordered_full_data_row_based = transform_data(comp_list, order='C')
        reordered_full_data = transform_data(comp_list, order='F')

        # for every compression tool, compress the data
        for comp_name, comp_tool in comp_tool_dict.items():
            full_compressed, full_comp_size = compress_data([data_set], comp_tool)

            c_data, decomp_compressed_size = compress_data(comp_list, comp_tool)
            c_data_row_based, decomp_compressed_size_row_based = compress_data(comp_list, comp_tool, order='C')

            c_reordered_date, reordered_compressed_size = compress_data([reordered_full_data], comp_tool)
            c_data_row, reordered_compressed_size_row_based = compress_data([reordered_full_data_row_based], comp_tool)

            stats[f'decomposed {comp_name} compressed size (B)'] = decomp_compressed_size
            stats[f'decomposed row-ordered {comp_name} compressed size (B)'] = decomp_compressed_size_row_based

            stats[f'reordered {comp_name} compressed size (B)'] = reordered_compressed_size
            stats[f'reordered row-based {comp_name} compressed size (B)'] = reordered_compressed_size_row_based

            stats[f'standard {comp_name} compressed size (B)'] = full_comp_size
            # Add compression ratios to the stats
            stats[f'standard {comp_name} compression ratio'] = len_bytes / full_comp_size
            stats[f'decomposed {comp_name} compression ratio'] = len_bytes / decomp_compressed_size
            stats[
                f'decomposed row-ordered {comp_name} compression ratio'] = len_bytes / decomp_compressed_size_row_based
            stats[f'reordered {comp_name} compression ratio'] = len_bytes / reordered_compressed_size
            stats[
                f'reordered row-based {comp_name} compression ratio'] = len_bytes / reordered_compressed_size_row_based

            print(f'decomp: {tuple_to_string(decomp)} : {comp_name} compression ratio: {len_bytes/full_comp_size}, decomposed compression ratio: {len_bytes/decomp_compressed_size},'
                  f' decomposed row-based compression ratio: {len_bytes/decomp_compressed_size_row_based}, reordered compression ratio: {len_bytes/reordered_compressed_size}, '
                  f' reordered row-based compression ratio: {len_bytes/reordered_compressed_size_row_based}')
        # calculate entropy
        entropy_list, WE, entropy_word, rle_max, uniq_ratio = analyze_data(comp_list, data_set)
        stats['WE'] = WE
        stats['entropy word'] = entropy_word
        stats['entropy list'] = list_to_string(entropy_list)
        stats['max rep'] = list_to_string(rle_max)
        stats['unique ratio'] = list_to_string(uniq_ratio)
        stat_array.append(stats)
        if out_log_dir != '' and (((idx+1) % 20 == 0) or ((idx+1) == len(all_4))):
            # store stats in a csv file
            stats_df = pd.DataFrame(stat_array)
            if not os.path.exists(out_log_dir):
                os.makedirs(out_log_dir)
            stats_df.to_csv(f'{out_log_dir}/{dataset_name}_decomposition_stats.csv', index=False)
    return stat_array


# use argparser to get the dataset path
def main():
    # Folder containing the datasets
    dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
    m = 4  # For float32
    # chunk_size = 65536
    chunk_size = -1  # If not -1, we do block-based
    contig_order = False

    # Ensure folder exists
    if not os.path.isdir(dataset_folder):
        print(f"Error: {dataset_folder} is not a valid directory.")
        return

    # Process each `.tsv` file in the folder
    for file_name in sorted(os.listdir(dataset_folder)):
        dataset_path = os.path.join(dataset_folder, file_name)
        if not os.path.isfile(dataset_path) or not dataset_path.endswith(".tsv"):
            continue  # Skip non-files and non-TSV files

        # Extract dataset name from path
        dataset_name = os.path.splitext(file_name)[0]
        print("Processing Dataset:", dataset_name)

        # Load the data from TSV
        data_df = pd.read_csv(dataset_path, sep='\t')

        # Cast to the appropriate float type
        if m == 2:
            sliced_data = data_df.values[:, 1].astype(np.float16)
        elif m == 4:
            sliced_data = data_df.values[:, 1].astype(np.float32)

        else:
            sliced_data = data_df.values[:, 1].astype(np.float64)

        if chunk_size == -1:
            comp_tool_dict = {
                # 'huffman_compress': huffman_compress,
                 #'zstd': zstd_comp,
                # 'zlib': zlib_comp,
                 #'bz2': bz2_comp,
                # 'snappy': snappy_comp,
                 #'fastlz': fastlz_compress,
                # 'rle': rle_compress,
               'blosc': blosc_comp,
                #'blosc_bit': blosc_comp_bit,  # Added blosc from compression_tools
            }
            stats = test_decomposition(
                sliced_data,
                dataset_name,
                m=m,
                comp_tool_dict=comp_tool_dict,
                contig_order=contig_order,
                out_log_dir='logs'
            )
        else:
            comp_tool_dict = {
                'zstd': zstd_comp,
                'zlib': zlib_comp,
                'bz2': bz2_comp,
                'snappy': snappy_comp,
                'xor': encode_xor_floats,
                'blosc': blosc_comp  # Added blosc from compression_tools
            }
            no_chunks = len(sliced_data) // chunk_size
            no_chunks = np.min([100, no_chunks])
            stats_array = []
            for i in range(no_chunks):
                stats = test_decomposition(
                    sliced_data[i * chunk_size:(i + 1) * chunk_size],
                    dataset_name,
                    m=m,
                    comp_tool_dict=comp_tool_dict,
                    chuck_no=i,
                    contig_order=contig_order
                )
                stats_array.extend(stats)
            # Store stats in a CSV file
            stats_df = pd.DataFrame(stats_array)
            if not os.path.exists('logs'):
                os.makedirs('logs')
            stats_df.to_csv(f'logs-snappy/{dataset_name}_decomposition_streaming_stats.csv', index=False)
if __name__ == "__main__":
    main()
