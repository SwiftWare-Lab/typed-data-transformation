#!/usr/bin/env python3
import os
import sys
import numpy as np
import itertools
import pandas as pd

# Adjust these imports to match your project structure:
from modeling.xor_based import encode_xor_floats
from modeling.utils import tuple_to_string, compute_entropy, list_to_string, find_max_consecutive_similar_values
from modeling.compression_tools import (
    zstd_comp, zlib_comp, bz2_comp, snappy_comp,
    fastlz_compress, rle_compress, huffman_compress,
    blosc_comp, blosc_comp_bit, blosc_comp_noshuff
)

########################################
# Predefined decompositions with 1-based indices
########################################
dataset_configs = {
    # ===========================
    # Float32 datasets
    # ===========================
    "acs_wht_f32": [
        [[1,2], [3], [4]],
        [[1], [2], [3], [4]]
    ],
    "citytemp_f32": [
        [[1,2], [3], [4]]
    ],
    "hdr_night_f32": [
        [[1,2], [3], [4]]
    ],
    "hdr_palermo_f32": [
        [[1,2], [3], [4]]
    ],
    "hst_wfc3_ir_f32": [
        [[1,2], [3], [4]]
    ],
    "hst_wfc3_uvis_f32": [
        [[1,2,3], [4]]
    ],
    "jw_mirimage_f32": [
        # e.g. [[1], [2], [3], [4]] commented out in your example
        [[1,2,3], [4]]
    ],
    "rsim_f32": [
        [[1,2,3], [4]]
    ],
    "solar_wind_f32": [
        [[1,2], [3], [4]]
    ],
    "spitzer_irac_f32": [
        [[1,2,3], [4]]
    ],
    "tpcds_catalog_f32": [
        [[1], [2], [3], [4]]
    ],
    "tpcds_store_f32": [
        [[1,2], [3], [4]]
    ],
    "tpcds_web_f32": [
        [[1], [2], [3], [4]]
    ],
    "tpch_lineitem_f32": [
        [[1,2,3], [4]]
    ],
    # wave_f32 with two possible solutions (the first uncommented, second commented out in your sample)
    "wave_f32": [
        [[1,2,3], [4]],
        # [[1,2], [3], [4]]
    ],

    # ===========================
    # Float64 datasets
    # ===========================
    # astro_mhd_f64 has a first solution, second commented out in your example
    "astro_mhd_f64": [
        [[1,2,3,4,5,6], [7,8]],
        # [[1,3,4], [5], [2], [6], [7], [8]]
    ],
    "tpch_order_f64": [
        [[5,6], [1,2,3,4], [7], [8]],
        # [[5], [6], [1,2,3], [4], [7], [8]]
    ],
    "astro_pt_f64": [
        # [[1,2,3,4,5,6,7], [8]],
        [[4,5], [1,2,3], [6], [7], [8]]
    ],
    "wesad_chest_f64": [
        [[1,2,3,4,8], [5,6,7]],
        # [[2,3], {4}, {8}, {1}, {5}, {7}, {6}]
    ],
    "phone_gyro_f64": [
        [[1,2,3,4,5,6,7], [8]],
        # [[2,3], [1], [7], [4], [6], [5], [8]]
    ],
    "tpcxbb_store_f64": [
        [[6,7], [1,2,3,4,5], [8]],
        # [[6], [7], [1,2,4], [3], [5], [8]]
    ],
    # num_brain_f64 has two sets
    "num_brain_f64": [
        [[2,5], [1,3,4], [6], [7], [8]],
        [[1,2], [3], [6], [4], [5], [7], [8]]
    ],
    "msg_bt_f64": [
        # [[1,2,3,4,5,6,7], [8]],
        [[2,3], [1], [4], [5], [6], [7], [8]]
    ],
    "tpcxbb_web_f64": [
        [[6,7], [1,2,3,4,5], [8]],
        # [[6], [7], [2,3,4], [1], [5], [8]]
    ],
    "nyc_taxi2015_f64": [
        [[1,2,3,8], [4,5,6,7]],
        # [[2,3], [1], [8], [4], [7], [6], [5]]
    ],

    # ===========================
    # Default entry
    # ===========================
    "default": [
        [[1], [2], [3], [4]]
    ],
}


#########################
# Utility functions
#########################

def possible_sum(m):
    possible_sets = []
    for i in range(1, m+1):
        if i == m:
            possible_sets.append([i])
        else:
            for j in possible_sum(m-i):
                possible_sets.append([i] + j)
    return possible_sets

def find_all_combinations(all_possible_consecutive_comp, m, contiguous=True):
    byte_loc = np.arange(0, m)
    if contiguous:
        all_permutations = [tuple(range(0, m))]
    else:
        all_permutations = list(itertools.permutations(byte_loc))
    all_decomposition = []
    for composition in all_possible_consecutive_comp:
        for permutation in all_permutations:
            cur_comp = merge_order_with_decomposition(permutation, composition)
            all_decomposition.append(cur_comp)
    all_perm_length = len(all_decomposition)
    all_decomposition = list(set([tuple(x) for x in all_decomposition]))
    return all_decomposition, all_perm_length

def merge_order_with_decomposition(order, decomposition):
    # Merges a permutation of positions (e.g. [0,1,2,3]) with a composition like [2,2].
    # For instance, if order=[0,1,2,3] and decomposition=[2,2],
    # the first chunk is order[:2], the second chunk is order[2:4].
    cur_len = 0
    merged_order = set()
    for comp_len in decomposition:
        cur_comp = order[cur_len:cur_len+comp_len]
        merged_order.add(cur_comp)
        cur_len += comp_len
    return merged_order

def compress_data(data_set_list, compress_method, order='F'):
    compressed_data, compressed_size = [], 0
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_comp = compress_method(data_set_bytes)
        compressed_data.append(compressed_comp)
        compressed_size += len(compressed_comp)
    return compressed_data, compressed_size

def transform_data(data_set_list, order='C'):
    compressed_data = []
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_data.append(data_set_bytes)
    compressed_data = np.concatenate(compressed_data, axis=0)
    return compressed_data

def analyze_data(data_set_list, data_set_word):
    from modeling.utils import compute_entropy, list_to_string, find_max_consecutive_similar_values
    entropy_list, WE, tot_size, entropy_word = [], 0, 0, 0
    max_rep_lst, uniq_ratio_lst = [], []
    for cmp in data_set_list:
        tot_size += len(cmp.flatten().tobytes())
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten('F').tobytes(), dtype=np.byte)
        entropy = compute_entropy(data_set_bytes)
        entropy_list.append(entropy)
        max_rep = find_max_consecutive_similar_values(data_set_bytes)
        max_rep_lst.append(max_rep)
        uniq_ratio_lst.append(len(set(data_set_bytes))/len(data_set_bytes))
        WE += entropy * (len(data_set_bytes) / tot_size)
    entropy_word = compute_entropy(data_set_word)
    return entropy_list, WE, entropy_word, max_rep_lst, uniq_ratio_lst

#########################
# The main decomposition function
#########################
def test_decomposition(data_set, dataset_name, comp_tool_dict={},
                       given_decomp=None, m=8, chuck_no=-1,
                       contig_order=True, out_log_dir=''):
    type_byte = np.uint8

    # If given_decomp is None, we revert to the original approach
    # Otherwise we skip find_all_combinations
    if not given_decomp:
        from modeling.decompose_code import possible_sum, find_all_combinations
        all_4, len_4 = find_all_combinations(possible_sum(m), m, contig_order)
    else:
        all_4, len_4 = given_decomp  # expecting something like (myconfigs, len(myconfigs))

    data_set_bytes = data_set.view(type_byte)
    len_bytes = len(data_set_bytes)
    comps = np.zeros((m, len(data_set)), dtype=type_byte)
    for i in range(m):
        comps[i] = data_set_bytes[i:len_bytes:m]

    stat_array = []
    for idx, decomp in enumerate(all_4):
        stats = {
            'dataset name': dataset_name,
            'original size': len_bytes,
            'type width': m,
            'Dimension': len(data_set),
            'decomposition': tuple_to_string(decomp),
            'chunk no': chuck_no
        }

        # Build the 'comp_list'
        comp_list = []
        for cur_comp in decomp:
            # cur_comp is a list of 1-based indices e.g. [1,2]
            # We create an array with shape (len(cur_comp), len(data_set))
            cur_comp_data = np.zeros((len(cur_comp), len(data_set)), dtype=type_byte)
            for j in range(len(cur_comp)):
                # shift by 1 to convert from 1-based to 0-based
                zero_based_idx = cur_comp[j] - 1
                # e.g. comps[0..3]
                cur_comp_data[j] = comps[zero_based_idx]
            comp_list.append(cur_comp_data)

        # transform data for row-based / reordered, etc.
        reordered_full_data_row_based = transform_data(comp_list, order='C')
        reordered_full_data = transform_data(comp_list, order='F')

        # For each compression tool
        for comp_name, comp_tool in comp_tool_dict.items():
            # standard compression on entire data
            full_compressed, full_comp_size = compress_data([data_set], comp_tool)

            # decomposed
            c_data, decomp_compressed_size = compress_data(comp_list, comp_tool)
            c_data_row_based, decomp_compressed_size_row_based = compress_data(comp_list, comp_tool, order='C')

            # reordered
            c_reordered_date, reordered_compressed_size = compress_data([reordered_full_data], comp_tool)
            c_data_row, reordered_compressed_size_row_based = compress_data([reordered_full_data_row_based], comp_tool)

            stats[f'decomposed {comp_name} compressed size (B)'] = decomp_compressed_size
            stats[f'decomposed row-ordered {comp_name} compressed size (B)'] = decomp_compressed_size_row_based
            stats[f'reordered {comp_name} compressed size (B)'] = reordered_compressed_size
            stats[f'reordered row-based {comp_name} compressed size (B)'] = reordered_compressed_size_row_based
            stats[f'standard {comp_name} compressed size (B)'] = full_comp_size

            stats[f'standard {comp_name} compression ratio'] = len_bytes / full_comp_size
            stats[f'decomposed {comp_name} compression ratio'] = len_bytes / decomp_compressed_size
            stats[f'decomposed row-ordered {comp_name} compression ratio'] = len_bytes / decomp_compressed_size_row_based
            stats[f'reordered {comp_name} compression ratio'] = len_bytes / reordered_compressed_size
            stats[f'reordered row-based {comp_name} compression ratio'] = len_bytes / reordered_compressed_size_row_based

            print(f"decomp: {tuple_to_string(decomp)} : {comp_name} => standard ratio: {len_bytes/full_comp_size:.2f}, decomposed ratio: {len_bytes/decomp_compressed_size:.2f}")

        # analyze entropy
        entropy_list, WE, entropy_word, rle_max, uniq_ratio = analyze_data(comp_list, data_set)
        stats['WE'] = WE
        stats['entropy word'] = entropy_word
        stats['entropy list'] = list_to_string(entropy_list)
        stats['max rep'] = list_to_string(rle_max)
        stats['unique ratio'] = list_to_string(uniq_ratio)
        stat_array.append(stats)

        if out_log_dir != '' and (((idx+1) % 20 == 0) or ((idx+1) == len(all_4))):
            stats_df = pd.DataFrame(stat_array)
            if not os.path.exists(out_log_dir):
                os.makedirs(out_log_dir)
            stats_df.to_csv(f'{out_log_dir}/{dataset_name}_decomposition_stats.csv', index=False)

    return stat_array


############################
# Main
############################
def main():
    dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64"
    m = 8
    contig_order = False

    comp_tool_dict = {
        #"zstd": zstd_comp,
        #'blosc': blosc_comp,
        #'bosc-bit':blosc_comp_bit,
        #'blosc-Noshuff': blosc_comp_noshuff,
       # 'rle': rle_compress,
        'xor': encode_xor_floats,
        # only the ones you want to test
    }

    if not os.path.isdir(dataset_folder):
        print(f"Error: {dataset_folder} is not a valid directory.")
        return

    # Sorted list of .tsv files
    file_list = sorted([f for f in os.listdir(dataset_folder) if f.lower().endswith(".tsv")])
    if not file_list:
        print(f"No TSV files in {dataset_folder}")
        return

    for file_name in file_list:
        dataset_path = os.path.join(dataset_folder, file_name)
        dataset_name = os.path.splitext(file_name)[0]
        print(f"Processing {dataset_name}...")

        df = pd.read_csv(dataset_path, sep='\t')
        # we do float32 for m=4
        data_array = df.values[:, 1].astype(np.float64)
        # Check if dataset_name is in our dataset_configs
        if dataset_name in dataset_configs:
            my_configs = dataset_configs[dataset_name]
        else:
            my_configs = dataset_configs["default"]

        # We pass (my_configs, len(my_configs)) so test_decomposition uses them directly
        test_decomposition(
            data_array,
            dataset_name,
            comp_tool_dict=comp_tool_dict,
            given_decomp=(my_configs, len(my_configs)),
            m=m,
            contig_order=contig_order,
            out_log_dir="/home/jamalids/Documents/logs1"
        )

if __name__ == "__main__":
    main()
