import sys
import os
import numpy as np
import itertools
import pandas as pd

# --------------------------------------------------------------------
# Add your compression and utility functions here (from compression_tools, utils, etc.)
# For example:
from modeling.xor_based import encode_xor_floats
from modeling.utils import tuple_to_string, compute_entropy, list_to_string, find_max_consecutive_similar_values
from modeling.compression_tools import (
    zstd_comp, zlib_comp, bz2_comp, snappy_comp, fastlz_compress,
    rle_compress, huffman_compress,
    blosc_comp, blosc_comp_bit
)

# --------------------------------------------------------------------
 # Example compConfigMap
# -------------------------------------------------------------------------
compConfigMap = {
    "acs_wht_f32": [[[1, 2], [3], [4]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]],[[1, 2,3], [4]]],
    "wave_f32": [[[1, 2], [3], [4]],[[1, 2,3], [4]]],
    "hdr_night_f32": [
        [[1, 4], [2], [3]],

    ],
    "ts_gas_f32": [[[1],[ 2,3], [4]],[[1,4],[ 2],[3]] ],
    "solar_wind_f32": [[[1], [2, 3], [4]]],
    "tpch_lineitem_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_web_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_store_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_catalog_f32": [
        [[1, 2, 3], [4]],

    ],
    "citytemp_f32": [
        [[1, 4], [2, 3]],

    ],
    "hst_wfc3_ir_f32": [
        [[1, 2], [3], [4]],

    ],
    "hst_wfc3_uvis_f32": [
        [[1, 2], [3], [4]],

    ],
    "rsim_f32": [
        [[1, 2],[ 3], [4]],

    ],
    "astro_mhd_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "astro_pt_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "jane_street_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "msg_bt_f64": [
        [[1, 2, 3, 4, 5], [6], [7], [8]],

    ],
    "num_brain_f64": [
        [[3, 2, 4, 5, 1, 6], [7], [8]],

    ],
    "num_control_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "nyc_taxi2015_f64": [
        [[7, 4, 6], [5], [3, 2, 1, 8]],

    ],
    "phone_gyro_f64": [
        [[4, 6], [8], [3, 2, 1, 7], [5]],

    ],
    "tpch_order_f64": [
        [[1, 2, 3, 4], [7], [6, 5], [8]],

    ],
    "tpcxbb_store_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],

    ],
    "tpcxbb_web_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],

    ],
    "wesad_chest_f64": [
        [[7, 5, 6], [8, 4, 1, 3, 2]],

    ],
    "default": [[[1], [2], [3], [4]]],
}

# --------------------------------------------------------------------
# Utility to find all integer compositions (not used if we only want custom)
# --------------------------------------------------------------------
def possible_sum(m):
    """
    Compute all possible sets of positive integers that sum to m.
    Not used if we always provide `given_decomp`.
    """
    possible_sets = []
    for i in range(1, m + 1):
        if i == m:
            possible_sets.append([i])
        else:
            for j in possible_sum(m - i):
                possible_sets.append([i] + j)
    return possible_sets


def merge_order_with_decomposition(order, decomposition):
    """
    Merge a given order (a permutation of 0..m-1) with a composition,
    building sets of consecutive components. Not used if we pass `given_decomp`.
    """
    cur_len = 0
    merged_order = set()
    for i in range(len(decomposition)):
        comp_len = decomposition[i]
        cur_comp = order[cur_len : cur_len + comp_len]
        merged_order.add(cur_comp)
        cur_len += comp_len
    return merged_order


def find_all_combinations(all_possible_consecutive_comp, m, contiguous=True):
    """
    Naive approach for enumerating all permutations of 0..m-1,
    and merging them with each composition. Not used if `given_decomp` is provided.
    """
    byte_loc = np.arange(0, m)
    if contiguous:
        # only one 'permutation' => (0,1,2,...,m-1)
        all_permutations = [tuple(range(0, m))]
    else:
        # get all permutations of length m
        all_permutations = list(itertools.permutations(byte_loc))

    all_decomposition = []
    for composition in all_possible_consecutive_comp:
        for permutation in all_permutations:
            cur_comp = merge_order_with_decomposition(permutation, composition)
            all_decomposition.append(cur_comp)

    # remove duplicates
    all_perm_length = len(all_decomposition)
    all_decomposition = list(set([tuple(x) for x in all_decomposition]))
    return all_decomposition, all_perm_length


# --------------------------------------------------------------------
# Generic compression / transformation utilities
# --------------------------------------------------------------------
def compress_data(data_set_list, compress_method, order='F'):
    """
    Compress each of the subarrays in data_set_list using compress_method.
    Return the total compressed size and the compressed data.
    """
    compressed_data, compressed_size = [], 0
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_comp = compress_method(data_set_bytes)
        compressed_data.append(compressed_comp)
        compressed_size += len(compressed_comp)
    return compressed_data, compressed_size


def transform_data(data_set_list, order='C'):
    """
    Flatten each subarray in data_set_list with given memory order,
    then concatenate them into one big byte array.
    """
    data_out = []
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        data_out.append(data_set_bytes)
    # concatenate
    data_out = np.concatenate(data_out, axis=0)
    return data_out


def analyze_data(data_set_list, data_set_word):
    """
    For each component array in data_set_list, compute:
      - Byte-level entropy
      - Weighted average entropy across them
      - Rep stats
      - Unique ratio
    Return them for logging.
    """
    entropy_list, WE, tot_size = [], 0, 0
    max_rep_lst, uniq_ratio_lst = [], []

    # total size = sum of all sub-arrays in bytes
    for cmp in data_set_list:
        tot_size += len(cmp.flatten().tobytes())

    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten('F').tobytes(), dtype=np.byte)
        # entropy of these bytes
        ent = compute_entropy(data_set_bytes)
        entropy_list.append(ent)

        # find max consecutive repeated values
        max_rep = find_max_consecutive_similar_values(data_set_bytes)
        max_rep_lst.append(max_rep)

        # unique ratio
        uniq_ratio_lst.append(len(set(data_set_bytes)) / len(data_set_bytes))

        # Weighted sum
        WE += ent * (len(data_set_bytes) / tot_size)

    # Also compute entropy of the entire data_set_word
    entropy_word = compute_entropy(data_set_word)

    return entropy_list, WE, entropy_word, max_rep_lst, uniq_ratio_lst


# --------------------------------------------------------------------
# test_decomposition function
# --------------------------------------------------------------------
def test_decomposition(data_set, dataset_name, comp_tool_dict={}, given_decomp=None,
                       m=4, chuck_no=-1, contig_order=True, out_log_dir=''):
    """
    data_set: 1D array of floats
    dataset_name: string for logging
    comp_tool_dict: dictionary { 'tool_name': compress_func, ... }
    given_decomp: pre-defined decompositions (0-based!). If None => fallback enumerations
    m: # of bytes, e.g. 4 for float32, 8 for float64
    contig_order: if True, only use contiguous permutations
    out_log_dir: path for CSV logs
    """

    type_byte = np.uint8

    # If no custom decomposition => fallback enumerations
    if not given_decomp:
        all_possible = possible_sum(m)
        all_4, len_4 = find_all_combinations(all_possible, m, contig_order)
    else:
        all_4, len_4 = given_decomp, len(given_decomp)

    # Convert data_set to bytes
    data_set_bytes = data_set.view(type_byte)
    len_bytes = len(data_set_bytes)

    # 2D array of shape (m, len(data_set)) for easy slicing
    comps = np.zeros((m, len(data_set)), dtype=type_byte)
    for i in range(m):
        comps[i] = data_set_bytes[i:len_bytes:m]

    # We'll store stats for each decomposition
    stat_array = []

    # For every decomposition in all_4
    for idx, decomp in enumerate(all_4):
        # Build initial row
        stats = {
            'dataset name': dataset_name,
            'original size': len_bytes,
            'type width': m,
            'Dimension': len(data_set),
            'decomposition': tuple_to_string(decomp),
            'chunk no': chuck_no
        }

        # Build a list of sub-component arrays for each group
        comp_list = []
        for group_tuple in decomp:
            # e.g. group_tuple might be (0,1)
            group_len = len(group_tuple)
            cur_comp_data = np.zeros((group_len, len(data_set)), dtype=type_byte)
            for idx2, byte_idx in enumerate(group_tuple):
                cur_comp_data[idx2] = comps[byte_idx]
            comp_list.append(cur_comp_data)

        # Reordered data => flatten sub-arrays (col and row order)
        reordered_full_data_col = transform_data(comp_list, order='F')  # Fortran order flatten
        reordered_full_data_row = transform_data(comp_list, order='C')  # C order flatten

        # For each compression tool
        for comp_name, comp_tool in comp_tool_dict.items():
            # Standard single-chunk compress
            _, full_comp_size = compress_data([data_set], comp_tool)

            # Decomposed => compress each sub-component
            _, decomp_compressed_size_col = compress_data(comp_list, comp_tool, order='F')
            _, decomp_compressed_size_row = compress_data(comp_list, comp_tool, order='C')

            # Reordered single-chunk
            _, reordered_compressed_size_col = compress_data([reordered_full_data_col], comp_tool)
            _, reordered_compressed_size_row = compress_data([reordered_full_data_row], comp_tool)

            # Store compressed sizes
            stats[f'standard {comp_name} compressed size (B)'] = full_comp_size
            stats[f'decomposed {comp_name} col-order (B)'] = decomp_compressed_size_col
            stats[f'decomposed {comp_name} row-order (B)'] = decomp_compressed_size_row
            stats[f'reordered {comp_name} col-order (B)'] = reordered_compressed_size_col
            stats[f'reordered {comp_name} row-order (B)'] = reordered_compressed_size_row

            # ---- Compute compression ratios ----
            # Ratio = original_size / compressed_size
            def ratio(orig, comp):
                return float('inf') if comp == 0 else orig / comp

            stats[f'standard {comp_name} ratio'] = ratio(len_bytes, full_comp_size)
            stats[f'decomposed {comp_name} col-order ratio'] = ratio(len_bytes, decomp_compressed_size_col)
            stats[f'decomposed {comp_name} row-order ratio'] = ratio(len_bytes, decomp_compressed_size_row)
            stats[f'reordered {comp_name} col-order ratio'] = ratio(len_bytes, reordered_compressed_size_col)
            stats[f'reordered {comp_name} row-order ratio'] = ratio(len_bytes, reordered_compressed_size_row)

        # Compute entropy, WE, etc.
        entropy_list, WE, entropy_word, rle_max, uniq_ratio = analyze_data(comp_list, data_set)
        stats['WE'] = WE
        stats['entropy word'] = entropy_word
        stats['entropy list'] = list_to_string(entropy_list)
        stats['max rep'] = list_to_string(rle_max)
        stats['unique ratio'] = list_to_string(uniq_ratio)

        stat_array.append(stats)

        # Optionally write partial CSV logs every 20 decompositions
        if out_log_dir != '' and (((idx + 1) % 20 == 0) or ((idx + 1) == len_4)):
            stats_df = pd.DataFrame(stat_array)
            if not os.path.exists(out_log_dir):
                os.makedirs(out_log_dir)
            stats_df.to_csv(f'{out_log_dir}/{dataset_name}_decomposition_stats.csv', index=False)

    return stat_array

import os
import sys
import numpy as np
import pandas as pd

def main():
    # Folder containing the datasets
   # dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64"
    dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    m = 4          # For float32
    chunk_size = -1    # If not -1, we do block-based
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

        if m == 2:
            sliced_data = data_df.values[:, 1].astype(np.float16)
        elif m == 4:
            sliced_data= data_df.values[:, 1].astype(np.float32)
           # sliced_data = sliced_data1[49151:65536]
        else:
            sliced_data = data_df.values[:, 1].astype(np.float64)
           # sliced_data = sliced_data1[49151:65536]

        # ---------------------------------------------------------------
        # 1) Find decompositions from compConfigMap
        # ---------------------------------------------------------------
        if dataset_name in compConfigMap:
            dataset_configs = compConfigMap[dataset_name]
        else:
            dataset_configs = compConfigMap["default"]

        # Convert 1-based => 0-based for each group
        converted_decomps = []
        for grouping in dataset_configs:
            zero_based_groups = []
            for group in grouping:
                zero_based_groups.append(tuple(g - 1 for g in group))  # e.g. [1,2] => (0,1)
            converted_decomps.append(tuple(zero_based_groups))

        # ---------------------------------------------------------------
        # 2) Build compression tool dictionary
        # ---------------------------------------------------------------
        comp_tool_dict = {
           'zstd': zstd_comp,
           'snappy': snappy_comp,
           'blosc': blosc_comp,
           'bz2': bz2_comp,
           'huffman_compress': huffman_compress,
           'fastlz': fastlz_compress,
           'rle': rle_compress,
        }

        # ---------------------------------------------------------------
        # 3) Run either once (chunk_size = -1) or in blocks
        # ---------------------------------------------------------------
        if chunk_size == -1:
            # Single pass
            stats = test_decomposition(
                data_set=sliced_data,
                dataset_name=dataset_name,
                m=m,
                comp_tool_dict=comp_tool_dict,
                given_decomp=converted_decomps,  # Our custom decomposition
                contig_order=contig_order,
                out_log_dir='logs'
            )
            print(f"Done: wrote stats in logs/{dataset_name}_decomposition_stats.csv")
        else:
            # Chunked approach
            no_chunks = len(sliced_data) // chunk_size
           # no_chunks = min(no_chunks, 100)  # if you want a limit
            stats_array = []
            for i in range(no_chunks):
                block_data = sliced_data[i * chunk_size : (i + 1) * chunk_size]
                block_stats = test_decomposition(
                    data_set=block_data,
                    dataset_name=dataset_name,
                    m=m,
                    comp_tool_dict=comp_tool_dict,
                    chuck_no=i,
                    given_decomp=converted_decomps,
                    contig_order=contig_order,
                    out_log_dir=''  # or 'logs' if you want partial logs each iteration
                )
                stats_array.extend(block_stats)

            # Save final CSV
            stats_df = pd.DataFrame(stats_array)
            if not os.path.exists('logs'):
                os.makedirs('logs')
            stats_df.to_csv(f'logs/{dataset_name}_decomposition_streaming_stats.csv', index=False)
            print(f"Done: wrote chunk-based stats to logs/{dataset_name}_decomposition_streaming_stats.csv")


if __name__ == "__main__":
    main()
