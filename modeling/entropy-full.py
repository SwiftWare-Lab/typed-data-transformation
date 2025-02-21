#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import entropy

def compute_shannon_entropy_8bit(data_bytes):
    """
    Computes the Shannon entropy (base-2) of a block of 8-bit bytes.
    Uses: H(X) = -sum(p_i * log2(p_i)).
    """
    arr = np.asarray(data_bytes, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    unique_vals, counts = np.unique(arr, return_counts=True)
    probs = counts / arr.size
    return -np.sum(probs * np.log2(probs))


def measure_entropy_summary(dataset_path, block_size=65536):
    """
    Reads the dataset from TSV (2nd column is float data),
    partitions into blocks, computes 8-bit entropy for each block,
    returns a dictionary of summary stats for the dataset.

    This function does NOT write any files. It just returns a dict
    you can append to a list for a final CSV.
    """
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Attempt to read the dataset
    try:
        df = pd.read_csv(dataset_path, sep='\t')
        float_data = df.values[:, 1].astype(np.float64)
    except Exception as e:
        # Return an entry indicating failure
        return {
            "dataset": dataset_name,
            "error": f"Failed to load data: {e}"
        }

    # Convert float array to bytes
    data_bytes = float_data.view(np.uint8)
    total_bytes = len(data_bytes)

    # If no data, just return
    if total_bytes == 0:
        return {
            "dataset": dataset_name,
            "block_size": block_size,
            "total_bytes": 0,
            "num_blocks": 0,
            "entropy_avg": np.nan,
            "entropy_std": np.nan,
            "entropy_min": np.nan,
            "block_index_min": -1,
            "entropy_max": np.nan,
            "block_index_max": -1
        }

    # Partition into blocks
    num_full_blocks = total_bytes // block_size
    remainder = total_bytes % block_size
    num_blocks = num_full_blocks + (1 if remainder != 0 else 0)

    # Compute per-block entropies
    block_entropies = []
    start_idx = 0
    for blk_i in range(num_blocks):
        end_idx = min(start_idx + block_size, total_bytes)
        block_data = data_bytes[start_idx:end_idx]
        ent = compute_shannon_entropy_8bit(block_data)
        block_entropies.append(ent)
        start_idx += block_size

    # Summaries
    ent_arr = np.array(block_entropies)
    avg_ent = ent_arr.mean()
    std_ent = ent_arr.std()
    min_ent = ent_arr.min()
    max_ent = ent_arr.max()
    min_ent_idx = int(np.argmin(ent_arr))
    max_ent_idx = int(np.argmax(ent_arr))

    # Return a dict summarizing results
    return {
        "dataset": dataset_name,
        "block_size": block_size,
        "total_bytes": total_bytes,
        "num_blocks": num_blocks,
        "entropy_avg": avg_ent,
        "entropy_std": std_ent,
        "entropy_min": min_ent,
        "block_index_min": min_ent_idx,
        "entropy_max": max_ent,
        "block_index_max": max_ent_idx
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_entropy_folder.py <folder_path> [block_size_bytes]")
        sys.exit(1)

    folder_path = sys.argv[1]

    block_size = 65536
    if len(sys.argv) >= 3:
        block_size = int(sys.argv[2])

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory.")
        sys.exit(1)

    # We'll store all dataset summaries here
    results_list = []

    # Iterate all files in the folder
    all_files = sorted(os.listdir(folder_path))
    for fname in all_files:
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path):
            # We measure entropy summary
            summary_dict = measure_entropy_summary(full_path, block_size=block_size)
            results_list.append(summary_dict)
            # Print a quick line
            ds_name = summary_dict.get("dataset", fname)
            err_msg = summary_dict.get("error", "")
            if err_msg:
                print(f"[FAIL] {ds_name}: {err_msg}")
            else:
                print(f"[OK] {ds_name} => blocks={summary_dict['num_blocks']}, "
                      f"avg={summary_dict['entropy_avg']:.4f} bits, "
                      f"min={summary_dict['entropy_min']:.4f} (blk#{summary_dict['block_index_min']}), "
                      f"max={summary_dict['entropy_max']:.4f} (blk#{summary_dict['block_index_max']})")

    # Convert to DataFrame
    df_final = pd.DataFrame(results_list)

    # Save one CSV with all dataset summaries
    out_csv = "/home/jamalids/Documents/all_entropy_summaries64.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"Saved all summaries to: {out_csv}")


if __name__ == "__main__":
    main()
