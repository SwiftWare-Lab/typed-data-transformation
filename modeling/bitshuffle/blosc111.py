#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import zstandard as zstd
import blosc

def raw_zstd_compress_file(filename, data, level=3):
    """
    Compress the raw data using the zstd library and write the compressed bytes to a file.

    Parameters:
      filename : The output filename (e.g. .bin).
      data     : A NumPy array (e.g. float32).
      level    : Compression level for Zstandard.

    Returns:
      The compressed data as bytes.
    """
    data_bytes = data.tobytes()
    comp_bytes = zstd.compress(data_bytes, level=level)
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes

def blosc_zstd_byteshuffle_compress_file(filename, data, level=3):
    """
    Compress the raw data using Blosc with built-in byte-shuffle (blosc.SHUFFLE) and the zstd backend.
    """
    data_bytes = data.tobytes()
    comp_bytes = blosc.compress(data_bytes,
                                typesize=data.dtype.itemsize,
                                clevel=level,
                                shuffle=blosc.SHUFFLE,
                                cname="zstd")
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes

def process_dataset(tsv_path, out_dir, dtype="float64", target_column=1):
    """
    Process one dataset: read the TSV file, convert the specified column to the given float type,
    and produce two compressed files:
      1. Raw ZSTD
      2. Blosc with Byte Shuffle using ZSTD

    Returns a dictionary with the dataset name, original size, compressed sizes, and ratios.
    """
    dataset_name = os.path.basename(tsv_path).split('.')[0]
    df = pd.read_csv(tsv_path, sep='\t')
    data = df.values[:, target_column].astype(dtype)
    original_size = data.nbytes
    print(f"Dataset: {dataset_name}  | Original data size: {original_size} bytes")

    # 1. Raw ZSTD
    file_raw_zstd = os.path.join(out_dir, f"{dataset_name}_raw_zstd.bin")
    raw_zstd_bytes = raw_zstd_compress_file(file_raw_zstd, data, level=3)
    size_raw_zstd = os.path.getsize(file_raw_zstd)
    ratio_raw_zstd = original_size / size_raw_zstd

    # 2. Blosc-based Byte Shuffle with ZSTD
    file_blosc_zstd_bs = os.path.join(out_dir, f"{dataset_name}_blosc_zstd_byteshuffle.bin")
    blosc_zstd_bs_bytes = blosc_zstd_byteshuffle_compress_file(file_blosc_zstd_bs, data, level=3)
    size_blosc_zstd_bs = os.path.getsize(file_blosc_zstd_bs)
    ratio_blosc_zstd_bs = original_size / size_blosc_zstd_bs

    print(f"  Raw ZSTD:                {size_raw_zstd} bytes  | Ratio: {ratio_raw_zstd:.2f}")
    print(f"  Blosc+ZSTD+ByteShuffle:  {size_blosc_zstd_bs} bytes  | Ratio: {ratio_blosc_zstd_bs:.2f}")

    return {
        "dataset": dataset_name,
        "original_size": original_size,
        "raw_zstd_size": size_raw_zstd,
        "blosc_zstd_byteshuffle_size": size_blosc_zstd_bs,
        "raw_zstd_ratio": ratio_raw_zstd,
        "blosc_zstd_byteshuffle_ratio": ratio_blosc_zstd_bs,
    }

def main():
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64"
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'. Starting...\n")
    dataset_paths = [os.path.join(folder_path, f) for f in tsv_files]

    out_dir = "/home/jamalids/Documents/remaind/h5_datasets"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for path in dataset_paths:
        res = process_dataset(path, out_dir, dtype="float64", target_column=1)
        results.append(res)

    df_results = pd.DataFrame(results)
    csv_output = os.path.join(out_dir, "blosc_zstd_only_results64.csv")
    df_results.to_csv(csv_output, index=False)
    print(f"\nResults saved to {csv_output}")

if __name__ == "__main__":
    main()
