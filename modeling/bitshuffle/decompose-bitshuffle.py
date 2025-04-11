#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
import zstandard as zstd
import bitshuffle.h5

# --- For raw compression using libraries directly ---
import lz4.frame


# ----------------- Original Bitshuffle HDF5 Functions -----------------

def create_hdf5_bitshuffle(filename, data, compressor_opts, dataset_name="data"):
    """
    Create an HDF5 file with a dataset that uses the Bitshuffle HDF5 filter.

    Parameters:
      filename        : Output HDF5 file name.
      data            : A NumPy array that will be stored (e.g., float32).
      compressor_opts : Compression options tuple.
                        For example:
                          - For ZSTD: (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_ZSTD, COMP_LEVEL)
                          - For LZ4:  (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_LZ4)
                        Use BLOCK_SIZE = 0 to let Bitshuffle choose the optimal block size.
      dataset_name    : The name of the dataset to create (default "data").

    Returns:
      The data array that was saved.
    """
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(
            dataset_name,
            data=data,
            compression=bitshuffle.h5.H5FILTER,
            compression_opts=compressor_opts
        )
    return data


def read_hdf5_bitshuffle(filename, dataset_name="data"):
    """
    Read and return the dataset from an HDF5 file created with the Bitshuffle filter.
    """
    with h5py.File(filename, "r") as f:
        data = f[dataset_name][...]
    return data


# ----------------- Added: Raw Compression Functions -----------------

def raw_zstd_compress_file(filename, data, level=3):
    """
    Compress the raw data using the zstd library and write the compressed bytes to a file.

    Parameters:
      filename : The output filename (e.g., ending in .bin).
      data     : A NumPy array (e.g., float32).
      level    : Compression level for Zstandard.

    Returns:
      The compressed data as bytes.
    """
    data_bytes = data.tobytes()
    comp_bytes = zstd.compress(data_bytes, level=level)
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes


def raw_lz4_compress_file(filename, data):
    """
    Compress the raw data using the lz4 library and write the compressed bytes to a file.

    Parameters:
      filename : The output filename (e.g., ending in .bin).
      data     : A NumPy array (e.g., float32).

    Returns:
      The compressed data as bytes.
    """
    data_bytes = data.tobytes()
    comp_bytes = lz4.frame.compress(data_bytes)
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes


# ----------------- Raw HDF5 Compression Using h5py is Removed -----------------
# (We now compress raw data using the libraries directly, not via HDF5.)

# ----------------- Original Process Dataset Function (Bitshuffle-based) -----------------

def process_dataset(tsv_path, out_dir, dtype="float32", target_column=1):
    """
    Process one dataset: read the TSV file, convert the desired column to the specified float type,
    and write HDF5 files using Bitshuffle compression as well as raw compressed files using the zstd
    and lz4 libraries directly.

    This function creates four files:
      1. Bitshuffle+ZSTD HDF5 file.
      2. Bitshuffle+LZ4 HDF5 file.
      3. Raw ZSTD compressed binary file.
      4. Raw LZ4 compressed binary file.

    Returns a dictionary with the dataset name, original size in bytes, file sizes, and the corresponding
    compression ratios.
    """
    dataset_name = os.path.basename(tsv_path).split('.')[0]
    # Read dataset using pandas.
    data_set = pd.read_csv(tsv_path, sep='\t')
    # Convert the chosen column to the target float type.
    data = data_set.values[:, target_column].astype(dtype)

    # Get the original data size in bytes.
    original_size = data.nbytes
    print(f"Dataset: {dataset_name}  | Original data size: {original_size} bytes")

    # Set Bitshuffle parameters.
    BLOCK_SIZE = 0  # Let Bitshuffle choose an optimal block size.
    COMP_LEVEL = 3  # Compression level for ZSTD (for Bitshuffle mode).

    # Set filenames for Bitshuffle-compressed HDF5 files.
    file_zstd = os.path.join(out_dir, f"{dataset_name}_bitshuffle_zstd.h5")
    file_lz4 = os.path.join(out_dir, f"{dataset_name}_bitshuffle_lz4.h5")

    # Create HDF5 files using Bitshuffle.
    compressor_opts_zstd = (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_ZSTD, COMP_LEVEL)
    create_hdf5_bitshuffle(file_zstd, data, compressor_opts_zstd)
    size_zstd = os.path.getsize(file_zstd)

    compressor_opts_lz4 = (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_LZ4)
    create_hdf5_bitshuffle(file_lz4, data, compressor_opts_lz4)
    size_lz4 = os.path.getsize(file_lz4)

    # --- Added: Create raw compressed files using zstd and lz4 libraries directly ---
    file_raw_zstd = os.path.join(out_dir, f"{dataset_name}_raw_zstd.bin")
    file_raw_lz4 = os.path.join(out_dir, f"{dataset_name}_raw_lz4.bin")

    raw_zstd_bytes = raw_zstd_compress_file(file_raw_zstd, data, level=3)
    size_raw_zstd = os.path.getsize(file_raw_zstd)

    raw_lz4_bytes = raw_lz4_compress_file(file_raw_lz4, data)
    size_raw_lz4 = os.path.getsize(file_raw_lz4)

    # Calculate compression ratios.
    ratio_bitshuffle_zstd = original_size / size_zstd
    ratio_bitshuffle_lz4 = original_size / size_lz4
    ratio_raw_zstd = original_size / size_raw_zstd
    ratio_raw_lz4 = original_size / size_raw_lz4

    print(f"  Bitshuffle+ZSTD: {size_zstd} bytes  | Ratio: {ratio_bitshuffle_zstd:.2f}")
    print(f"  Bitshuffle+LZ4:  {size_lz4} bytes  | Ratio: {ratio_bitshuffle_lz4:.2f}")
    print(f"  Raw ZSTD:        {size_raw_zstd} bytes  | Ratio: {ratio_raw_zstd:.2f}")
    print(f"  Raw LZ4:         {size_raw_lz4} bytes  | Ratio: {ratio_raw_lz4:.2f}")

    return {
        "dataset": dataset_name,
        "original_size": original_size,
        "bitshuffle_zstd_size": size_zstd,
        "bitshuffle_lz4_size": size_lz4,
        "raw_zstd_size": size_raw_zstd,
        "raw_lz4_size": size_raw_lz4,
        "bitshuffle_zstd_ratio": ratio_bitshuffle_zstd,
        "bitshuffle_lz4_ratio": ratio_bitshuffle_lz4,
        "raw_zstd_ratio": ratio_raw_zstd,
        "raw_lz4_ratio": ratio_raw_lz4,
    }


def main():
    # List of TSV dataset file paths.
    dataset_paths = [
        "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/hst_wfc3_ir_f32.tsv",
        # Add more dataset paths here...
    ]

    # Output directory for HDF5 files and raw compressed files.
    out_dir = "/home/jamalids/Documents/remaind/h5_datasets"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for path in dataset_paths:
        res = process_dataset(path, out_dir, dtype="float32", target_column=1)
        results.append(res)

    results_df = pd.DataFrame(results)
    csv_output = os.path.join(out_dir, "datasets_compression_stats.csv")
    results_df.to_csv(csv_output, index=False)
    print(f"\nResults saved to {csv_output}")


if __name__ == "__main__":
    main()
