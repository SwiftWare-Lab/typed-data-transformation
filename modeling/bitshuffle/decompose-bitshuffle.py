#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
import zstandard as zstd
import bitshuffle.h5

# --- For raw compression using libraries directly ---
# We use lz4.block to allow passing a compression level parameter (similar to your C++ code).
import lz4.block

#############################################
# Original Bitshuffle HDF5 Functions
#############################################

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

#############################################
# Raw Compression Functions (Without Preconditioning)
#############################################

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

# --- Modified Raw LZ4 Compression Function ---
def raw_lz4_compress_file(filename, data, compressionLevel=3):
    """
    Compress the raw data using the lz4.block API and write the compressed bytes to a file.

    If compressionLevel > 0, use high-compression mode with that level;
    otherwise use the default (fast) compression mode.
    """
    data_bytes = data.tobytes()
    if compressionLevel > 0:
        comp_bytes = lz4.block.compress(data_bytes, mode="high_compression", compression=compressionLevel)
    else:
        comp_bytes = lz4.block.compress(data_bytes, mode="default")
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes

#############################################
# Added: Byte Shuffle Functions
#############################################

def byte_shuffle(data_bytes, element_size):
    """
    Apply a byte-level shuffle to the input bytes.

    The function reshapes the raw byte stream into a 2D array of shape
    (num_elements, element_size), transposes it (so that the nth byte of every
    element is contiguous), and then flattens it back.

    Parameters:
      data_bytes   : A bytes object.
      element_size : The number of bytes per element (e.g., 4 for float32, 8 for float64).

    Returns:
      A NumPy array of type uint8 containing the shuffled bytes.
    """
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    if arr.size % element_size != 0:
        raise ValueError("Data size must be a multiple of the element size.")
    num_elements = arr.size // element_size
    reshaped = arr.reshape((num_elements, element_size))
    transposed = reshaped.transpose()
    return transposed.flatten()

def raw_zstd_byteshuffle_compress_file(filename, data, level=3):
    """
    Apply byte shuffle preconditioning to the raw data and compress using zstd.
    """
    data_bytes = data.tobytes()
    shuffled = byte_shuffle(data_bytes, data.dtype.itemsize)
    comp_bytes = zstd.compress(shuffled.tobytes(), level=level)
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes

# --- Modified Raw LZ4 Byte Shuffle Compression Function ---
def raw_lz4_byteshuffle_compress_file(filename, data, compressionLevel=3):
    """
    Apply byte shuffle preconditioning to the raw data and compress using the lz4.block API.

    If compressionLevel > 0, use high-compression mode with that level.
    """
    data_bytes = data.tobytes()
    shuffled = byte_shuffle(data_bytes, data.dtype.itemsize)
    if compressionLevel > 0:
        comp_bytes = lz4.block.compress(shuffled.tobytes(), mode="high_compression", compression=compressionLevel)
    else:
        comp_bytes = lz4.block.compress(shuffled.tobytes(), mode="default")
    with open(filename, "wb") as f:
        f.write(comp_bytes)
    return comp_bytes

#############################################
# Process Dataset Function (Bitshuffle-based with added raw functions)
#############################################

def process_dataset(tsv_path, out_dir, dtype="float32", target_column=1):
    """
    Process one dataset: read the TSV file, convert the desired column to the specified float type,
    and write HDF5 files using Bitshuffle compression as well as raw compressed files using
    the zstd and lz4 libraries directly—both with and without byte shuffle preconditioning.

    This function creates six files:
      1. Bitshuffle+ZSTD HDF5 file.
      2. Bitshuffle+LZ4 HDF5 file.
      3. Raw ZSTD compressed binary file.
      4. Raw LZ4 compressed binary file.
      5. Raw ZSTD+ByteShuffle compressed binary file.
      6. Raw LZ4+ByteShuffle compressed binary file.
    """
    dataset_name = os.path.basename(tsv_path).split('.')[0]
    # Read dataset using pandas.
    data_set = pd.read_csv(tsv_path, sep='\t')
    data = data_set.values[:, 1].astype(dtype)
    original_size = data.nbytes
    print(f"Dataset: {dataset_name}  | Original data size: {original_size} bytes")

    # Bitshuffle parameters.
    BLOCK_SIZE = 0   # Let Bitshuffle choose an optimal block size.
    COMP_LEVEL = 3   # Compression level for ZSTD (for Bitshuffle mode).

    # Filenames for Bitshuffle-based HDF5 files.
    file_zstd = os.path.join(out_dir, f"{dataset_name}_bitshuffle_zstd.h5")
    file_lz4 = os.path.join(out_dir, f"{dataset_name}_bitshuffle_lz4.h5")
    create_hdf5_bitshuffle(file_zstd, data, (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_ZSTD, COMP_LEVEL))
    create_hdf5_bitshuffle(file_lz4, data, (BLOCK_SIZE, bitshuffle.h5.H5_COMPRESS_LZ4))
    size_bitshuffle_zstd = os.path.getsize(file_zstd)
    size_bitshuffle_lz4 = os.path.getsize(file_lz4)

    # Filenames for raw compressed binary files (without preconditioning).
    file_raw_zstd = os.path.join(out_dir, f"{dataset_name}_raw_zstd.bin")
    file_raw_lz4 = os.path.join(out_dir, f"{dataset_name}_raw_lz4.bin")
    raw_zstd_bytes = raw_zstd_compress_file(file_raw_zstd, data, level=3)
    raw_lz4_bytes = raw_lz4_compress_file(file_raw_lz4, data, compressionLevel=3)
    size_raw_zstd = os.path.getsize(file_raw_zstd)
    size_raw_lz4 = os.path.getsize(file_raw_lz4)

    # Filenames for raw compressed binary files using byte-shuffle preconditioning.
    file_raw_zstd_bs = os.path.join(out_dir, f"{dataset_name}_raw_zstd_byteshuffle.bin")
    file_raw_lz4_bs = os.path.join(out_dir, f"{dataset_name}_raw_lz4_byteshuffle.bin")
    raw_zstd_bs_bytes = raw_zstd_byteshuffle_compress_file(file_raw_zstd_bs, data, level=3)
    raw_lz4_bs_bytes = raw_lz4_byteshuffle_compress_file(file_raw_lz4_bs, data, compressionLevel=3)
    size_raw_zstd_bs = os.path.getsize(file_raw_zstd_bs)
    size_raw_lz4_bs = os.path.getsize(file_raw_lz4_bs)

    # Calculate compression ratios.
    ratio_bitshuffle_zstd = original_size / size_bitshuffle_zstd
    ratio_bitshuffle_lz4 = original_size / size_bitshuffle_lz4
    ratio_raw_zstd = original_size / size_raw_zstd
    ratio_raw_lz4 = original_size / size_raw_lz4
    ratio_raw_zstd_bs = original_size / size_raw_zstd_bs
    ratio_raw_lz4_bs = original_size / size_raw_lz4_bs

    print(f"  Bitshuffle+ZSTD:         {size_bitshuffle_zstd} bytes  | Ratio: {ratio_bitshuffle_zstd:.2f}")
    print(f"  Bitshuffle+LZ4:          {size_bitshuffle_lz4} bytes  | Ratio: {ratio_bitshuffle_lz4:.2f}")
    print(f"  Raw ZSTD:                {size_raw_zstd} bytes  | Ratio: {ratio_raw_zstd:.2f}")
    print(f"  Raw LZ4:                 {size_raw_lz4} bytes  | Ratio: {ratio_raw_lz4:.2f}")
    print(f"  Raw ZSTD+ByteShuffle:    {size_raw_zstd_bs} bytes  | Ratio: {ratio_raw_zstd_bs:.2f}")
    print(f"  Raw LZ4+ByteShuffle:     {size_raw_lz4_bs} bytes  | Ratio: {ratio_raw_lz4_bs:.2f}")

    return {
        "dataset": dataset_name,
        "original_size": original_size,
        "bitshuffle_zstd_size": size_bitshuffle_zstd,
        "bitshuffle_lz4_size": size_bitshuffle_lz4,
        "raw_zstd_size": size_raw_zstd,
        "raw_lz4_size": size_raw_lz4,
        "raw_zstd_byteshuffle_size": size_raw_zstd_bs,
        "raw_lz4_byteshuffle_size": size_raw_lz4_bs,
        "bitshuffle_zstd_ratio": ratio_bitshuffle_zstd,
        "bitshuffle_lz4_ratio": ratio_bitshuffle_lz4,
        "raw_zstd_ratio": ratio_raw_zstd,
        "raw_lz4_ratio": ratio_raw_lz4,
        "raw_zstd_byteshuffle_ratio": ratio_raw_zstd_bs,
        "raw_lz4_byteshuffle_ratio": ratio_raw_lz4_bs,
    }

#############################################
# Main Function
#############################################

def main():
    # Specify the folder that contains your TSV datasets.
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"

    # Retrieve all .tsv files in the folder.
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in the folder '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'. Starting analysis...\n")
    # Build full paths for each TSV file.
    dataset_paths = [os.path.join(folder_path, f) for f in tsv_files]

    # Output directory for HDF5 and raw files.
    out_dir = "/home/jamalids/Documents/remaind/h5_datasets"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for path in dataset_paths:
        res = process_dataset(path, out_dir, dtype="float32", target_column=1)
        results.append(res)

    results_df = pd.DataFrame(results)
    csv_output = os.path.join(out_dir, "datasets_compression_spritz.csv")
    results_df.to_csv(csv_output, index=False)
    print(f"\nResults saved to {csv_output}")


if __name__ == "__main__":
    main()
