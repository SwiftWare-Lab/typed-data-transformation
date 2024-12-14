import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zstandard as zstd
import os


def decompose_array_three(max_lead, min_tail, array):
    leading_zero_array = array[:, :max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_zero_array = array[:, min_tail:]
    return leading_zero_array, content_array, trailing_zero_array


def compress_with_zstd(data, level=3):
    print("Compression level:", level)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    print("Original data size:", len(data))
    print("Compressed size:", len(compressed))
    comp_ratio = len(data) / len(compressed) if compressed else float('inf')  # Prevent division by zero
    return compressed, comp_ratio


def float_to_ieee754(f):
    """Convert a float to its IEEE 754 binary representation as an integer array."""
    binary_str = format(np.float32(f).view(np.uint32), '032b')
    return np.array([int(bit) for bit in binary_str], dtype=np.uint8)


def plot_bitmap_standalone(bool_array, name):
    plt.figure(figsize=(12, 8))
    plt.imshow(bool_array, cmap='gray_r', aspect='auto')
    plt.xlabel('Bit Position (0-32)')
    plt.ylabel('Float Index')
    plt.colorbar(label='Bit Value')
    plt.savefig(name)
    plt.show()


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    chunk_size = 256 * 1024  # 256 KB chunks

    results = []
    try:
        # Assuming the file is small enough to be read in full for simplicity
        # For truly large files, use `pd.read_csv(..., chunksize=...)`
        data_full = pd.read_csv(dataset_path, delimiter='\t', header=None)
        data_full = data_full.drop(data_full.columns[0], axis=1).astype(np.float32)
        num_rows = data_full.shape[0]

        # Process data in chunks
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            data_chunk = data_full.iloc[start:end]
            data_array = data_chunk.to_numpy().T.flatten()

            bool_array = np.array([float_to_ieee754(f) for f in data_array]).reshape(-1, 32)
            plot_bitmap_standalone(bool_array, f"bitmap_{start}_{end}.png")

            # Example compression call - modify as needed
            compressed, ratio = compress_with_zstd(data_array.tobytes())
            print(f"Compression ratio for chunk {start}-{end}: {ratio}")
            results.append((start, end, ratio))

    except Exception as e:
        print(f"Error processing dataset: {e}")

    return results


if __name__ == "__main__":
    results = run_and_collect_data()
    print("Compression Results:", results)
