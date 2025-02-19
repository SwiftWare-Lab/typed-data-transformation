import os
import time
import numpy as np
import bitshuffle

# Maximum allowed size for one compression call (in bytes)
MAX_SIZE = 2147483631  # ~2 GB


def chunked_compress_bitshuffle(file_bytes, block_size, max_size=MAX_SIZE):
    """
    Compress the input bytes in chunks using Bitshuffle's ZSTD compressor.
    Returns a list of tuples: (compressed_chunk, original_chunk_length)
    """
    chunks = []
    total = len(file_bytes)
    for i in range(0, total, max_size):
        chunk = file_bytes[i:i + max_size]
        # Convert the chunk into a 1D NumPy array of uint8.
        data = np.frombuffer(chunk, dtype=np.uint8)
        # Compress using Bitshuffle's ZSTD compressor.
        comp_chunk = bitshuffle.compress_zstd(data, data.dtype.itemsize, block_size)
        chunks.append((comp_chunk, len(chunk)))
    return chunks


def chunked_decompress_bitshuffle(chunks, block_size):
    """
    Decompress each chunk using Bitshuffle's ZSTD decompressor and concatenate the results.
    Each tuple in 'chunks' is (compressed_chunk, original_chunk_length).
    """
    decompressed_list = []
    for comp_chunk, orig_chunk_length in chunks:
        # For decompression, the original data is a 1D array with length equal to the chunk's byte length.
        decomp = bitshuffle.decompress_zstd(comp_chunk, 1, (orig_chunk_length,), block_size)
        decompressed_list.append(decomp.tobytes())
    return b''.join(decompressed_list)


def compress_and_verify(file_path, block_size):
    """
    Compress a file's binary data with Bitshuffle (using chunked compression if necessary),
    then decompress and verify that the decompressed data matches the original.

    Returns a tuple with:
      - original size in bytes,
      - total compressed size in bytes,
      - compression time in seconds,
      - decompression time in seconds,
      - a boolean indicating whether the decompressed data matches the original.
    """
    # Read the file as binary data.
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    orig_size = len(file_bytes)

    if orig_size <= MAX_SIZE:
        # Single-chunk compression.
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        start = time.time()
        comp_bytes = bitshuffle.compress_zstd(data, data.dtype.itemsize, block_size)
        comp_time = time.time() - start
        comp_size = len(comp_bytes)

        start = time.time()
        # Decompress using the original shape (a 1D array of length orig_size).
        decomp = bitshuffle.decompress_zstd(comp_bytes, 1, (orig_size,), block_size)
        decomp_time = time.time() - start
        valid = (decomp.tobytes() == file_bytes)
        return orig_size, comp_size, comp_time, decomp_time, valid
    else:
        # File is too large; compress in chunks.
        start = time.time()
        comp_chunks = chunked_compress_bitshuffle(file_bytes, block_size=block_size, max_size=MAX_SIZE)
        comp_time = time.time() - start
        comp_size = sum(len(chunk) for chunk, _ in comp_chunks)

        start = time.time()
        decompressed_bytes = chunked_decompress_bitshuffle(comp_chunks, block_size=block_size)
        decomp_time = time.time() - start
        valid = (decompressed_bytes == file_bytes)
        return orig_size, comp_size, comp_time, decomp_time, valid


def run_analysis(folder_path, block_size):
    """
    Process all .tsv files in the given folder using Bitshuffle for compression.
    Prints the original size, compressed size, compression ratio, and timing for each file.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Find all .tsv files (case-insensitive)
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print(f"No .tsv files found in folder '{folder_path}'.")
        return

    print(f"Found {len(tsv_files)} .tsv file(s) in '{folder_path}'.\n")
    for tsv_file in tsv_files:
        file_path = os.path.join(folder_path, tsv_file)
        print(f"Processing '{tsv_file}'...")
        orig_size, comp_size, comp_time, decomp_time, valid = compress_and_verify(file_path, block_size=block_size)
        comp_ratio = orig_size / comp_size if comp_size else float('inf')
        print(f"Original size: {orig_size} bytes")
        print(f"Compressed size: {comp_size} bytes")
        print(f"Compression ratio: {comp_ratio:.2f}x")
        print(f"Compression time: {comp_time:.4f} s")
        print(f"Decompression time: {decomp_time:.4f} s")
        if not valid:
            print("Error: Decompressed data does not match the original!")
        print("-" * 60)


if __name__ == "__main__":
    # Update this path to point to your dataset folder.
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
    # Set a positive block size; 32768 is a common value.
    run_analysis(folder_path, block_size=32768)
