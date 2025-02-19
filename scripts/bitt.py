import os
import time
import blosc

# Blosc's maximum input size (in bytes) is limited by a signed 32-bit integer.
MAX_SIZE = 1073741824  # 1 GB

def chunked_compress(file_bytes, compression_codec='zstd', clevel=3, shuffle=blosc.SHUFFLE, max_size=MAX_SIZE):
    """
    Compresses the input bytes in chunks if needed.
    Returns a list of compressed chunks.
    """
    chunks = []
    total = len(file_bytes)
    for i in range(0, total, max_size):
        chunk = file_bytes[i:i+max_size]
        comp_chunk = blosc.compress(chunk, typesize=4, cname=compression_codec,
                                    clevel=clevel, shuffle=shuffle)
        chunks.append(comp_chunk)
    return chunks

def chunked_decompress(chunks):
    """
    Decompresses each chunk and concatenates the results.
    Returns the full decompressed bytes.
    """
    decompressed_data = b''.join(blosc.decompress(chunk) for chunk in chunks)
    return decompressed_data

def compress_and_verify(file_path, compression_codec='zstd', clevel=5, shuffle=blosc.SHUFFLE):
    """
    Compress a file's binary data with Blosc (using chunked compression if necessary),
    then decompress and verify the data.

    Returns a tuple with:
      - original size in bytes,
      - total compressed size in bytes,
      - compression time in seconds,
      - decompression time in seconds,
      - a boolean indicating whether decompressed data matches the original.
    """
    # Read the file as binary data.
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    orig_size = len(file_bytes)
    typesize = 1  # For binary/text data, each byte is one element.

    # Compress the data (chunked if needed)
    start = time.time()
    if orig_size <= MAX_SIZE:
        # Single-chunk compression.
        compressed = blosc.compress(file_bytes, typesize=typesize, cname=compression_codec,
                                    clevel=clevel, shuffle=shuffle)
        comp_chunks = [compressed]
    else:
        # File is too large; compress in chunks.
        comp_chunks = chunked_compress(file_bytes, compression_codec, clevel, shuffle)
    comp_time = time.time() - start
    comp_size = sum(len(chunk) for chunk in comp_chunks)

    # Decompress the data.
    start = time.time()
    if len(comp_chunks) == 1:
        decompressed = blosc.decompress(comp_chunks[0])
    else:
        decompressed = chunked_decompress(comp_chunks)
    decomp_time = time.time() - start

    # Verify that decompressed data matches the original.
    valid = (decompressed == file_bytes)

    return orig_size, comp_size, comp_time, decomp_time, valid

def run_analysis(folder_path, compression_codec='zstd', clevel=5, shuffle=blosc.SHUFFLE):
    """
    Process all .tsv files in the given folder using Blosc for compression.
    Prints original size, compressed size, compression ratio, and timing for each file.
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

        orig_size, comp_size, comp_time, decomp_time, valid = compress_and_verify(
            file_path, compression_codec=compression_codec, clevel=clevel, shuffle=shuffle
        )

        # Calculate compression ratio: original size divided by compressed size.
        comp_ratio = orig_size / comp_size if comp_size else float('inf')

        print(f"Original size: {orig_size} bytes")
        print(f"Compressed size: {comp_size} bytes")
        print(f"Compression ratio: {comp_ratio:.2f}x")
        print(f"Compression time: {comp_time:.4f} s")
        print(f"Decompression time: {decomp_time:.4f} s")
        if not valid:
            print("Error: Decompressed data does not match original!")
        print("-" * 60)

if __name__ == "__main__":
    # Update this path to point to your dataset folder.
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"

    # You can adjust the codec, compression level, and shuffle filter as needed.
    run_analysis(folder_path, compression_codec='zstd', clevel=3, shuffle=blosc.SHUFFLE)
