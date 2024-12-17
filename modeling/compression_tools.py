import numpy as np
import zstd
import zlib
import bz2
import snappy
import lzma as lz2
import lz4.frame as fastlz
from dahuffman import HuffmanCodec


def zstd_comp(data_set):
    return zstd.compress(data_set, 3)


def zlib_comp(data, level=9):
    return zlib.compress(data, level)

def bz2_comp(data, level=9):
    return bz2.compress(data, level)

def snappy_comp(data):
    return snappy.compress(data)

# def lzma_comp(data, level=9):
#     return lz2.compress(data, level)

def fastlz_compress(data):
    return  fastlz.compress(data)


def rle_compress(data):

    if isinstance(data, np.ndarray):
        data = data.tobytes()
    if not data:
        return b'', 0
    compressed_data = []
    count = 1
    last = data[0]
    for current in data[1:]:
        if current == last and count < 255:
            count += 1
        else:
            compressed_data.extend([count, last])
            last = current
            count = 1
    compressed_data.extend([count, last])
    compressed_bytes = bytes(compressed_data)
    #compression_ratio = len(data) / len(compressed_bytes) if compressed_bytes else 1
    return compressed_bytes

def huffman_compress(data):
    # Assume data is a bytes
    codec = HuffmanCodec.from_data(data)
    compressed_data = codec.encode(data)
    return compressed_data