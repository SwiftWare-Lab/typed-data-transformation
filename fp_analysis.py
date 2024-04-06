import struct
import zlib
import numpy as np

from block_compression import generate_patterns, get_pattern_occurance_non_overlapping
from huffman_coding import huffman_codes


# convert a floating point number to a binary string
def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')


# convert a floating point number to a binary string
def xor_float(v1, v2):
    v1_str = format(struct.unpack('!I', struct.pack('!f', v1))[0], '032b')
    v2_str = format(struct.unpack('!I', struct.pack('!f', v2))[0], '032b')
    xor_str = format(int(v1_str, 2) ^ int(v2_str, 2), '032b')
    return xor_str


# convert a binary string to a floating point number
def bin_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]


# convert an array of floating point numbers to an array of binary strings
def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array


# compute all xors of every consecutive float32 number in an array and retuns the array of xors
def compute_xors(a):
    xors = []
    for i in range(len(a) - 1):
        xors.append(xor_float(a[i], a[i + 1]))
    return xors


# convert an array binary strings to a bit map image
def bin_to_image(b):
    img = []
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    return img


# plot a bit map image
def plot_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


# def generate random array of float32 numbers smoothly increasing from 0 to 1
def generate_smooth_array(n):
    import numpy as np
    a = np.linspace(0, 1, n).astype(np.float32)
    return a

# it takes a bit representation of a float32 number and estimates the compressed size
def compress_block_based(mat, m, n):
    stats = {}
    # original array size in bits
    stats['original_size'] = mat.shape[0] * mat.shape[1]
    # a list to store all patterns
    pattern_list = generate_patterns(m, n)
    stats['num_patterns'] = len(pattern_list)
    stats['m'] = m
    stats['n'] = n
    # get the occurance of each pattern in the matrix
    pattern_occurance = get_pattern_occurance_non_overlapping(mat, pattern_list)
    # total occurence
    sum_all_occurrences = sum(pattern_occurance)
    stats['total_occurrences'] = sum_all_occurrences
    # get the size of each pattern in bit
    num_nz_pattern_occured = sum(np.array(pattern_occurance) > 0)
    stats['num_nz_patterns'] = num_nz_pattern_occured
    #print(num_nz_pattern_occured)
    size_per_pattern = np.log2(num_nz_pattern_occured)  # in bits
    stats['size_per_pattern'] = size_per_pattern
    # round up to bit
    size_per_pattern_bit_roundup = np.ceil(np.log2(num_nz_pattern_occured))  # in bits
    stats['size_per_pattern_bit_roundup'] = size_per_pattern_bit_roundup
    # round up to the nearest integer of multiple of 8 (byte)
    size_per_pattern_byte_roundup = np.ceil(size_per_pattern / 8) * 8
    stats['size_per_pattern_byte_roundup'] = size_per_pattern_byte_roundup

    # uniform coding size in bit #
    size_uniform_code = size_per_pattern * sum_all_occurrences
    size_uniform_bit_roundup = size_per_pattern_bit_roundup * sum_all_occurrences
    size_unifrom_byte_roundup = size_per_pattern_byte_roundup * sum_all_occurrences
    stats['size_uniform_code'] = size_uniform_code
    stats['size_uniform_bit_roundup'] = size_uniform_bit_roundup
    stats['size_unifrom_byte_roundup'] = size_unifrom_byte_roundup

    # non-uniform coding using huffman
    dic_pattern = {}
    for i in range(len(pattern_occurance)):
        if pattern_occurance[i] > 0:
            dic_pattern[str(pattern_list[i])] = pattern_occurance[i]
    huffman_code = huffman_codes(dic_pattern)
    #print(huffman_code)
    # compute non-unofrm coding size
    size_huffman = 0
    for i in range(len(pattern_occurance)):
        if pattern_occurance[i] > 0:
            size_huffman += len(huffman_code[str(pattern_list[i])]) * pattern_occurance[i]
    stats['size_huffman'] = size_huffman
    return stats, pattern_occurance


# define main entry point
if __name__ == '__main__':
    # convert a floating point number to a binary string
    f = 3.14159
    b = float_to_bin(f)
    print(f, b)

    # print(xor_float(3.14159, 2.71828))
    # print(xor_float(3.14159, 3.14159))

    # convert a binary string to a floating point number
    f = bin_to_float(b)
    print(b, f)

    # convert an array binary strings to a bit map image
    a_len = 32

    # generate a random array of float32 numbers
    a = np.random.rand(a_len).astype(np.float32)
    str_array = float_to_bin_array(a)
    img = bin_to_image(str_array)
    plot_image(img)

    # use zlib to compress a
    a_bytes = a.tobytes()
    a_compressed = zlib.compress(a_bytes)
    print('ZLib Compression, Original size in bits: ', len(a_bytes)*8, 'Compressed size:', len(a_compressed)*8)

    # compress a using block based compression
    img_array = np.array(img)
    m, n = 8, 2
    stats, list_pattern_orig = compress_block_based(img_array, m, n)
    print(stats)

    # generate a smooth random array of float32 of less than 1
    a = generate_smooth_array(a_len)
    str_array = float_to_bin_array(a)
    img_orig = bin_to_image(str_array)
    plot_image(img_orig)

    # compute all xors
    # xors = compute_xors(a)
    # img_xor = bin_to_image(xors)
    # plot_image(img_xor)

    # use zlib to compress a
    a_bytes = a.tobytes()
    a_compressed = zlib.compress(a_bytes)
    print('ZLib Compression, Original size in bits: ', len(a_bytes)*8, 'Compressed size:', len(a_compressed)*8)

    # compress a using block based compression
    img_array = np.array(img_orig)
    m, n = 8, 2
    stats_smooth, list_pattern_smooth = compress_block_based(img_array, m, n)
    print(stats_smooth)

    # stats_xor, list_pattern_xor = compress_block_based(np.array(img_xor), m, n)
    # print(stats_xor)

    exit(0)
