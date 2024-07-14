import sys

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import io
import numpy as np
from utils import generate_smooth_array, generate_oscillating_2d_array, floats_to_bool_arrays
from huffman_code import create_huffman_tree, create_huffman_codes, encode, decode


# compress with zstd
def compress_with_zstd(data, level=3):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    # comp ratio
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio

def generate_random_array(n, type=np.float32):
    a = np.random.rand(n).astype(type)
    return a

def b_array_to_int_array(b_array, type=np.int64):
    int_array = np.zeros(b_array.shape, dtype=type)
    for i in range(b_array.shape[0]):
        for j in range(b_array.shape[1]):
            int_array[i, j] = int.from_bytes(b_array[i, j], byteorder='big')
    return int_array


# create all permutations of nx1 binary pattern
def create_binary_patterns(n):
    binary_patterns = []
    binary_code = np.zeros(2**n, dtype=np.int64)
    for i in range(2**n):
        list_char = np.array(list(format(i, f'0{n}b')))
        list_int = list(map(int, list_char))
        one_d = np.array(list_int)
        two_d = np.array(one_d).reshape(-1, 1)
        binary_patterns.append(two_d)
        binary_code[i] = int(format(i, f'0{n}b'), 2)
    return binary_patterns, binary_code


# conver 1d aray to code
def binary_to_code(binary_array):
    code = 0
    for i in range(binary_array.shape[0]):
        code += binary_array[i] * 2**i
    return code

# find all occurrences of a pattern in a 2D array
def find_pattern_in_array(array, pattern, binary_code):
    pattern_positions = []
    pattern_count = np.zeros(len(binary_code))
    for i in range(0, array.shape[0], pattern[0].shape[0]):
        for j in range(0, array.shape[1], pattern[0].shape[1]):
            cur_pat = array[i:i+pattern[0].shape[0], j:j+pattern[0].shape[1]]
            # convert the pattern to a single integer
            pat_id = binary_to_code(cur_pat.flatten())
            pattern_positions.append((i, j))
            pattern_count[pat_id] += 1
    return pattern_positions, pattern_count


# get unique lbp
def uniqe_lbp(lbp_array, shape):
    reduced_lbp = np.zeros((lbp_array.shape[0]//shape[0], lbp_array.shape[1]//shape[1]))
    # for every shape 3x3 in the lbp array
    lbp_array_shapes = []
    for i in range(0, lbp_array.shape[0], shape[0]):
        for j in range(0, lbp_array.shape[1], shape[1]):
            lbp_array_shapes.append(lbp_array[i:i+shape[0], j:j+shape[1]])
    #print(lbp_array_shapes)
    return reduced_lbp


def compute_huffman_code(pattern_count, binary_code):
    dict_code = {}
    for i in range(len(pattern_count)):
        dict_code[str(binary_code[i])] = pattern_count[i]
    # make the huffman code
    root = create_huffman_tree(dict_code)

    # Create Huffman codes dictionary
    codes = {}
    create_huffman_codes(root, "", codes)
    #print(codes)
    estimated_size = 0
    for key in codes:
        estimated_size += len(codes[key]) * pattern_count[int(key)]

    dict_size = 0
    # for key and value in codes
    for key, value in codes.items():
        dict_size += len(value)
    # # Encode the text
    # encoded_text = encode(text, codes)
    #
    # # Decode the encoded text
    # decoded_text = decode(encoded_text, root)

    return estimated_size, estimated_size + dict_size


def profile_data(image_smooth, binary_patterns, binary_code, data_name):
    sm_pat_pos, sm_pat_count = find_pattern_in_array(image_smooth, binary_patterns, binary_code)
    # sum of all counts
    assert np.sum(sm_pat_count)*binary_patterns[0].shape[0] == image_smooth.shape[0] * image_smooth.shape[1]
    est_size, est_tot_size = compute_huffman_code(sm_pat_count, binary_code)
    print((image_smooth.shape[0] * image_smooth.shape[1]) /  est_size, " -- ", (image_smooth.shape[0] * image_smooth.shape[1]) / est_tot_size)
    # plot the count
    plt.bar(np.arange(len(sm_pat_count)), sm_pat_count, color='b')
    plt.title(data_name)
    #plt.show()
    #plt.close()


array_size = int(sys.argv[1])
pattern_size = int(sys.argv[2])
binary_patterns, binary_code = create_binary_patterns(pattern_size)
# Load the image
#image = io.imread("texture.jpg", grayscale=True)  # Assuming grayscale image

# generate a random binary texture image
# image = np.random.choice([0, 1], size=(100, 100))
smooth_array = generate_smooth_array(array_size)
bool_array = floats_to_bool_arrays(smooth_array)
image_smooth = b_array_to_int_array(bool_array)
profile_data(image_smooth, binary_patterns, binary_code, "Smooth TS")
zstd_compressed, comp_ratio = compress_with_zstd(smooth_array)
print("zstd:", comp_ratio)

oscilate_array = generate_oscillating_2d_array((array_size, 1))
bool_array = floats_to_bool_arrays(oscilate_array[:, 0])
image_oscilate = b_array_to_int_array(bool_array)
profile_data(image_oscilate, binary_patterns, binary_code, "Oscilating TS")
zstd_compressed, comp_ratio = compress_with_zstd(oscilate_array)
print("zstd:", comp_ratio)

random_array = generate_random_array(array_size)
bool_array = floats_to_bool_arrays(random_array)
image_random = b_array_to_int_array(bool_array)
profile_data(image_random, binary_patterns, binary_code, "Random TS")
zstd_compressed, comp_ratio = compress_with_zstd(random_array)
print("zstd:", comp_ratio)



#
# # Define parameters for LBP
# radius = 1  # Radius of the neighborhood
# neighbors = 8  # Number of neighbors
#
# # Calculate LBP features
# lbp = local_binary_pattern(image_smooth, neighbors, radius)
# # make a histogram of the LBP features
# n_bins = int(lbp.max() + 1)
# hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
#
# uniqe_lbp(lbp, (3, 3))
#
#
#
# # Visualize the LBP features (optional)
# plt.imshow(image_smooth, cmap="gray")
# plt.title("Local Binary Pattern (LBP) Features")
# plt.show()
# plt.close()
#
#
# # Plot the histogram
# plt.bar(np.arange(n_bins), hist, color='b')
# plt.title("LBP Histogram")
# plt.show()
# plt.close()
#
#
#
# lbp_osc = local_binary_pattern(image_oscilate, neighbors, radius)
# hist_osc, _ = np.histogram(lbp_osc, bins=n_bins, range=(0, n_bins))
#
# plt.imshow(image_oscilate, cmap="gray")
# plt.title("Local Binary Pattern (LBP) Oscilating Features")
# plt.show()
# plt.close()
#
#
#
# plt.bar(np.arange(n_bins), hist_osc, color='b')
# plt.title("LBP Oscilating Histogram")
# plt.show()
# plt.close()
#
#
# lbp_random = local_binary_pattern(image_random, neighbors, radius)
# hist_random, _ = np.histogram(lbp_random, bins=n_bins, range=(0, n_bins))
#
# plt.imshow(image_random, cmap="gray")
# plt.title("Local Binary Pattern (LBP) Random Features")
# plt.show()
# plt.close()
#
# plt.bar(np.arange(n_bins), hist_random, color='b')
# plt.title("LBP Random Histogram")
# plt.show()
# plt.close()