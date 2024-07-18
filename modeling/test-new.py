import sys

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import io
import numpy as np
from utils import generate_smooth_array, generate_oscillating_2d_array, floats_to_bool_arrays
from huffman_code import create_huffman_tree, create_huffman_codes, encode, decode
import pandas as pd

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
    com_ratio_tot=(image_smooth.shape[0] * image_smooth.shape[1]) / est_tot_size
    com_ratio= (image_smooth.shape[0] * image_smooth.shape[1]) / est_tot_size
    print(data_name, ": ", (image_smooth.shape[0] * image_smooth.shape[1]) /  est_size,
          " -- ", (image_smooth.shape[0] * image_smooth.shape[1]) / est_tot_size)
    # plot the count
    plt.bar(np.arange(len(sm_pat_count)), sm_pat_count, color='b')
    plt.title(data_name)
    plt.show()
    plt.savefig("1.png")
    return est_size, est_tot_size,com_ratio,com_ratio_tot


#array_size = int(sys.argv[1])
#pattern_size = int(sys.argv[2])
#array_size =10000000

# make array size to be a multiple of pattern size
#array_size = array_size + (pattern_size - array_size % pattern_size)

# Load the image
#image = io.imread("texture.jpg", grayscale=True)  # Assuming grayscale image


pattern_sizes = [12]  # Example pattern sizes
results_df=[]

for pattern_size in pattern_sizes:
    binary_patterns, binary_code = create_binary_patterns(pattern_size)

    # path_tsv = sys.argv[3]
    # Load TSV into DataFrame
    path_tsv = '/home/jamalids/Documents/2D/data1/num_brain_f64.tsv'
    ts_dataset = pd.read_csv(path_tsv, delimiter="\t")

    # Drop the first column
    ts_dataset.drop(ts_dataset.columns[0], axis=1, inplace=True)

    # Convert to numpy array
    ts_dataset = ts_dataset.T
    ts_array = ts_dataset.to_numpy().flatten().astype(np.float32)

    new_array_size = ts_array.shape[0] - ts_array.shape[0] % pattern_size
    ts_array = ts_array[:new_array_size]

    bool_array = floats_to_bool_arrays(ts_array)
    image_ts = b_array_to_int_array(bool_array)

    est_size, est_tot_size,com_ratio, com_ratio_tot = profile_data(image_ts, binary_patterns, binary_code, "TS: " + path_tsv)


    zstd_compressed_ts, comp_ratio_zstd = compress_with_zstd(ts_array)
    print("zstd:", comp_ratio_zstd)

    # Append the results to the DataFrame
    results_df.append({
        'Pattern Size': pattern_size,
        'Estimated Size': est_size,
        'Estimated Total Size': est_tot_size,
        'Compression Ratio': com_ratio,
        'Compression Ratio_tot': com_ratio_tot,
        'Compression Ratio_zstd': comp_ratio_zstd
    })
    results=pd.DataFrame(results_df)
    radius = 1  # Radius of the neighborhood
    neighbors = 8  # Number of neighbors
    #
    # # Calculate LBP features
    lbp = local_binary_pattern(image_ts, neighbors, radius)
    # # make a histogram of the LBP features
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    #
    uniqe_lbp(lbp, (3, 3))
    # # Visualize the LBP features (optional)
    plt.imshow(image_ts, cmap="gray")
    plt.title("Local Binary Pattern (LBP) Features")
    plt.savefig("2.png")
    plt.show()
    plt.bar(np.arange(n_bins), hist, color='b')
    plt.title("LBP Histogram")
    plt.savefig("3.png")
    plt.show()


# Save the DataFrame to a CSV file
results.to_csv('hst_wfc3_ir_f32.csv', index=False)




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