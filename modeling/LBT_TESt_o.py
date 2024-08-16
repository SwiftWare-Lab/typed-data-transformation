import sys

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import io
import numpy as np
from utils import generate_smooth_array, generate_oscillating_2d_array, floats_to_bool_arrays
from huffman_code import create_huffman_tree, create_huffman_codes
import pandas as pd


# compress with zstd
def compress_with_zstd(data, level=3):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    # comp ratio
    comp_ratio = len(data.tobytes()) / len(compressed)
    return compressed, comp_ratio




def synthetic_test(array_size, binary_patterns):
    # make array size to be a multiple of pattern size
    array_size = array_size + (pattern_size - array_size % pattern_size)

    # Load the image
    # image = io.imread("texture.jpg", grayscale=True)  # Assuming grayscale image
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
    zstd_compressed_osc, comp_ratio = compress_with_zstd(oscilate_array)
    print("zstd:", comp_ratio)
    random_array = generate_random_array(array_size)
    bool_array = floats_to_bool_arrays(random_array)
    image_random = b_array_to_int_array(bool_array)
    profile_data(image_random, binary_patterns, binary_code, "Random TS")
    zstd_compressed_rand, comp_ratio = compress_with_zstd(random_array)
    print("zstd:", comp_ratio, "\n")


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


# find all occurrences of a pattern in a 2D array not in the given mask
def find_pattern_in_array_not_in_mask(array, pattern, mask, binary_code):
    pattern_positions = []
    pattern_count = np.zeros(len(binary_code))
    mask_copy = mask.copy()
    for i in range(0, array.shape[0], pattern[0].shape[0]):
        for j in range(0, array.shape[1], pattern[0].shape[1]):
            cur_pat = array[i:i+pattern[0].shape[0], j:j+pattern[0].shape[1]]
            # convert the pattern to a single integer
            pat_id = binary_to_code(cur_pat.flatten())
            mask_pat = np.sum(mask[i:i+pattern[0].shape[0], j:j+pattern[0].shape[1]])
            if mask_pat == 0:
                pattern_positions.append((i, j))
                pattern_count[pat_id] += 1
                mask_copy[i:i + pattern[0].shape[0], j:j + pattern[0].shape[1]] = 1
    return pattern_positions, pattern_count, mask_copy


# take an array and compute repetition of values
def compute_repetition(array):
    unique, counts = np.unique(array, return_counts=True)
    # get the top 10 values in counts
    max_top_10 = np.argsort(counts)[-50:]
    #print("Top 10 values: ", unique[max_top_10], counts[max_top_10])
    return dict(zip(unique, counts))


PLOTING_DISABLE = False
def plot_historgam(freq_dict, ax=None, log_scale=False, y_label=""):
    if PLOTING_DISABLE:
        return
    ax.bar(freq_dict.keys(), freq_dict.values(), color='b')
    # log scale
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel("Frequency of "+ y_label)
    # set y label


def plot_ts(ts_array, ax=None, plot_y_axis=""):
    if PLOTING_DISABLE:
        return
    ax.plot(ts_array)
    #ax.set_title(plot_title)
    # set y axis
    ax.set_ylabel(plot_y_axis)


def plot_bar(values, x_labels, y_label, ax=None):
    if PLOTING_DISABLE:
        return
    ax.bar(np.arange(len(values)), values, color='b')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel(y_label)


def compress_float_repetition(array, frq_dict):
    # compute the frequency of the values
    # frq_dict = compute_repetition(array)
    # compute the huffman code
    # TODO replcae keys with the binary code
    est_size, est_tot_size = huffman_code_array(array)
    # est_size, est_tot_size = compute_huffman_code(list(frq_dict.values()), list(frq_dict.keys()))
    # total unique values
    uniq_vals = len(frq_dict)
    required_bits = np.ceil(np.log2(uniq_vals))
    # uniform coding
    uniform_code_len = (required_bits * array.shape[0]) + (uniq_vals * (required_bits + 32))
    return est_size, est_tot_size, uniform_code_len


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


def compute_leading_tailing_zeros(array):
    leading_zeros_array = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        leading_zeros = 0
        # find first non zero in array[i]
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                leading_zeros += 1
            else:
                break
        leading_zeros_array[i] = leading_zeros

    trailing_zeros_array = np.zeros(array.shape[0])
    for i in range(array.shape[0]-1, -1, -1):
        trailing_zeros = 0
        for j in range(array.shape[1]-1, -1, -1):
            if array[i, j] == 0:
                trailing_zeros += 1
            else:
                break
        trailing_zeros_array[i] = 32 - trailing_zeros
    return leading_zeros_array, trailing_zeros_array


def huffman_code_array(array):
    # compute the frequency of the values
    frq_dict = compute_repetition(array)
    pattern_count = np.fromiter(frq_dict.values(), dtype=int)
    binary_code = np.array(range(len(pattern_count)), dtype=int)
    dict_code = {}
    for i in range(len(pattern_count)):
        dict_code[str(binary_code[i])] = pattern_count[i]
    root = create_huffman_tree(dict_code)
    # Create Huffman codes dictionary
    codes = {}
    create_huffman_codes(root, "", codes)
    # print(codes)
    estimated_size = 0
    for key in codes:
        estimated_size += len(codes[key]) * pattern_count[int(key)]
    dict_size = 0
    # for key and value in codes
    for key, value in codes.items():
        dict_size += len(value)
    return estimated_size, estimated_size + dict_size


def profile_data(image_smooth, binary_patterns, binary_code, data_name):
    sm_pat_pos, sm_pat_count = find_pattern_in_array(image_smooth, binary_patterns, binary_code)
    #max_top_10 = np.argsort(sm_pat_count)[-100:]
    #print("Top 10 values: ", sm_pat_count[max_top_10])
    # sum of all counts
    assert int(np.sum(sm_pat_count))*binary_patterns[0].shape[0] == image_smooth.shape[0] * image_smooth.shape[1]
    est_size, est_tot_size = compute_huffman_code(sm_pat_count, binary_code)
    print(data_name, ": ", (image_smooth.shape[0] * image_smooth.shape[1]) /  est_size,
          " -- ", (image_smooth.shape[0] * image_smooth.shape[1]) / est_tot_size)
    # plot the count
    # plt.bar(np.arange(len(sm_pat_count)), sm_pat_count, color='b')
    # plt.title(data_name)
    # plt.show()
    # plt.close()
    sm_pat_dict = {}
    for i in range(len(sm_pat_count)):
        sm_pat_dict[str(binary_code[i])] = sm_pat_count[i]
    return est_size, est_tot_size, sm_pat_dict


def decompose_array(min_lead, max_lead, min_tail, max_tail, array):
    leading_zero_array = array[:, :min_lead]
    assert np.sum(leading_zero_array) == 0
    leading_mixed_array = array[:, min_lead:max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_mixed_array = array[:, min_tail:max_tail]
    trailing_zero_array = array[:, max_tail:]
    assert np.sum(trailing_zero_array) == 0
    return leading_zero_array, leading_mixed_array, content_array, trailing_mixed_array, trailing_zero_array


def decompose_array_three(max_lead, min_tail, array):
    leading_zero_array = array[:, :max_lead]
    content_array = array[:, max_lead:min_tail]
    trailing_zero_array = array[:, min_tail:]
    return leading_zero_array, content_array, trailing_zero_array


def decomposition_based_compression(image_ts, leading_zero_pos, tail_zero_pos):
    min_lead, max_lead, avg_lead, min_tail, max_tail, avg_tail = int(np.min(leading_zero_pos)), int(np.max(leading_zero_pos)), \
        int(np.mean(leading_zero_pos)), int(np.min(tail_zero_pos)), int(np.max(tail_zero_pos)), int(np.mean(tail_zero_pos))
    print("Min Lead: ", min_lead, "Max Lead: ", max_lead, "avg lead: ", avg_lead, "Min Tail: ", min_tail, "Max Tail: ", max_tail, "Avg Tail: ", avg_tail)
    bnd1 = max_lead if max_lead < 28 else avg_lead  # 28 and 4 are ad hoc number to avoid weird case all zeros
    bnd2 = min_tail if min_tail >= 4 else avg_tail
    print("Bnd1: ", bnd1, "Bnd2: ", bnd2)
    # plot the leading and trailing zeros
    plot_ts(leading_zero_pos, axs[1, 1], "Leading Zeros")
    plot_ts(tail_zero_pos, axs[1, 2], "Trailing Zeros")

    plot_historgam(dict_freq_pattern, axs[1, 0], True, "Patterns")
    lead_comp_size, tail_comp_size, content_comp_size = [], [], []
    tune_decomp = [0, 1, 2]
    for i in tune_decomp:
        print("Tune Decomp: ", i)
        bnd1 = bnd1 + i
        bnd2 = bnd2 - i
        leading_zero_array_orig, content_array_orig, trailing_mixed_array_orig = decompose_array_three(bnd1, bnd2, image_ts)
        pattern_size_list = [4, 6, 8, 10, 12] # this is a  tuning parameter
        for pattern_size in pattern_size_list:
            print("Pattern Size: ", pattern_size)
            binary_patterns, binary_code = create_binary_patterns(pattern_size)
            new_array_size = image_ts.shape[0] - image_ts.shape[0] % pattern_size
            leading_zero_array, trailing_mixed_array, content_array = leading_zero_array_orig[:new_array_size], trailing_mixed_array_orig[:new_array_size], content_array_orig[:new_array_size]
            est_size_pattern_lead, est_tot_size_pattern_lead, dict_freq_pattern_lead = profile_data(leading_zero_array, binary_patterns, binary_code, "TS: "+path_tsv+" Leading Zeros")
            est_size_pattern_tail, est_tot_size_pattern_tail, dict_freq_pattern_tail = profile_data(trailing_mixed_array, binary_patterns, binary_code, "TS: "+path_tsv+" Trailing Zeros")
            est_size_pattern_content, est_tot_size_pattern_content, dict_freq_pattern_content = profile_data(content_array, binary_patterns, binary_code, "TS: "+path_tsv+" Content")
            lead_comp_size.append(est_tot_size_pattern_lead)
            tail_comp_size.append(est_tot_size_pattern_tail)
            content_comp_size.append(est_tot_size_pattern_content)
        cont_size, tot_size = huffman_code_array(leading_zero_pos)
        lead_comp_size.append(tot_size)
        cont_size, tot_size = huffman_code_array(tail_zero_pos)
        tail_comp_size.append(tot_size)

    tot_decomp_tot_size = np.min(lead_comp_size) + np.min(content_comp_size) + np.min(tail_comp_size)

    return tot_decomp_tot_size


# main
#array_size = int(sys.argv[1])
#pattern_size = int(sys.argv[2])
array_size = 100
pattern_size =4
binary_patterns, binary_code = create_binary_patterns(pattern_size)

# make a plot with 4x2 subplots
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# increase the distance of two row subplots
plt.subplots_adjust(hspace=0.5)

if 1==1:
    log_dict = {}
    #path_tsv = sys.argv[3]
    path_tsv = "/home/jamalids/Documents/2D/UCRArchive_2018/ACSF1/ACSF1_TEST.tsv"
    # load tsf in df
    ts_dataset = pd.read_csv(path_tsv, delimiter="\t")
    # drop the first column
    ts_dataset.drop(ts_dataset.columns[0], axis=1, inplace=True)
    # convert to numpy array
    ts_array = ts_dataset.to_numpy().flatten().astype(np.float32)
    original_size_bits = len(ts_array.tobytes()) * 8
    # plot the ts
    plot_ts(ts_dataset.values.flatten(), axs[0, 0], "Original Values")

    log_dict["Original Size (bytes)"] = original_size_bits // 8
    log_dict["Dataset Name"] = path_tsv.split("/")[-1].split(".")[0]
    log_dict["Number of Signals"] = ts_dataset.shape[0]
    log_dict["Number of Samples"] = ts_dataset.shape[1]

    #zstd:
    zstd_compressed_ts, comp_ratio_zstd_default = compress_with_zstd(ts_array)
    print("len:", len(zstd_compressed_ts) * 8, "zstd default:", comp_ratio_zstd_default)
    log_dict["Zstd Default-3 (bytes)"] = len(zstd_compressed_ts)

    zstd_compressed_ts_l22, comp_ratio_l22 = compress_with_zstd(ts_array, 22)
    print("len:", len(zstd_compressed_ts_l22) * 8, "zstd ultimate:", comp_ratio_l22)
    log_dict["Zstd Ultimate-22 (bytes)"] = len(zstd_compressed_ts_l22)

    # 1x4 compression
    frq_dict = compute_repetition(ts_array)
    plot_historgam(frq_dict, axs[0, 1], False, "Pattern 1x4")

    est_size, Non_uniform_1x4, uniform_code_len = compress_float_repetition(ts_array, frq_dict)
    print("TS: ", est_size, Non_uniform_1x4, "Non-Uniform:", original_size_bits / Non_uniform_1x4, "Uniform:", original_size_bits / uniform_code_len)
    log_dict["Uniform Code 1x4 (bytes)"] = uniform_code_len//8
    log_dict["Non-Uniform Code 1x4 (bytes)"] = Non_uniform_1x4 // 8

    # pattern based
    new_array_size = ts_array.shape[0] - ts_array.shape[0] % pattern_size
    ts_array_trimmed = ts_array[:new_array_size]
    bool_array = floats_to_bool_arrays(ts_array_trimmed)
    image_ts = b_array_to_int_array(bool_array)
    est_size_pattern, est_tot_size_pattern, dict_freq_pattern = profile_data(image_ts, binary_patterns, binary_code, "TS: "+path_tsv)
    print("CR Global pattern based ", pattern_size, " : ", original_size_bits / est_tot_size_pattern)
    log_dict["Global Pattern Based (bytes)"] = est_tot_size_pattern//8
    log_dict["Global Pattern Size"] = pattern_size

    # pattern based decomposition
    l_z_array, t_z_array = compute_leading_tailing_zeros(image_ts)
    tot_decomp_tot_size = decomposition_based_compression(image_ts, l_z_array, t_z_array)
    print("TS: ", tot_decomp_tot_size, est_tot_size_pattern, original_size_bits, "--", original_size_bits / tot_decomp_tot_size)
    log_dict["Decomposition Based (bytes)"] = tot_decomp_tot_size//8

    comp_ratio_array = np.array([comp_ratio_zstd_default, comp_ratio_l22, original_size_bits / Non_uniform_1x4, original_size_bits / uniform_code_len, original_size_bits / est_tot_size_pattern, original_size_bits / tot_decomp_tot_size])

    # plot the compression ratio
    plot_bar(comp_ratio_array, ["Zstd Default-3", "Zstd Ultimate-22", "Huffman 1x4", "Uniform 1x4", "Huffman Enumerated "+str(pattern_size), "Decomposition Pattern"], "Compression Ratio", axs[0, 2])

    name_dataset = log_dict["Dataset Name"]
    if not PLOTING_DISABLE:
        plt.title(name_dataset)
        #plt.show()
        plt.savefig(f"results/{name_dataset}.png")
        plt.close()
    # convert to dataframe
    log_df = pd.DataFrame([log_dict])
    log_df.to_csv(f"results/{name_dataset}.csv", index=False)

else:
    synthetic_test(array_size, binary_patterns)





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