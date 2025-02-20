import numpy as np
from pandas import value_counts
from scipy.stats import entropy
import math
from collections import Counter
import sys
import pandas as pd
import matplotlib
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Helper Functions: Probability and Entropy
# -------------------------------------------------------------------------
def compute_probablity(data):
    """Computes the probability of each element in a 1D array."""
    data = np.asarray(data)
    uniq_vals = np.unique(data)
    num_bins = len(uniq_vals)
    value_counts = np.zeros(num_bins)
    for i, val in enumerate(uniq_vals):
        value_counts[i] = np.sum(data == val)
    assert np.sum(value_counts) == len(data)
    return value_counts, value_counts / len(data)

def compute_entropy(data):
    """Computes the Shannon entropy of a 1D array."""
    data = np.asarray(data)
    value_counts, probabilities = compute_probablity(data)
    assert np.abs(np.sum(probabilities) - 1) < 1e-8
    entropy_manual = -np.sum(probabilities * np.log2(probabilities))
    entropy_scipy = entropy(np.array(probabilities), base=2)
    return entropy_manual, entropy_scipy

# -------------------------------------------------------------------------
# k-th Order Entropy Functions
# -------------------------------------------------------------------------
def calculate_kth_order_entropy_paper(sequence, k, modified=True):
    """
    Calculates the kth-order Shannon entropy of a sequence.
    """
    n = len(sequence)
    if k >= n:
        raise ValueError("k must be less than the length of the sequence.")
    k_grams = [tuple(sequence[i: i + k]) for i in range(n - k)]
    unique_k_grams = set(k_grams)
    k_th_order = 0.0
    for k_gram in unique_k_grams:
        loc = [j for j in range(n - k) if tuple(sequence[j: j + k]) == k_gram]
        next_symbols = [sequence[j + k] for j in loc]
        entropy_i, entropy_i_sci = compute_entropy(next_symbols)
        len_i = len(next_symbols)
        if entropy_i == 0 and modified:
            entropy_i_sci = (1 + np.log2(n)) / n
        k_th_order += len_i * entropy_i_sci
    k_th_order /= n
    return k_th_order

def compute_kth_order_entropy_optimized(sequence, k, modified=True):
    """
    Optimized computation of kth-order entropy.
    """
    n = len(sequence)
    if k >= n:
        raise ValueError("k must be less than the length of the sequence.")
    k_gram_map = {}
    for i in range(n - k):
        k_gram = tuple(sequence[i: i + k])
        if k_gram not in k_gram_map:
            k_gram_map[k_gram] = []
        k_gram_map[k_gram].append(i)
    k_th_order = 0.0
    for k_gram, locations in k_gram_map.items():
        next_symbols = [sequence[j + k] for j in locations]
        entropy_i, entropy_i_sci = compute_entropy(next_symbols)
        len_i = len(next_symbols)
        if entropy_i == 0 and modified:
            entropy_i_sci = (1 + np.log2(n)) / n
        k_th_order += len_i * entropy_i_sci
    k_th_order /= n
    return k_th_order

# -------------------------------------------------------------------------
# Fixed-width Entropy Function (Modified)
# -------------------------------------------------------------------------
def compute_entropy_w(data, w):
    """
    Computes the entropy of a 1D array after casting it to a fixed-width integer type.
    This version uses the byte representation and trims extra bytes so that its
    length is divisible by the dtype itemsize.
    """
    data = np.asarray(data)
    b = data.tobytes()
    dtype_dict = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    itemsize = np.dtype(dtype_dict[w]).itemsize
    trim = len(b) % itemsize
    if trim != 0:
        b = b[:-trim]
    data_casted = np.frombuffer(b, dtype=dtype_dict[w])
    manual_entropy, entropy_scipy = compute_entropy(data_casted)
    return manual_entropy, entropy_scipy

# -------------------------------------------------------------------------
# Standard TDT Transform Function (Fixed: 4 groups for 32-bit)
# -------------------------------------------------------------------------
def tdt_transform(data):
    """
    Transpose-Downsample-Transpose transform for a 1D array of bytes.
    Splits the data into 4 groups (assuming 32-bit words) and computes kth-order
    entropy for each group. If the data length is not divisible by 4, it is trimmed.
    """
    data = np.asarray(data)
    trim = len(data) % 4
    if trim != 0:
        data = data[:-trim]
    bytes0 = data[0::4]
    bytes1 = data[1::4]
    bytes2 = data[2::4]
    bytes3 = data[3::4]
    entropy_info = {str(k): [] for k in range(6)}
    for k_level in range(6):
        for group in [bytes0, bytes1, bytes2, bytes3]:
            ent = calculate_kth_order_entropy_paper(group, k_level)
            entropy_info[str(k_level)].append(ent)
    data_transformed = np.zeros(len(data), dtype=np.uint8)
    p_len = len(data) // 4
    data_transformed[0:p_len] = bytes0
    data_transformed[p_len:2*p_len] = bytes1
    data_transformed[2*p_len:3*p_len] = bytes2
    data_transformed[3*p_len:4*p_len] = bytes3
    return data_transformed, entropy_info

# -------------------------------------------------------------------------
# Split and Group Functions (for custom transform)
# -------------------------------------------------------------------------
def split_bytes_into_components(byte_array, component_sizes):
    """
    Splits a byte array into individual components based on specified sizes.
    """
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        comp = byte_array[offset::total_bytes]
        components.append(comp)
    return components

def group_components(components, grouping_config):
    """
    Groups a list of component arrays according to a grouping configuration.
    The configuration uses 1-indexed component numbers.
    """
    grouped = []
    for group in grouping_config:
        indices = [i - 1 for i in group]
        group_array = np.concatenate([components[i] for i in indices])
        grouped.append(group_array)
    return grouped

# -------------------------------------------------------------------------
# Custom TDT Transform Function (Based on compConfigMap)
# -------------------------------------------------------------------------
def tdt_custom_transform(data, comp_sizes, grouping_config, num_k=6):
    """
    Custom TDT transform: splits data into components using comp_sizes,
    groups them according to grouping_config, and computes kth-order entropy
    for each group. It then concatenates the groups to produce a "transformed"
    byte array.
    """
    data = np.asarray(data)
    b = data.tobytes()
    components = split_bytes_into_components(b, comp_sizes)
    grouped = group_components(components, grouping_config)

    entropy_info = {str(idx): [] for idx in range(len(grouped))}
    for idx, group in enumerate(grouped):
        for k in range(num_k):
            ent = compute_kth_order_entropy_optimized(group, k)
            entropy_info[str(idx)].append(ent)
    transformed_bytes = b"".join([group.tobytes() for group in grouped])
    transformed_data = np.frombuffer(transformed_bytes, dtype=np.uint8)
    return transformed_data, entropy_info, grouped

# -------------------------------------------------------------------------
# Sample Grouping Configuration Map
# -------------------------------------------------------------------------
# The base configuration is defined here.
# For this example, the default configuration is set to ([[1,2,3],[4]]).
compConfigMap = {
    "acs_wht_f32": [[[1, 2], [3], [4]], [[1, 2, 3], [4]], [[1, 2, 3, 4]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]], [[1, 2, 3, 4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]], [[1, 2, 3, 4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]], [[1, 2, 3], [4]], [[1, 2, 3, 4]]],
    "wave_f32": [[[1, 2], [3], [4]], [[1, 2, 3], [4]], [[1, 2, 3, 4]]],
    "hdr_night_f32": [
        [[1, 4], [2], [3]],
        [[1], [2], [3], [4]],
        [[1, 4], [2, 3]],
        [[1, 4, 2, 3]],
    ],
    "ts_gas_f32": [[[1, 2], [3], [4]], [[1, 2, 3, 4]]],
    "solar_wind_f32": [[[1], [2, 3], [4]], [[1, 2, 3, 4]]],
    "tpch_lineitem_f32": [
        [[1, 2, 3], [4]],
        [[1, 2], [3], [4]],
        [[1, 2, 3, 4]],
    ],
    "tpcds_web_f32": [
        [[1, 2, 3], [4]],
        [[1], [2, 3], [4]],
        [[1, 2, 3, 4]],
    ],
    "tpcds_store_f32": [
        [[1, 2, 3], [4]],
        [[1], [2, 3], [4]],
        [[1, 2, 3, 4]],
    ],
    "tpcds_catalog_f32": [
        [[1, 2, 3], [4]],
        [[1], [2, 3], [4]],
        [[1, 2, 3, 4]],
    ],
    "citytemp_f32": [
        [[1, 4], [2, 3]],
        [[1], [2], [3], [4]],
        [[1, 2], [3], [4]],
        [[1, 2, 3, 4]],
    ],
    "hst_wfc3_ir_f32": [

        [[1, 2], [3], [4]],
        [[1, 2, 3, 4]],
    ],
    "hst_wfc3_uvis_f32": [

        [[1, 2], [3], [4]],
        [[1, 2, 3, 4]],
    ],
    "rsim_f32": [
        [[1, 2, 3], [4]],
        [[1, 2], [3], [4]],
        [[1, 2, 3, 4]],
    ],
    "astro_mhd_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],
        [[1, 2, 3, 4, 5], [6], [7], [8]],
    ],
    "astro_pt_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],
        [[1, 2, 3, 4, 5], [6], [7], [8]],
    ],
    "jane_street_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],
        [[3, 2, 5, 6, 4, 1], [7], [8]],
        [[3, 2, 5, 6, 4, 1, 7, 8]],
    ],
    "msg_bt_f64": [
        [[1, 2, 3, 4, 5], [6], [7], [8]],
        [[3, 2, 1, 4, 5, 6], [7], [8]],
        [[3, 2, 1, 4, 5], [6], [7], [8]],
        [[3, 2, 1, 4, 5, 6, 7, 8]],
    ],
    "num_brain_f64": [
        [[3, 2, 4, 5, 1, 6], [7], [8]],
        [[3, 2, 4, 5, 1, 6, 7, 8]],
    ],
    "num_control_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],
        [[4, 5], [6, 3], [1, 2], [7], [8]],
        [[4, 5, 6, 3, 1, 2], [7], [8]],
        [[4, 5, 6, 3, 1, 2, 7, 8]],
    ],
    "nyc_taxi2015_f64": [
        [[7, 4, 6], [5], [3, 2, 1, 8]],
        [[7, 4, 6, 5], [3, 2, 1, 8]],
        [[7, 4, 6], [5], [3, 2, 1], [8]],
        [[7, 4], [6], [5], [3, 2], [1], [8]],
        [[7, 4, 6, 5, 3, 2, 1, 8]],
    ],
    "phone_gyro_f64": [
        [[4, 6], [8], [3, 2, 1, 7], [5]],
        [[4, 6], [1], [3, 2], [5], [7], [8]],
        [[6, 4, 3, 2, 1, 7], [5], [8]],
        [[6, 4, 3, 2, 1, 7, 5], [8]],
        [[6, 4, 3, 2, 1, 7], [5, 8]],
        [[6, 4, 3, 2, 1, 7, 5, 8]],
    ],
    "tpch_order_f64": [
        [[1, 2, 3, 4], [7], [6, 5], [8]],
        [[3, 2, 4, 1], [7], [6, 5], [8]],
        [[3, 2, 4, 1, 7], [6, 5], [8]],
        [[3, 2, 4, 1, 7, 6, 5, 8]],
    ],
    "tpcxbb_store_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],
        [[4, 2, 3, 1], [5], [7, 6], [8]],
        [[4, 2, 3, 1, 5], [7, 6], [8]],
        [[4, 2, 3, 1, 5, 7, 6, 8]],
    ],
    "tpcxbb_web_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],
        [[4, 2, 3, 1], [5], [7, 6], [8]],
        [[4, 2, 3, 1, 5], [7, 6], [8]],
        [[4, 2, 3, 1, 5, 7, 6, 8]],
    ],
    "wesad_chest_f64": [
        [[7, 5, 6], [8, 4, 1, 3, 2]],
        [[7, 5], [6], [8, 4, 1], [3, 2]],
        [[7, 5], [6], [8, 4], [1], [3, 2]],
        [[7, 5], [6], [8, 4, 1, 3, 2]],
        [[7, 5, 6, 8, 4, 1, 3, 2]],
    ],
    "default": [[[1], [2], [3], [4]]],
}

# -------------------------------------------------------------------------
# Main Code: Data Loading, Transformation, Entropy Computation, CSV Saving, and Plotting
# -------------------------------------------------------------------------
chunk_size = -1
data_size = 5000  # Adjust as needed
data = np.random.rand(data_size)
dataset_name = "random"

# If a dataset path is passed, load it
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    try:
        data_loaded = pd.read_csv(dataset_path, sep="\t")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    sliced_data = data_loaded.values[:, 1].astype(np.float32)
    data = sliced_data.flatten()

# If a chunk_size is passed, use it
if len(sys.argv) > 2:
    chunk_size = int(sys.argv[2])
if chunk_size == -1:
    chunk_size = 65536


data_size = chunk_size * 2
data = data[:data_size]
data = np.float32(data)
bit_width = data.dtype.itemsize * 8  # 32 or 64
data_int = data.view(dtype=np.uint8)

num_blocs = len(data_int) // chunk_size
num_entropies = 5

# Dictionaries for storing results (8-bit fixed-width measures)
kth_order_ent = {str(k): [] for k in range(num_entropies)}
orig_entropy = {"8": []}
tdt_kth_order_ent = {str(k): [] for k in range(num_entropies)}
tdt_orig_entropy = {"8": []}
custom_decomposed_entropy = []  # for custom TDT groups
custom_tdt_orig_entropy = {"8": []}

# For TDT-after-Custom measures (compute weighted entropy using configuration weights)
tdt_after_custom_fw_8bit = []
tdt_after_custom_kth_order_ent = {str(k): [] for k in range(num_entropies)}

# Use compConfigMap to get the grouping configuration (base configuration will be used)
if dataset_name in compConfigMap:
    grouping_configs = compConfigMap[dataset_name]
    grouping_config = grouping_configs[0]
else:
    grouping_config = compConfigMap["default"][0]

# Compute comp_sizes based on bit_width (for 32-bit: [1,1,1,1]; for 64-bit: eight components)
if bit_width == 32:
    comp_sizes = [1, 1, 1, 1]
elif bit_width == 64:
    comp_sizes = [1,1,1,1,1,1,1,1]
else:
    comp_sizes = [1, 1, 1, 1]

last_custom_groups = None

for i in range(num_blocs):
    data_block = data_int[i * chunk_size: (i + 1) * chunk_size]

    # 1) Original entropy computations
    for k in range(num_entropies):
        ent_val = compute_kth_order_entropy_optimized(data_block, k)
        kth_order_ent[str(k)].append(ent_val)
        print(f"{k}-th order entropy for original block {i}: {ent_val}")
    manual_entropy, _ = compute_entropy_w(data_block, 8)
    orig_entropy["8"].append(manual_entropy)
    print(f"Original 8-bit fixed-width entropy (block {i}): {manual_entropy}")

    # 2) Standard TDT transformation + entropy computations
    tdt_data, _ = tdt_transform(data_block)
    tdt_data_int = tdt_data.view(dtype=np.uint8)
    manual_entropy, _ = compute_entropy_w(tdt_data, 8)
    tdt_orig_entropy["8"].append(manual_entropy)
    print(f"Standard TDT 8-bit fixed-width entropy (block {i}): {manual_entropy}")
    for k in range(num_entropies):
        ent_val = compute_kth_order_entropy_optimized(tdt_data_int, k)
        tdt_kth_order_ent[str(k)].append(ent_val)
        print(f"{k}-th order entropy for standard TDT block {i}: {ent_val}")

    # 3) Custom TDT transformation (using the base configuration from compConfigMap)
    custom_data, custom_entropy, custom_groups = tdt_custom_transform(
        tdt_data, comp_sizes, grouping_config, num_k=num_entropies
    )
    custom_decomposed_entropy.append(custom_entropy)
    if i == num_blocs - 1:
        last_custom_groups = custom_groups
    for key, ent_list in custom_entropy.items():
        print(f"Block {i}, Custom Group {key} entropy: {ent_list}")
    manual_entropy, _ = compute_entropy_w(custom_data, 8)
    custom_tdt_orig_entropy["8"].append(manual_entropy)
    print(f"Custom TDT 8-bit fixed-width entropy (block {i}): {manual_entropy}")

    # 4) TDT-after-Custom:
    # Process each custom group individually with standard TDT transform.
    # Compute weighted entropy using configuration weights automatically derived from the grouping configuration.
    # For example, if grouping_config = [[1,2,3],[4]], then weights = [len([1,2,3])/4, len([4])/4] = [3/4, 1/4].
    grouping_config_weights = [len(g) / sum(len(g) for g in grouping_config) for g in grouping_config]
    group_fw_entropies = []
    group_kth_entropies = {str(k): [] for k in range(num_entropies)}
    for grp in custom_groups:
        tdt_after_grp, _ = tdt_transform(grp)
        tdt_after_grp_int = tdt_after_grp.view(dtype=np.uint8)
        group_entropy_fw, _ = compute_entropy_w(tdt_after_grp, 8)
        group_fw_entropies.append(group_entropy_fw)
        for k in range(num_entropies):
            ent_val = compute_kth_order_entropy_optimized(tdt_after_grp_int, k)
            group_kth_entropies[str(k)].append(ent_val)
    weighted_fw_entropy = sum(w * e for w, e in zip(grouping_config_weights, group_fw_entropies))
    tdt_after_custom_fw_8bit.append(weighted_fw_entropy)
    print(f"TDT-after-Custom (weighted) 8-bit fixed-width entropy (block {i}): {weighted_fw_entropy}")
    for k in range(num_entropies):
        weighted_kth = sum(w * e for w, e in zip(grouping_config_weights, group_kth_entropies[str(k)]))
        tdt_after_custom_kth_order_ent[str(k)].append(weighted_kth)
        print(f"{k}-th order weighted entropy for TDT-after-Custom block {i}: {weighted_kth}")

# -------------------- Combine Results into One CSV File --------------------
rows = []
num_blocks = len(next(iter(kth_order_ent.values())))
for i in range(num_blocks):
    row = {}
    row["block"] = i
    # Original data measures
    for k in range(num_entropies):
        row[f"orig_k{k}"] = kth_order_ent[str(k)][i]
    row["orig_fw_8bit"] = orig_entropy["8"][i]
    # Standard TDT measures
    for k in range(num_entropies):
        row[f"tdt_k{k}"] = tdt_kth_order_ent[str(k)][i]
    row["tdt_fw_8bit"] = tdt_orig_entropy["8"][i]
    # Custom TDT measures (averaged across custom groups)
    custom_block = custom_decomposed_entropy[i]
    for k in range(num_entropies):
        k_vals = [custom_block[g][k] for g in custom_block]
        row[f"custom_k{k}"] = np.mean(k_vals)
    row["custom_fw_8bit"] = custom_tdt_orig_entropy["8"][i]
    # TDT-after-Custom measures
    for k in range(num_entropies):
        row[f"tdt_after_custom_k{k}"] = tdt_after_custom_kth_order_ent[str(k)][i]
    row["tdt_after_custom_fw_8bit"] = tdt_after_custom_fw_8bit[i]
    rows.append(row)

df_all = pd.DataFrame(rows)
output_csv = "/home/jamalids/Documents/combined_entropy_results_8bit.csv"
df_all.to_csv(output_csv, index=False)
print(f"Combined 8-bit entropy results saved to {output_csv}")

#
# -------------------- Plotting --------------------
# Create subplots comparing the average k-order entropy ratios:
# x-axis: k (0 to num_entropies-1)
# y-axis: ratio of orig_k / tdt_k, orig_k / custom_k, orig_k / tdt_after_custom_k
ks = np.arange(num_entropies)
avg_orig = np.array([df_all[f"orig_k{k}"].mean() for k in range(num_entropies)])
avg_tdt = np.array([df_all[f"tdt_k{k}"].mean() for k in range(num_entropies)])
avg_custom = np.array([df_all[f"custom_k{k}"].mean() for k in range(num_entropies)])
avg_tdt_after_custom = np.array([df_all[f"tdt_after_custom_k{k}"].mean() for k in range(num_entropies)])

# Determine a common y-axis limit for all plots (starting at 0)
all_ratios = np.concatenate([avg_orig/avg_tdt, avg_orig/avg_custom, avg_orig/avg_tdt_after_custom])
ymax = np.max(all_ratios) * 1.1

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(ks, avg_orig / avg_tdt, marker='o')
axs[0].set_title("orig_k / NormalTransform_k")
axs[0].set_xlabel("k")
axs[0].set_ylabel("Ratio")
axs[0].set_ylim(0, ymax)

axs[1].plot(ks, avg_orig / avg_custom, marker='o')
axs[1].set_title("orig_k / TDT_k")
axs[1].set_xlabel("k")
axs[1].set_ylabel("Ratio")
axs[1].set_ylim(0, ymax)

axs[2].plot(ks, avg_orig / avg_tdt_after_custom, marker='o')
axs[2].set_title("orig_k / decompose_k")
axs[2].set_xlabel("k")
axs[2].set_ylabel("Ratio")
axs[2].set_ylim(0, ymax)

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/plot.png")
plt.show()