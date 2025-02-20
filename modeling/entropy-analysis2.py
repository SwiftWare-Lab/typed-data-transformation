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
# Standard TDT Transform Function (Fixed: 4 groups)
# -------------------------------------------------------------------------
def tdt_transform(data):
    """
    Transpose-Downsample-Transpose transform for a 1D array of bytes.
    Splits the data into 4 groups (assuming 32-bit words) and computes
    kth-order entropy for each group.
    """
    data = np.asarray(data)
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
    data_transformed[0: p_len] = bytes0
    data_transformed[p_len: 2 * p_len] = bytes1
    data_transformed[2 * p_len: 3 * p_len] = bytes2
    data_transformed[3 * p_len: 4 * p_len] = bytes3
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
    for each group. It then concatenates the groups to produce a
    "transformed" byte array.

    Parameters:
      data: 1D array-like data.
      comp_sizes: list of sizes for each component (e.g., [1,1,1,1] for 32-bit data).
      grouping_config: grouping configuration (list of lists, 1-indexed).
      num_k: number of k orders to compute.

    Returns:
      transformed_data: a 1D np.uint8 array created by concatenating
                        the grouped components.
      entropy_info: dict mapping group index (as str) to a list of
                    entropy values for k = 0,...,num_k-1.
      grouped: the list of grouped component arrays.
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
        [[1, 2, 3, 4, 5, 6], [7], [8]],
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
# Standard Plot Function (Original vs. Standard TDT)
# -------------------------------------------------------------------------
def plot(kth_order_ent, orig_entropy, tdt_kth_order_ent, tdt_orig_entropy, output_path, metadata):
    """
    Plots a 1x4 grid comparing:
      - Original kth-order entropy curves
      - Original fixed-width entropy curves
      - Standard TDT kth-order entropy curves
      - Standard TDT fixed-width entropy curves
    """
    fig, axs = plt.subplots(1, 4, figsize=(25, 10))
    max_entropy = 0

    # find maximum y-value among all curves
    for k, values in kth_order_ent.items():
        max_entropy = max(max_entropy, max(values))
    for w, values in orig_entropy.items():
        max_entropy = max(max_entropy, max(values))
    for k, values in tdt_kth_order_ent.items():
        max_entropy = max(max_entropy, max(values))
    for w, values in tdt_orig_entropy.items():
        max_entropy = max(max_entropy, max(values))

    for ax in axs:
        ax.set_ylim(0, max_entropy)
        ax.set_yticks(np.arange(0, max_entropy, 0.3))
        ax.grid(True)

    axs[0].set_title(
        f"Original K-th Order Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[0].set_xlabel("Block Number")
    axs[0].set_ylabel("Entropy")
    for k, values in kth_order_ent.items():
        axs[0].plot(range(len(values)), values, label=f"K={k}")
    axs[0].legend()

    axs[1].set_title(
        f"Original Fixed-Width Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[1].set_xlabel("Block Number")
    axs[1].set_ylabel("Entropy")
    for w, values in orig_entropy.items():
        axs[1].plot(range(len(values)), values, label=f"{w}-bit", linestyle="--")
    axs[1].legend()

    axs[2].set_title(
        f"Standard TDT K-th Order Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[2].set_xlabel("Block Number")
    axs[2].set_ylabel("Entropy")
    for k, values in tdt_kth_order_ent.items():
        axs[2].plot(range(len(values)), values, label=f"TDT K={k}")
    axs[2].legend()

    axs[3].set_title(
        f"Standard TDT Fixed-Width Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[3].set_xlabel("Block Number")
    axs[3].set_ylabel("Entropy")
    for w, values in tdt_orig_entropy.items():
        axs[3].plot(range(len(values)), values, label=f"TDT {w}-bit", linestyle="--")
    axs[3].legend()

    plt.tight_layout()

    # make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


# -------------------------------------------------------------------------
# New Plot Function for Custom Transform Groups (Custom TDT)
# -------------------------------------------------------------------------
def plot_custom_groups(custom_entropy, output_path, metadata):
    """
    Creates a combined plot with one subplot per custom group
    (from the custom TDT transform).
    In each subplot, for each k order, the entropy values for that group
    across blocks are plotted.
    """
    global num_entropies

    group_keys = sorted({key for block in custom_entropy for key in block.keys()}, key=lambda x: int(x))
    num_groups = len(group_keys)

    fig, axs = plt.subplots(1, num_groups, figsize=(5 * num_groups, 5), squeeze=False)
    for idx, g in enumerate(group_keys):
        for k in range(num_entropies):
            curve = []
            for block in custom_entropy:
                if g in block:
                    curve.append(block[g][k])
                else:
                    curve.append(np.nan)
            axs[0, idx].plot(range(len(curve)), curve, marker="o", linestyle="-", label=f"K={k}")
        axs[0, idx].set_title(f"Custom Group {g}")
        axs[0, idx].set_xlabel("Block Number")
        axs[0, idx].set_ylabel("Entropy")
        axs[0, idx].grid(True)
        axs[0, idx].legend()

    plt.suptitle(
        f"Custom TDT Transform Group Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


# -------------------------------------------------------------------------
# New Overlay Plot Function: Compare Standard TDT vs. Custom TDT (Averaged)
# -------------------------------------------------------------------------
def plot_tdt_vs_custom(tdt_kth, tdt_fw, custom_kth, custom_fw, output_path, metadata):
    """
    Creates a 2x1 grid overlaying standard TDT and custom TDT
    (averaged over groups) entropy curves.
    Top panel: kth-order entropy curves.
    Bottom panel: fixed-width entropy curves.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: kth-order overlay
    max_top = 0
    for k, values in tdt_kth.items():
        max_top = max(max_top, max(values))
    for k, values in custom_kth.items():
        max_top = max(max_top, max(values))
    axs[0].set_ylim(0, max_top)
    axs[0].grid(True)
    for k, values in tdt_kth.items():
        axs[0].plot(range(len(values)), values, label=f"Std TDT K={k}", linestyle="-")
    for k, values in custom_kth.items():
        axs[0].plot(range(len(values)), values, label=f"Custom TDT K={k}", linestyle="--")
    axs[0].set_title(
        f"TDT vs. Custom TDT K-th Order Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[0].set_xlabel("Block Number")
    axs[0].set_ylabel("Entropy")
    axs[0].legend()

    # Bottom panel: fixed-width overlay
    max_bot = 0
    for w, values in tdt_fw.items():
        max_bot = max(max_bot, max(values))
    if isinstance(custom_fw, dict):
        for w, values in custom_fw.items():
            max_bot = max(max_bot, max(values))
    else:
        max_bot = max(max_bot, max(custom_fw))

    axs[1].set_ylim(0, max_bot)
    axs[1].grid(True)
    for w, values in tdt_fw.items():
        axs[1].plot(range(len(values)), values, label=f"Std TDT {w}-bit", linestyle="-")

    if isinstance(custom_fw, dict):
        for w, values in custom_fw.items():
            axs[1].plot(range(len(values)), values, label=f"Custom TDT {w}-bit", linestyle="--")
    else:
        axs[1].plot(range(len(custom_fw)), custom_fw, label="Custom TDT Fixed-Width", linestyle="--")

    axs[1].set_title(
        f"TDT vs. Custom TDT Fixed-Width Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[1].set_xlabel("Block Number")
    axs[1].set_ylabel("Entropy")
    axs[1].legend()

    plt.tight_layout()

    # make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


# -------------------------------------------------------------------------
# Additional Plot Function for side-by-side Original vs. Custom TDT
# -------------------------------------------------------------------------
def plot_custom_tdt(orig_kth, orig_fw, custom_kth, custom_fw, output_path, metadata):
    """
    Plots a 1x4 grid comparing:
      - Original kth-order entropy curves
      - Original fixed-width entropy curves
      - Custom TDT kth-order entropy curves
      - Custom TDT fixed-width entropy curves
    """
    fig, axs = plt.subplots(1, 4, figsize=(25, 10))

    # Find maximum y-value among all curves
    max_entropy = 0
    for k, values in orig_kth.items():
        max_entropy = max(max_entropy, max(values))
    for w, values in orig_fw.items():
        max_entropy = max(max_entropy, max(values))
    for k, values in custom_kth.items():
        max_entropy = max(max_entropy, max(values))

    if isinstance(custom_fw, dict):
        for w, values in custom_fw.items():
            max_entropy = max(max_entropy, max(values))
    else:
        max_entropy = max(max_entropy, max(custom_fw))

        # Set y-axis limits and grid
    for ax in axs:
        ax.set_ylim(0, max_entropy)
        ax.set_yticks(np.arange(0, max_entropy, 0.3))
        ax.grid(True)

        # Subplot 0: Original kth-order entropy
    axs[0].set_title(
        f"Original K-th Order Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[0].set_xlabel("Block Number")
    axs[0].set_ylabel("Entropy")
    for k, values in orig_kth.items():
        axs[0].plot(range(len(values)), values, label=f"K={k}")
    axs[0].legend()

    # Subplot 1: Original fixed-width entropy
    axs[1].set_title(
        f"Original Fixed-Width Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[1].set_xlabel("Block Number")
    axs[1].set_ylabel("Entropy")
    for w, values in orig_fw.items():
        axs[1].plot(range(len(values)), values, label=f"{w}-bit", linestyle="--")
    axs[1].legend()

    # Subplot 2: Custom TDT kth-order entropy
    axs[2].set_title(
        f"Custom TDT K-th Order Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[2].set_xlabel("Block Number")
    axs[2].set_ylabel("Entropy")
    for k, values in custom_kth.items():
        axs[2].plot(range(len(values)), values, label=f"K={k}")
    axs[2].legend()

    # Subplot 3: Custom TDT fixed-width entropy
    axs[3].set_title(
        f"Custom TDT Fixed-Width Entropy\n(dataset: {metadata['dataset']}, block: {metadata['chunk_size']})"
    )
    axs[3].set_xlabel("Block Number")
    axs[3].set_ylabel("Entropy")

    if isinstance(custom_fw, dict):
        for w, values in custom_fw.items():
            axs[3].plot(range(len(values)), values, label=f"{w}-bit", linestyle="--")
    else:
        axs[3].plot(range(len(custom_fw)), custom_fw, label="Custom Fixed-Width", linestyle="--")

    axs[3].legend()

    plt.tight_layout()

    # make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


# -------------------------------------------------------------------------
# Main Code: Data Loading, Transformation, Entropy Computation, and CSV Saving
# -------------------------------------------------------------------------
chunk_size = -1
data_size = 5000  # Adjust as needed
data = np.random.rand(data_size)
dataset_name = "random"

# If the user passes a dataset path, we load it
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    try:
        data_loaded = pd.read_csv(dataset_path, sep="\t")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        # Adjust slicing logic as needed
    sliced_data = data_loaded.values[:, 1].astype(np.float32)
    data = sliced_data.flatten()

# If the user passes a chunk_size, we use it
if len(sys.argv) > 2:
    chunk_size = int(sys.argv[2])
if chunk_size == -1:
    #chunk_size = 65536
    chunk_size =65536

data_size = chunk_size * 2
data = data[:data_size]
data = np.float64(data)
bit_width = data.dtype.itemsize * 8  # 32 or 64
data_int = data.view(dtype=np.uint8)

num_blocs = len(data_int) // chunk_size
num_entropies = 5

# Dictionaries for storing results
kth_order_ent = {str(k): [] for k in range(num_entropies)}
orig_entropy = {str(w): [] for w in [8, 16, 32]}
tdt_kth_order_ent = {str(k): [] for k in range(num_entropies)}
tdt_orig_entropy = {str(w): [] for w in [8, 16, 32]}

# For custom transform (Custom TDT), store a dictionary per block:
custom_decomposed_entropy = []

# Dictionary to store custom TDT fixed-width entropies
custom_tdt_orig_entropy = {str(w): [] for w in [8, 16, 32]}

byte_cluster_entropy = []

# Use compConfigMap to get the grouping configuration
if dataset_name in compConfigMap:
    grouping_configs = compConfigMap[dataset_name]
    grouping_config = grouping_configs[0]  # choose the first
else:
    grouping_config = compConfigMap["default"][0]

if bit_width == 32:
    comp_sizes = [1, 1, 1, 1]
elif bit_width == 64:
    comp_sizes = [1, 1, 1, 1, 1, 1, 1, 1]
else:
    comp_sizes = [1, 1, 1, 1]

last_custom_groups = None

for i in range(num_blocs):
    data_block = data_int[i * chunk_size: (i + 1) * chunk_size]

    # 1) Original entropy computations: K-th order and fixed-width
    for k in range(num_entropies):
        ent_val = compute_kth_order_entropy_optimized(data_block, k)
        kth_order_ent[str(k)].append(ent_val)
        print(f"{k}-th order entropy for original block {i}: {ent_val}")

    for w in [8, 16, 32]:
        manual_entropy, _ = compute_entropy_w(data_block, w)
        orig_entropy[str(w)].append(manual_entropy)
        print(f"Original {w}-bit fixed-width entropy (block {i}): {manual_entropy}")

        # 2) Standard TDT transformation + Entropy computations
    tdt_data, _ = tdt_transform(data_block)
    tdt_data_int = tdt_data.view(dtype=np.uint8)

    # (a) Fixed-width on TDT data
    for w in [8, 16, 32]:
        manual_entropy, _ = compute_entropy_w(tdt_data, w)
        tdt_orig_entropy[str(w)].append(manual_entropy)
        print(f"Standard TDT {w}-bit fixed-width entropy (block {i}): {manual_entropy}")

        # (b) K-th order on TDT data
    for k in range(num_entropies):
        ent_val = compute_kth_order_entropy_optimized(tdt_data_int, k)
        tdt_kth_order_ent[str(k)].append(ent_val)
        print(f"{k}-th order entropy for standard TDT block {i}: {ent_val}")

        # 3) Custom TDT transformation
    custom_data, custom_entropy, custom_groups = tdt_custom_transform(
        tdt_data, comp_sizes, grouping_config, num_k=num_entropies
    )
    custom_decomposed_entropy.append(custom_entropy)

    if i == num_blocs - 1:
        last_custom_groups = custom_groups

    for key, ent_list in custom_entropy.items():
        print(f"Block {i}, Custom Group {key} entropy: {ent_list}")

        # (a) Custom TDT fixed-width for each w
    for w in [8, 16, 32]:
        manual_entropy, _ = compute_entropy_w(custom_data, w)
        custom_tdt_orig_entropy[str(w)].append(manual_entropy)
        print(f"Custom TDT {w}-bit fixed-width entropy (block {i}): {manual_entropy}")

    # -------------------- CSV Saving --------------------
df_orig_kth = pd.DataFrame(kth_order_ent)
df_tdt_kth = pd.DataFrame(tdt_kth_order_ent)
df_orig_fw = pd.DataFrame(orig_entropy)
df_tdt_fw = pd.DataFrame(tdt_orig_entropy)

# convert custom TDT group entropies to CSV
df_custom = pd.DataFrame(custom_decomposed_entropy).applymap(
    lambda x: ",".join(map(str, x)) if isinstance(x, list) else x
)

df_orig_kth.to_csv("original_kth_entropy.csv", index=False)
df_tdt_kth.to_csv("tdt_kth_entropy.csv", index=False)
df_orig_fw.to_csv("original_fixedwidth_entropy.csv", index=False)
df_tdt_fw.to_csv("tdt_fixedwidth_entropy.csv", index=False)
df_custom.to_csv("custom_decomposed_entropy.csv", index=False)

# custom TDT fixed-width dictionary -> CSV
df_custom_fw = pd.DataFrame(custom_tdt_orig_entropy)
df_custom_fw.to_csv("custom_tdt_fixedwidth_entropy.csv", index=False)

print("Entropy results saved to CSV files.")

# -------------------- Plotting --------------------
# Use the same folder as dataset_path (if provided). Otherwise, current dir.
if len(sys.argv) > 1:
    output_dir = os.path.dirname(dataset_path)
else:
    output_dir = os.getcwd()

metadata = {"dataset": dataset_name, "chunk_size": chunk_size}

# Build output paths
plot_filename = f"{dataset_name}_entropy_plot.png"
plot_filepath = os.path.join(output_dir, plot_filename)

# 1) Plot Standard Results (Original vs. Standard TDT)
plot(kth_order_ent, orig_entropy, tdt_kth_order_ent, tdt_orig_entropy, plot_filepath, metadata)

# 2) Plot Custom Transform Groups: Each group
custom_groups_filename = f"{dataset_name}_custom_groups_plot.png"
custom_groups_filepath = os.path.join(output_dir, custom_groups_filename)
plot_custom_groups(custom_decomposed_entropy, custom_groups_filepath, metadata)

# 3) Overlay Plot: Compare Standard TDT vs. Custom TDT (Averaged)
custom_kth_avg = {str(k): [] for k in range(num_entropies)}
custom_fw_avg = []

for block in custom_decomposed_entropy:
    # average K-th order across groups
    for k in range(num_entropies):
        vals = [block[g][k] for g in block]
        custom_kth_avg[str(k)].append(np.mean(vals))

        # Example logic for average fixed-width across groups
    fw_vals = []
    for g in block:
        # for demonstration, just take the last_custom_groups
        group_array = last_custom_groups[int(g)]
        manual_fw, _ = compute_entropy_w(group_array, bit_width)
        fw_vals.append(manual_fw)
    custom_fw_avg.append(np.mean(fw_vals))

overlay_filename = f"{dataset_name}_tdt_vs_custom_overlay.png"
overlay_filepath = os.path.join(output_dir, overlay_filename)
plot_tdt_vs_custom(tdt_kth_order_ent, tdt_orig_entropy, custom_kth_avg, custom_fw_avg, overlay_filepath, metadata)

# 4) Plot side-by-side Original vs. Custom TDT (if desired)
custom_tdt_filename = f"{dataset_name}_custom_tdt_plot.png"
custom_tdt_filepath = os.path.join(output_dir, custom_tdt_filename)
plot_custom_tdt(
    orig_kth=kth_order_ent,
    orig_fw=orig_entropy,
    custom_kth=custom_kth_avg,  # your custom kth-order dictionary
    custom_fw=custom_tdt_orig_entropy,  # pass the dictionary with bit-width keys
    output_path=custom_tdt_filepath,
    metadata=metadata,
)

