import numpy as np
import pandas as pd
import matplotlib
import os
import sys

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.stats import entropy

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def compute_probablity(data):
    """Computes the probability of each element in a 1D array."""
    data = np.asarray(data)
    uniq_vals = np.unique(data)
    num_bins = len(uniq_vals)
    value_counts = np.zeros(num_bins, dtype=np.int64)
    for i, val in enumerate(uniq_vals):
        value_counts[i] = np.sum(data == val)
    return value_counts, value_counts / len(data)


def compute_entropy(data):
    """Computes the Shannon entropy of a 1D array (base 2)."""
    data = np.asarray(data)
    counts, probs = compute_probablity(data)
    # Guard against any zero probabilities (to avoid NaN in log2)
    probs = probs[probs > 0]
    entropy_manual = -np.sum(probs * np.log2(probs))
    entropy_scipy = entropy(probs, base=2)
    return entropy_manual, entropy_scipy


def compute_kth_order_entropy(sequence, k, modified=True):
    """
    k-th-order Shannon entropy. If zero-entropy, optionally apply a fallback.
    """
    seq = np.asarray(sequence, dtype=np.uint8)
    n = len(seq)
    if k >= n:
        return 0.0

    kgram_map = {}
    for i in range(n - k):
        kgram = tuple(seq[i : i + k])
        if kgram not in kgram_map:
            kgram_map[kgram] = []
        kgram_map[kgram].append(seq[i + k])

    total_entropy = 0.0
    for kgram, next_syms in kgram_map.items():
        e, e_scipy = compute_entropy(next_syms)
        c = len(next_syms)
        # fallback if e == 0
        if e == 0 and modified:
            # fallback, e.g. (1 + log2(n)) / n
            e_scipy = (1 + np.log2(n)) / n
        total_entropy += c * e_scipy

    return total_entropy / n


def compute_entropy_w(data, w_bits):
    """
    Cast 'data' to a fixed-width type (8,16,32,64 bits), then compute Shannon entropy.
    """
    data = np.asarray(data)
    b = data.tobytes()
    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    dt = dtype_map[w_bits]
    itemsize = np.dtype(dt).itemsize
    trim = len(b) % itemsize
    if trim != 0:
        b = b[:-trim]
    data_casted = np.frombuffer(b, dtype=dt)
    e_man, e_sp = compute_entropy(data_casted)
    return e_man, e_sp


# -------------------------------------------------------------------------
# Standard TDT, parameterized for 32-bit (4 groups) or 64-bit (8 groups)
# -------------------------------------------------------------------------
def tdt_transform(data, word_bits=32):
    """
    If word_bits=32 => 4 interleaved groups.
    If word_bits=64 => 8 interleaved groups.
    """
    arr = np.asarray(data, dtype=np.uint8)
    group_count = word_bits // 8  # 4 if 32-bit, 8 if 64-bit, etc.

    # Trim so length is multiple of group_count
    trim = len(arr) % group_count
    if trim != 0:
        arr = arr[:-trim]

    # Build groups interleaved by group_count
    groups = []
    for i in range(group_count):
        groups.append(arr[i::group_count])

    # Concatenate them in that order
    transformed = np.concatenate(groups)
    return transformed


# -------------------------------------------------------------------------
# Split & Group for Custom TDT
# -------------------------------------------------------------------------
def split_bytes_into_components(data_bytes, component_sizes):
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    total = sum(component_sizes)
    comps = []
    for i, sz in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        comp = arr[offset::total]
        comps.append(comp)
    return comps


def group_components(components, grouping_config):
    grouped = []
    for group_list in grouping_config:
        zero_based = [g - 1 for g in group_list]
        merged = np.concatenate([components[i] for i in zero_based])
        grouped.append(merged)
    return grouped


# -------------------------------------------------------------------------
# Custom TDT (WITH concatenation)
# -------------------------------------------------------------------------
def tdt_custom_transform_with_concat(data, comp_sizes, grouping_config, num_k=5):
    arr = np.asarray(data, dtype=np.uint8)
    b = arr.tobytes()

    components = split_bytes_into_components(b, comp_sizes)
    grouped = group_components(components, grouping_config)

    group_entropy = {str(idx): [] for idx in range(len(grouped))}
    for idx, grp in enumerate(grouped):
        for k in range(num_k):
            e_k = compute_kth_order_entropy(grp, k)
            group_entropy[str(idx)].append(e_k)

    # Concatenate all groups
    merged_bytes = b"".join([g.tobytes() for g in grouped])
    final_data = np.frombuffer(merged_bytes, dtype=np.uint8)
    return final_data, group_entropy, grouped


# -------------------------------------------------------------------------
# Custom TDT (NO concatenation) => "TDT-after-Custom"
# -------------------------------------------------------------------------
def tdt_custom_transform_no_concat(data, comp_sizes, grouping_config, num_k=5):
    arr = np.asarray(data, dtype=np.uint8)
    b = arr.tobytes()

    components = split_bytes_into_components(b, comp_sizes)
    grouped = group_components(components, grouping_config)

    group_entropy = {str(idx): [] for idx in range(len(grouped))}
    for idx, grp in enumerate(grouped):
        for k in range(num_k):
            e_k = compute_kth_order_entropy(grp, k)
            group_entropy[str(idx)].append(e_k)

    return grouped, group_entropy


# -------------------------------------------------------------------------
# Example compConfigMap
# -------------------------------------------------------------------------
compConfigMap = {
    "acs_wht_f32": [[[1, 2], [3], [4]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]]],
    "wave_f32": [[[1, 2], [3], [4]]],
    "hdr_night_f32": [
        [[1, 4], [2], [3]],

    ],
    "ts_gas_f32": [[[1, 2], [3], [4]], ],
    "solar_wind_f32": [[[1], [2, 3], [4]], ],
    "tpch_lineitem_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_web_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_store_f32": [
        [[1, 2, 3], [4]],

    ],
    "tpcds_catalog_f32": [
        [[1, 2, 3], [4]],

    ],
    "citytemp_f32": [
        [[1, 4], [2, 3]],

    ],
    "hst_wfc3_ir_f32": [
        [[1, 2], [3], [4]],

    ],
    "hst_wfc3_uvis_f32": [
        [[1, 2], [3], [4]],

    ],
    "rsim_f32": [
        [[1, 2, 3], [4]],

    ],
    "astro_mhd_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "astro_pt_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "jane_street_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "msg_bt_f64": [
        [[1, 2, 3, 4, 5], [6], [7], [8]],

    ],
    "num_brain_f64": [
        [[3, 2, 4, 5, 1, 6], [7], [8]],

    ],
    "num_control_f64": [
        [[1, 2, 3, 4, 5, 6], [7], [8]],

    ],
    "nyc_taxi2015_f64": [
        [[7, 4, 6], [5], [3, 2, 1, 8]],

    ],
    "phone_gyro_f64": [
        [[4, 6], [8], [3, 2, 1, 7], [5]],

    ],
    "tpch_order_f64": [
        [[1, 2, 3, 4], [7], [6, 5], [8]],

    ],
    "tpcxbb_store_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],

    ],
    "tpcxbb_web_f64": [
        [[4, 2, 3], [1], [5], [7], [6], [8]],

    ],
    "wesad_chest_f64": [
        [[7, 5, 6], [8, 4, 1, 3, 2]],

    ],
    "default": [[[1], [2], [3], [4]]],
}


# -------------------------------------------------------------------------
def process_single_dataset(
    dataset_path,
    chunk_size=2000,
    data_size=2000,
    num_k=5,
    transform_word_bits=32
):
    """
    Process a single dataset file with:
      - Original
      - Standard TDT
      - Custom TDT (with concat)
      - TDT-after-custom (no concat) -- NOW includes per-group entropies
    Saves a CSV + plot named after the dataset.
    """

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    try:
        data_loaded = pd.read_csv(dataset_path, sep="\t")
        sliced_data = data_loaded.values[:, 1].astype(np.float32)
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return

    # Crop data
    data = sliced_data[49151:data_size]
    data = np.float32(data)

    # Convert to bytes
    data_bytes = data.view(dtype=np.uint8)
    num_blocks = len(data_bytes) // chunk_size

    # Prepare results
    orig_kth = {str(k): [] for k in range(num_k)}
    orig_fw8 = []

    tdt_kth = {str(k): [] for k in range(num_k)}
    tdt_fw8 = []

    custom_kth = {str(k): [] for k in range(num_k)}
    custom_fw8 = []

    # TDT-after-custom (no concat)
    # We'll store both group entropies and final weighted average
    tdt_after_custom_kth = {str(k): [] for k in range(num_k)}
    tdt_after_custom_fw8 = []

    # We will also store group-level results separately for each block:
    # tdt_after_custom_k{k}_G{g} and tdt_after_custom_fw_8bit_G{g}
    # So let's define a structure to hold them. We'll fill them dynamically
    # because the number of groups can vary by dataset's config.
    tdt_after_custom_kth_groups = []
    tdt_after_custom_fw8_groups = []

    # pick grouping config
    if dataset_name in compConfigMap:
        grouping_configs = compConfigMap[dataset_name]
        grouping_config = grouping_configs[0]
    else:
        grouping_config = compConfigMap["default"][0]

    # Decide how many components for custom
    # (Typically 4 for 32-bit, 8 for 64-bit, etc.)
    if transform_word_bits == 32:
        comp_sizes = [1, 1, 1, 1]
    elif transform_word_bits == 64:
        comp_sizes = [1, 1, 1, 1, 1, 1, 1, 1]
    else:
        comp_sizes = [1, 1, 1, 1]

    # ---------------------------------------------------------------------
    # Process each block
    # ---------------------------------------------------------------------
    for i in range(num_blocks):
        block = data_bytes[i * chunk_size : (i + 1) * chunk_size]

        # -------- 1) Original --------
        for k in range(num_k):
            e_k = compute_kth_order_entropy(block, k)
            orig_kth[str(k)].append(e_k)
        fw_e, _ = compute_entropy_w(block, 8)
        orig_fw8.append(fw_e)

        # -------- 2) Standard TDT --------
        tdt_block = tdt_transform(block, word_bits=transform_word_bits)
        for k in range(num_k):
            e_k = compute_kth_order_entropy(tdt_block, k)
            tdt_kth[str(k)].append(e_k)
        fw_e_tdt, _ = compute_entropy_w(tdt_block, 8)
        tdt_fw8.append(fw_e_tdt)

        # -------- 3) Custom TDT (with concat) --------
        custom_final, custom_grp_entropy, custom_groups = tdt_custom_transform_with_concat(
            tdt_block, comp_sizes, grouping_config, num_k=num_k
        )
        for k in range(num_k):
            e_k = compute_kth_order_entropy(custom_final, k)
            custom_kth[str(k)].append(e_k)
        fw_e_custom, _ = compute_entropy_w(custom_final, 8)
        custom_fw8.append(fw_e_custom)

        # -------- 4) TDT-after-Custom (no concat) --------
        #     Now we also store group-level entropies and the final weighted average
        groups_no_concat, grp_entropy_noc = tdt_custom_transform_no_concat(
            tdt_block, comp_sizes, grouping_config, num_k=num_k
        )
        group_sizes = [len(g) for g in groups_no_concat]
        total_sz = sum(group_sizes)
        weights = [sz / total_sz for sz in group_sizes]

        group_fw_ents = []
        group_kth_ents = {str(k): [] for k in range(num_k)}

        for grpdata in groups_no_concat:
            # Optionally apply TDT again to each group (user's choice).
            # The code as given suggests we do TDT again on each group for consistency:
            tdt_g = tdt_transform(grpdata, word_bits=transform_word_bits)

            # per-group k-th order
            for k in range(num_k):
                e_k = compute_kth_order_entropy(tdt_g, k)
                group_kth_ents[str(k)].append(e_k)

            # per-group fw-8bit
            fw_e_g, _ = compute_entropy_w(tdt_g, 8)
            group_fw_ents.append(fw_e_g)

        # Weighted average across groups => final "tdt_after_custom"
        final_fw = sum(w * e for w, e in zip(weights, group_fw_ents))
        tdt_after_custom_fw8.append(final_fw)

        # For each K, get weighted average across groups
        final_k_vals = {}
        for k in range(num_k):
            final_k = sum(
                w * e for w, e in zip(weights, group_kth_ents[str(k)])
            )
            tdt_after_custom_kth[str(k)].append(final_k)
            final_k_vals[k] = final_k

        # Now store group-level data so we can write it to the CSV
        tdt_after_custom_kth_groups.append(group_kth_ents)
        tdt_after_custom_fw8_groups.append(group_fw_ents)

    # ---------------------------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------------------------
    # ...
    rows = []
    for i in range(num_blocks):
        row = {
            "block": i,
            "dataset_name": dataset_name,  # <--- Add the dataset name here
        }
        # Original
        for k in range(num_k):
            row[f"orig_k{k}"] = orig_kth[str(k)][i]
        row["orig_fw_8bit"] = orig_fw8[i]

        # Standard TDT
        for k in range(num_k):
            row[f"tdt_k{k}"] = tdt_kth[str(k)][i]
        row["tdt_fw_8bit"] = tdt_fw8[i]

        # Custom TDT (with concat)
        for k in range(num_k):
            row[f"custom_k{k}"] = custom_kth[str(k)][i]
        row["custom_fw_8bit"] = custom_fw8[i]

        # TDT-after-Custom (no concat) - Weighted average
        for k in range(num_k):
            row[f"tdt_after_custom_k{k}"] = tdt_after_custom_kth[str(k)][i]
        row["tdt_after_custom_fw_8bit"] = tdt_after_custom_fw8[i]

        # TDT-after-Custom (no concat) - Per group
        per_block_group_kth = tdt_after_custom_kth_groups[i]
        per_block_group_fw = tdt_after_custom_fw8_groups[i]

        # Add one column per group for each k
        for g, fw_g in enumerate(per_block_group_fw):
            row[f"tdt_after_custom_fw_8bit_G{g}"] = fw_g

        # Now fill in the kth entropies
        for k in range(num_k):
            for g, e_k_g in enumerate(per_block_group_kth[str(k)]):
                row[f"tdt_after_custom_k{k}_G{g}"] = e_k_g

        rows.append(row)
    # ...

    df = pd.DataFrame(rows)

    # CSV / plot filenames


    out_csv = f"/home/jamalids/Documents/combined_entropy_results_8bit_{dataset_name}.csv"

    df.to_csv(out_csv, index=False)
    print(f"[{dataset_name}] => Saved results to {out_csv}")

    # Make plot
    ks = np.arange(num_k)
    avg_orig = np.array([df[f"orig_k{k}"].mean() for k in ks])
    avg_tdt = np.array([df[f"tdt_k{k}"].mean() for k in ks])
    avg_custom = np.array([df[f"custom_k{k}"].mean() for k in ks])
    avg_after = np.array([df[f"tdt_after_custom_k{k}"].mean() for k in ks])

    ratio_tdt = avg_orig / avg_tdt
    ratio_custom = avg_orig / avg_custom
    ratio_after = avg_orig / avg_after

    all_ratios = np.concatenate([ratio_tdt, ratio_custom, ratio_after])
    y_max = np.max(all_ratios) * 1.1
    y_min = 0.0

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(ks, ratio_tdt, marker='o')
    axs[0].set_title("orig_k / standard_tdt_k")
    axs[0].set_xlabel("k")
    axs[0].set_ylabel("Ratio")
    axs[0].set_ylim(y_min, y_max)
    axs[0].grid(True)

    axs[1].plot(ks, ratio_custom, marker='o')
    axs[1].set_title("orig_k / custom_tdt_k")
    axs[1].set_xlabel("k")
    axs[1].set_ylabel("Ratio")
    axs[1].set_ylim(y_min, y_max)
    axs[1].grid(True)

    axs[2].plot(ks, ratio_after, marker='o')
    axs[2].set_title("orig_k / tdt_after_custom_k")
    axs[2].set_xlabel("k")
    axs[2].set_ylabel("Ratio")
    axs[2].set_ylim(y_min, y_max)
    axs[2].grid(True)

    plt.tight_layout()
    plot_name =  f"/home/jamalids/Documents/combined_entropy_results_8bitplot_{dataset_name}.png"
    plt.savefig(plot_name)
    print(f"[{dataset_name}] => Saved plot to {plot_name}")
    plt.close(fig)


def main():
    chunk_size = 65536
    data_size = 16384 *4
    num_k = 5
    transform_word_bits = 32  # default

    if len(sys.argv) < 2:
        print("Missing dataset path. Exiting.")
        return

    dataset_path = sys.argv[1]

    # If second arg is chunk_size
    if len(sys.argv) >= 3:
        chunk_size = int(sys.argv[2])

    # If third arg is data_size
    if len(sys.argv) >= 4:
        data_size = int(sys.argv[3])

    # If fourth arg is word_bits (32 or 64)
    if len(sys.argv) >= 5:
        transform_word_bits = int(sys.argv[4])  # 32 or 64 typically

    if os.path.isdir(dataset_path):
        all_files = sorted(os.listdir(dataset_path))
        for fname in all_files:
            full_path = os.path.join(dataset_path, fname)
            if os.path.isfile(full_path):
                process_single_dataset(full_path, chunk_size, data_size, num_k,
                                       transform_word_bits=transform_word_bits)
    else:
        # single file
        process_single_dataset(dataset_path, chunk_size, data_size, num_k,
                               transform_word_bits=transform_word_bits)


if __name__ == "__main__":
    main()
