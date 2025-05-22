import os
import math
import numpy as np
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

from scipy.cluster.hierarchy import linkage, fcluster
from compression_tools import fastlz_compress, huffman_compress,zstd_comp,zlib_comp,bz2_comp

# ---------------------- SYNTHETIC DATA GENERATION ---------------------- #

def generate_byte_stream(size, entropy):
    num_symbols = int(2**entropy)
    p = np.ones(num_symbols)/num_symbols
    return np.random.choice(num_symbols, size=size, p=p).astype(np.uint8)

def generate_float_stream(size, entropies):
    if len(entropies) != 4:
        raise ValueError("Need exactly four byte‐entropies")
    comps = [generate_byte_stream(size, e) for e in entropies]
    packed = np.zeros(size*4, dtype=np.uint8)
    for i in range(size):
        packed[4*i:4*i+4] = [c[i] for c in comps]
    return packed, comps

# ---------------------- ENTROPY & MI HELPERS ---------------------- #

def compute_entropy(arr):
    freq, total = Counter(arr), len(arr)
    return -sum((cnt/total)*math.log2(cnt/total) for cnt in freq.values() if cnt>0)

def compute_joint_entropy(groups):
    pairs, total = list(zip(*groups)), len(groups[0])
    freq = Counter(pairs)
    return -sum((cnt/total)*math.log2(cnt/total) for cnt in freq.values())

def compute_kth_entropy(arr, k=2):
    if len(arr)<k: return 0.0
    grams = [tuple(arr[i:i+k]) for i in range(len(arr)-k+1)]
    freq, total = Counter(grams), len(grams)
    return -sum((cnt/total)*math.log2(cnt/total) for cnt in freq.values())

def mutual_information_pair(x,y):
    Hx = compute_entropy(x)
    Hy = compute_entropy(y)
    Hxy= compute_joint_entropy([x,y])
    return max(Hx+Hy-Hxy,0.0)

# ---------------------- FEATURE EXTRACTORS ---------------------- #

def extract_entropy(byte_group, window=65536):
    ent = [compute_entropy(byte_group[i:i+window])
           for i in range(0, len(byte_group), window)]
    return np.array([np.mean(ent), np.std(ent), np.max(ent), np.min(ent)])

def extract_frequency(byte_group, *args):
    freq = Counter(byte_group)
    probs = np.array([freq.get(b,0)/len(byte_group) for b in range(256)])
    return np.array([np.std(probs)])

def extract_all(byte_group, window=65536):
    return np.concatenate((extract_entropy(byte_group, window),
                           extract_frequency(byte_group)))

def extract_features(byte_group, mode="frequency", window=65536):
    if mode=="entropy":   return extract_entropy(byte_group, window)
    if mode=="frequency": return extract_frequency(byte_group)
    if mode=="all":       return extract_all(byte_group, window)
    raise ValueError(f"Unknown mode: {mode}")

# ---------------------- COMPRESSION UTIL ---------------------- #



def ratio(orig_size, comp_bytes):
    return orig_size / len(comp_bytes) if len(comp_bytes)>0 else float('inf')

def interleave_bytes(arrs):
    maxlen = max(len(a) for a in arrs)
    out = bytearray()
    for i in range(maxlen):
        for a in arrs:
            if i < len(a):
                out.append(int(a[i]))
    return bytes(out)
#----------------------------------------------------------
def transform_data(data_set_list, order='C'):
    # view data_set as bytes
    compressed_data = []
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_data.append(data_set_bytes)
    # flatten the list
    compressed_data = np.concatenate(compressed_data, axis=0)
    return compressed_data

def generate_partitions(elements):
    if len(elements) == 1:
        yield [elements]
        return
    first = elements[0]
    for smaller in generate_partitions(elements[1:]):
        # Insert `first` in each position of each subset
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller



def normalize_cluster_config(cluster_dict):
    return tuple(sorted(tuple(sorted(g)) for g in cluster_dict.values()))

def normalize_partition(partition_list):
    return tuple(sorted(tuple(sorted(g)) for g in partition_list))

def test_synthetic_all_modes(SIZE=1024, ENT=[3, 5, 2, 1], mode="frequency", compress_method=None, comp_name=""):

    if compress_method is None:
        compress_method = fastlz_compress
        comp_name = "FastLZ"

    packed, groups = generate_float_stream(SIZE, ENT)
    arr = packed
    byte_groups = groups
    orig_size = len(arr)
    records = []

    # Step 1: Collect normalized HClust configurations
    feats = np.vstack([extract_features(g, mode) for g in byte_groups])
    linked = linkage(feats, method='complete')
    hclust_configs = set()

    for k in range(1, 5):
        labels = np.ones(4, int) if k == 1 else fcluster(linked, k, criterion='maxclust')
        clusters = {lab: [i for i, l in enumerate(labels) if l == lab] for lab in sorted(set(labels))}
        hclust_configs.add(normalize_cluster_config(clusters))

    # Step 2: Test each possible clustering
    for cluster_config in generate_partitions([0, 1, 2, 3]):
        normalized = normalize_partition(cluster_config)
        cfg = "|".join(f"({','.join(map(str, group))})" for group in cluster_config)
        label_type = "HClust" if normalized in hclust_configs else "Non-HClust"

        r_std = ratio(orig_size, compress_method(arr.tobytes()))
        dec_sizes_row_C, dec_sizes_row_F = [], []
        all_row_bytes_C, all_row_bytes_F = [], []

        for idxs in cluster_config:
            arr2d = np.stack([byte_groups[i] for i in idxs], axis=1)
            row_bytes_C = transform_data([arr2d], order='C')
            row_bytes_F = transform_data([arr2d], order='F')
            dec_sizes_row_C.append(len(compress_method(row_bytes_C)))
            dec_sizes_row_F.append(len(compress_method(row_bytes_F)))
            all_row_bytes_C.append(row_bytes_C)
            all_row_bytes_F.append(row_bytes_F)

        r_dec_row_C = orig_size / sum(dec_sizes_row_C)
        r_dec_row_F = orig_size / sum(dec_sizes_row_F)
        r_re_row_C = ratio(orig_size, compress_method(b"".join(all_row_bytes_C)))
        r_re_row_F = ratio(orig_size, compress_method(b"".join(all_row_bytes_F)))

        # Info metrics
        cluster_streams = []
        HC_H1, HC_H2 = [], []
        total_bytes = 0
        weighted_entropy, weighted_kth_entropy = 0.0, 0.0

        for idxs in cluster_config:
            arr2d = np.stack([byte_groups[i] for i in idxs], axis=1)
            flat_stream = transform_data([arr2d], order='F')
            flat = np.frombuffer(flat_stream, dtype=np.uint8)
            cluster_streams.append(flat)

            H1_val = compute_entropy(flat)
            H2_val = compute_kth_entropy(flat, 2)
            HC_H1.append(round(H1_val, 4))
            HC_H2.append(round(H2_val, 4))

            size = len(flat)
            total_bytes += size
            weighted_entropy += H1_val * size
            weighted_kth_entropy += H2_val * size

        jh = compute_joint_entropy(cluster_streams)
        mi = max(sum(compute_entropy(c) for c in cluster_streams) - jh, 0.0)
        kth = np.mean([compute_kth_entropy(c, 2) for c in cluster_streams])

        cluster_entropy_means, cluster_entropy_stds = [], []
        for c in cluster_streams:
            wins = [compute_entropy(c[i:i + 256]) for i in range(0, len(c), 256)]
            cluster_entropy_means.append(np.mean(wins))
            cluster_entropy_stds.append(np.std(wins))

        avg_within = float(np.mean(cluster_entropy_means))
        avg_within_std = float(np.mean(cluster_entropy_stds))
        between = float(np.std(cluster_entropy_means))

        rec = {
            "Dataset": "synthetic",
            "CompressionTool": comp_name,
            "FeatureMode": mode,
            "ClusterConfig": cfg,
            "ConfigType": label_type,
            "HC_H1": ",".join(map(str, HC_H1)),
            "HC_H2": ",".join(map(str, HC_H2)),
            "HC_H1_weighted": round(weighted_entropy / total_bytes, 5),
            "HC_H2_weighted": round(weighted_kth_entropy / total_bytes, 5),
            "BetweenClusterEntropySTD": between,
            "StandardRatio": r_std,
            "DecomposedRatio_Row_C": r_dec_row_C,
            "DecomposedRatio_Row_F": r_dec_row_F,
            "ReorderedRatio_Row_C": r_re_row_C,
            "ReorderedRatio_Row_F": r_re_row_F,
            "JointEntropy": jh,
            "MutualInfo": mi,
            "KthEntropy": float(kth),
            "WithinSTD": avg_within_std,
            "BetweenSTD": between,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(f"synthetic_all_partitions_{comp_name}.csv", index=False)
    print(f"Written synthetic_all_partitions_{comp_name}.csv")
    return df

###################################

def plot_mutual_and_joint_vs_ratio(df, comp_tool="FastLZ", ds_name="Synthetic"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ratio_col = "DecomposedRatio_Row_F"

    palette = {"HClust": "red", "Non-HClust": "blue"}

    # --- Plot 1: MutualInfo vs Ratio ---
    sns.scatterplot(data=df, x="MutualInfo", y=ratio_col, hue="ConfigType", palette=palette, ax=ax1)
    sns.regplot(data=df, x="MutualInfo", y=ratio_col, scatter=False, color='black', ax=ax1)
    ax1.set_title(f"{ds_name} : {comp_tool} — MutualInfo vs Ratio")
    ax1.set_xlabel("Mutual Information (bits)")
    ax1.set_ylabel("Decomposed Compression Ratio (Row-F)")

    # Annotate HClust points with their ClusterConfig
    for _, row in df[df["ConfigType"] == "HClust"].iterrows():
        ax1.text(row["MutualInfo"], row[ratio_col], row["ClusterConfig"], fontsize=8, color="black",
                 verticalalignment='bottom', horizontalalignment='left', rotation=20)

    r2_mi = stats.linregress(df["MutualInfo"], df[ratio_col]).rvalue ** 2
    ax1.text(0.05, 0.95, f"R² = {r2_mi:.2f}", transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', color='black')

    # --- Plot 2: JointEntropy vs Ratio ---
    sns.scatterplot(data=df, x="JointEntropy", y=ratio_col, hue="ConfigType", palette=palette, ax=ax2)
    sns.regplot(data=df, x="JointEntropy", y=ratio_col, scatter=False, color='black', ax=ax2)
    ax2.set_title(f"{ds_name} : {comp_tool} — JointEntropy vs Ratio")
    ax2.set_xlabel("Joint Entropy (bits)")
    ax2.set_ylabel("Decomposed Compression Ratio (Row-F)")

    # Annotate HClust points with ClusterConfig
    for _, row in df[df["ConfigType"] == "HClust"].iterrows():
        ax2.text(row["JointEntropy"], row[ratio_col], row["ClusterConfig"], fontsize=8, color="black",
                 verticalalignment='bottom', horizontalalignment='left', rotation=20)

    r2_jh = stats.linregress(df["JointEntropy"], df[ratio_col]).rvalue ** 2
    ax2.text(0.05, 0.95, f"R² = {r2_jh:.2f}", transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', color='black')

    plt.tight_layout()
    plot_path = f"{comp_tool.lower()}.png"

    plt.savefig(plot_path)
    plt.close()
    return plot_path




def plot_many_vs_ratio(
    df,
    comp_tool="FastLZ",
    ds_name="Synthetic",
    save_dir="/mnt/c/Users/jamalids/Downloads/",
):

    ratio_col = "DecomposedRatio_Row_F"
    base_line = df["StandardRatio"].iloc[0]
    palette   = {"HClust": "red", "Non-HClust": "blue"}


    x_cols = [
        ("MutualInfo",          "Mutual Information (bits)"),
        ("JointEntropy",        "Joint Entropy (bits)"),
        ("HC_H1_weighted",      "Weighted H1"),
        ("HC_H2_weighted",      "Weighted H2"),
        ("WithinSTD",           "Within-Cluster STD"),
        ("BetweenSTD",          "Between-Cluster STD"),
        ("HC_H1_num",           "Mean HC_H1"),
        ("HC_H2_num",           "Mean HC_H2"),
    ]

    if "HC_H1_num" not in df.columns:
        df = df.copy()
        df["HC_H1_num"] = df["HC_H1"].apply(
            lambda s: np.mean(list(map(float, s.split(",")))))
        df["HC_H2_num"] = df["HC_H2"].apply(
            lambda s: np.mean(list(map(float, s.split(",")))))

    # ---- Layout: 2 rows × 4 columns ----------------------------------------
    fig, axes_grid = plt.subplots(2, 4, figsize=(22, 8), sharey=True)
    axes = axes_grid.ravel()
    ncols = 4  # for “left-most column” test

    # ---- Plot each metric ---------------------------------------------------
    for idx, (ax, (xcol, xlabel)) in enumerate(zip(axes, x_cols)):
        # scatter with hue
        sns.scatterplot(data=df, x=xcol, y=ratio_col,
                        hue="ConfigType", palette=palette, ax=ax, legend=False)

        # trend line
        sns.regplot(data=df, x=xcol, y=ratio_col,
                    scatter=False, color="black", ax=ax)

        # baseline
        ax.axhline(base_line, ls="--", lw=1, color="grey", label="StandardRatio")

        # annotate HClust points
        for _, row in df[df["ConfigType"] == "HClust"].iterrows():
            ax.text(row[xcol], row[ratio_col], row["ClusterConfig"],
                    fontsize=8, color="black", rotation=20,
                    va="bottom", ha="left")

        # R² in corner
        r2 = stats.linregress(df[xcol], df[ratio_col]).rvalue ** 2
        ax.text(0.02, 0.94, f"R² = {r2:.2f}", transform=ax.transAxes,
                fontsize=8, va="top", ha="left")

        ax.set_title(xlabel, fontsize=11)
        ax.set_xlabel(xlabel)
        if idx % ncols == 0:
            ax.set_ylabel("Decomposed Ratio (Row-F)")
        else:
            ax.set_ylabel(None)

    fig.suptitle(
        f"{ds_name} – {comp_tool}: Row-F Compression vs Multiple Metrics",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()


    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"rowF_vs_metrics_{comp_tool.lower()}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


##################################
def plot_with_cluster_labels(df, comp_tool="FastLZ", ds_name="Synthetic"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ratio_col = "DecomposedRatio_Row_F"


    sns.scatterplot(data=df, x="MutualInfo", y=ratio_col, ax=ax1)
    sns.regplot(data=df, x="MutualInfo", y=ratio_col, scatter=False, color='red', ax=ax1)
    ax1.set_title(f"{ds_name} : {comp_tool} — MutualInfo vs Ratio")
    ax1.set_xlabel("Mutual Information (bits)")
    ax1.set_ylabel("Decomposed Compression Ratio (Row-F)")
    r2_mi = stats.linregress(df["MutualInfo"], df[ratio_col]).rvalue ** 2
    ax1.text(0.05, 0.95, f"R² = {r2_mi:.2f}", transform=ax1.transAxes, fontsize=10, verticalalignment='top', color='red')


    for i, row in df.iterrows():
        ax1.annotate(row["ClusterConfig"], (row["MutualInfo"], row[ratio_col]), fontsize=8, alpha=0.7)

    sns.scatterplot(data=df, x="JointEntropy", y=ratio_col, ax=ax2)
    sns.regplot(data=df, x="JointEntropy", y=ratio_col, scatter=False, color='blue', ax=ax2)
    ax2.set_title(f"{ds_name} : {comp_tool} — JointEntropy vs Ratio")
    ax2.set_xlabel("Joint Entropy (bits)")
    ax2.set_ylabel("Decomposed Compression Ratio (Row-F)")
    r2_jh = stats.linregress(df["JointEntropy"], df[ratio_col]).rvalue ** 2
    ax2.text(0.05, 0.95, f"R² = {r2_jh:.2f}", transform=ax2.transAxes, fontsize=10, verticalalignment='top', color='blue')

    for i, row in df.iterrows():
        ax2.annotate(row["ClusterConfig"], (row["JointEntropy"], row[ratio_col]), fontsize=8, alpha=0.7)

    plt.tight_layout()
    plot_path = f"/mnt/c/Users/jamalids/Downloads/{comp_tool.lower()}.png"


    plt.savefig(plot_path)
    plt.close()
    return plot_path





if __name__=="__main__":
   # recs = test_synthetic_all_modes()
   df_result = test_synthetic_all_modes()
  # plot_mutual_and_joint_vs_ratio(df_result)
   plot_mutual_and_joint_vs_ratio(df_result, comp_tool="fastlz")
   plot_many_vs_ratio(df_result, comp_tool="FastLZ", ds_name="Synthetic")

   #df_result.to_csv("/mnt/c/Users/jamalids/Downloads/synthetic_fastlz.csv")
    # Run with Zstd
   df_zstd = test_synthetic_all_modes(compress_method=zstd_comp, comp_name="Zstd")
  # df_zstd.to_csv("/mnt/c/Users/jamalids/Downloads/synthetic_zstd.csv")
   plot_mutual_and_joint_vs_ratio(df_zstd, comp_tool="zstd")
   plot_many_vs_ratio(df_zstd, comp_tool="Zstd", ds_name="Synthetic")
    # Run with Huffman
   df_huffman = test_synthetic_all_modes(compress_method=huffman_compress, comp_name="Huffman")
  # df_huffman.to_csv("/mnt/c/Users/jamalids/Downloads/synthetic_huffman.csv")
   plot_mutual_and_joint_vs_ratio(df_huffman, comp_tool="huffman")
   plot_many_vs_ratio(df_huffman, comp_tool="Huffman", ds_name="Synthetic")
   #################################
   df_zlib = test_synthetic_all_modes(compress_method=zlib_comp, comp_name="zlib")
  # df_zlib.to_csv("/mnt/c/Users/jamalids/Downloads/synthetic_zlib.csv")
   plot_mutual_and_joint_vs_ratio(df_zlib, comp_tool="zlib")
   plot_many_vs_ratio(df_zlib, comp_tool="Zlib", ds_name="Synthetic")

   #################################
   df_bzib = test_synthetic_all_modes(compress_method=bz2_comp, comp_name="bzib")
  # df_bzib.to_csv("/mnt/c/Users/jamalids/Downloads/synthetic_bzib.csv")
   plot_mutual_and_joint_vs_ratio(df_bzib, comp_tool="bzib")
   plot_many_vs_ratio(df_bzib, comp_tool="bzib", ds_name="Synthetic")


