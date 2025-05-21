import numpy as np
import math
from collections import Counter
import scipy.stats as stats
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from enum import Enum
from modeling.entropy_analysis import calculate_kth_order_entropy_paper


class CompressionTool(Enum):
    ZSTD = "zstd"
    ZLIB = "zlib"
    BZ2 = "bz2"
    SNAPPY = "snappy"
    FASTLZ = "fastlz"
    HUFFMAN = "huffman"
    RLE = "rle"

compression_tool = CompressionTool.FASTLZ


def calculate_entropy_from_counts(counts, total):
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c)


def calculate_joint_entropy(arrays):
    """
    Compute the joint entropy H(X1,...,Xk) by stacking arrays column-wise.
    """
    stacked = np.stack([a.flatten() for a in arrays], axis=-1)
    unique_rows, counts = np.unique(stacked, axis=0, return_counts=True)
    probs = counts.astype(float) / stacked.shape[0]
    return -np.sum(probs * np.log2(probs))


def calculate_mutual_information(arrays):
    """
    Compute multivariate mutual information I(X1;...;Xk).
    """
    individual_H = []
    flat = [a.flatten() for a in arrays]
    for f in flat:
        counts = Counter(f)
        individual_H.append(calculate_entropy_from_counts(counts, len(f)))
    H_joint = calculate_joint_entropy(arrays)
    return sum(individual_H) - H_joint


def compress_with_zstd(data, level=3):
    if compression_tool == CompressionTool.HUFFMAN:
        from compression_tools import huffman_compress
        return huffman_compress(data), None
    elif compression_tool == CompressionTool.FASTLZ:
        from compression_tools import fastlz_compress
        return fastlz_compress(data), None
    else:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=level)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        return cctx.compress(data), None


def get_compression_ratio_stats(components, partition, original_dataset=None):
    merged = [np.concatenate([components[i] for i in cluster], axis=0)
              for cluster in partition]
    total_bytes = sum(arr.nbytes for arr in merged)
    comp_sizes = []
    for arr in merged:
        comp, _ = compress_with_zstd(arr)
        comp_sizes.append(len(comp))
    decomposed_cr = total_bytes / sum(comp_sizes)
    all_bytes = np.concatenate(merged, axis=0)
    comp_reo, _ = compress_with_zstd(all_bytes)
    reordered_cr = all_bytes.nbytes / len(comp_reo)
    if original_dataset is None:
        original_dataset = np.concatenate(components, axis=0)
    comp_orig, _ = compress_with_zstd(original_dataset)
    original_cr = original_dataset.nbytes / len(comp_orig)
    return decomposed_cr, reordered_cr, original_cr


def generate_partitions(elements):
    if not elements:
        return [[]]
    first, rest = elements[0], elements[1:]
    partitions = []
    for p in generate_partitions(rest):
        for i in range(len(p)):
            partitions.append(p[:i] + [[first] + p[i]] + p[i+1:])
        partitions.append([[first]] + p)
    return partitions


def all_possible_merging(comp_array, ds_name, original_dataset=None):
    stat = []
    for partition in generate_partitions(list(range(len(comp_array)))):
        entry = {"Name": ds_name, "tool": compression_tool, "clustering": partition}
        # joint entropy and mutual information per cluster
        cluster_joint = []
        cluster_mi = []
        for cluster in partition:
            arrays = [comp_array[i] for i in cluster]
            Hc = calculate_joint_entropy(arrays)
            Ic = calculate_mutual_information(arrays)
            cluster_joint.append(Hc)
            cluster_mi.append(Ic)
        entry["cluster_joint_entropies"] = cluster_joint
        entry["sum_joint_entropy"] = sum(cluster_joint)
        entry["avg_joint_entropy"] = sum(cluster_joint)/len(cluster_joint)
        entry["cluster_mutual_info"] = cluster_mi
        entry["sum_mutual_info"] = sum(cluster_mi)
        entry["avg_mutual_info"] = sum(cluster_mi)/len(cluster_mi)
        # compression ratios
        dec_cr, reo_cr, orig_cr = get_compression_ratio_stats(comp_array, partition, original_dataset)
        entry.update({"decomposed cr": dec_cr,
                      "reordered cr": reo_cr,
                      "original cr": orig_cr})
        stat.append(entry)
    return stat


def plot_entropy_vs_cr(stat):
    df = pd.DataFrame(stat)
    sns.scatterplot(data=df, x='avg_joint_entropy', y='decomposed cr')
    plt.xlabel('Average Joint Entropy per Cluster')
    plt.ylabel('Decomposed Compression Ratio')
    plt.title('Entropy vs Decomposed CR')
    plt.show()


def plot_mi_vs_cr(stat):
    df = pd.DataFrame(stat)
    sns.scatterplot(data=df, x='avg_mutual_info', y='decomposed cr')
    plt.xlabel('Average Mutual Information per Cluster')
    plt.ylabel('Decomposed Compression Ratio')
    plt.title('Mutual Information vs Decomposed CR')
    plt.show()


if __name__ == '__main__':
    comp_array = [np.random.randint(0,256, size=1024, dtype=np.uint8) for _ in range(4)]
    stats = all_possible_merging(comp_array, 'Synthetic')
    plot_entropy_vs_cr(stats)
    plot_mi_vs_cr(stats)
