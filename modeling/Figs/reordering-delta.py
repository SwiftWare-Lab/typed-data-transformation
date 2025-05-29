import os
import math
import numpy as np
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import davies_bouldin_score
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr           # KL divergence
import seaborn as sns
import matplotlib.pyplot as plt


# Replace this import with your actual fastlz_compress location
from modeling.compression_tools import fastlz_compress, huffman_compress,zstd_comp,zlib_comp,bz2_comp,snappy_comp

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
        # Or put `first` in its own subset
        yield [[first]] + smaller

from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations
from modeling.utils import generate_partitions  # make sure this is defined or imported

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

def normalize_cluster_config(cluster_dict):
    """Normalize cluster dict or list to a hashable, sorted structure."""
    return tuple(sorted(tuple(sorted(g)) for g in cluster_dict.values()))

def normalize_partition(partition_list):
    """Same as normalize_cluster_config but from generate_partitions list of lists."""
    return tuple(sorted(tuple(sorted(g)) for g in partition_list))

def _hist256(vec: np.ndarray) -> np.ndarray:
    """Normalised 256-bin histogram (float64)."""
    return np.bincount(vec, minlength=256) / vec.size

def _dunn_index(D, labels):
    """
    D: full distance matrix (square), labels: 1-D array of ints.
    Returns Dunn index (∞ if every cluster has ≤1 point).
    """
    u = np.unique(labels)
    intra = 0.0
    inter = np.inf
    for i in u:
        idx_i = np.where(labels == i)[0]
        if idx_i.size > 1:
            intra = max(intra, D[np.ix_(idx_i, idx_i)].max())
        else:
            intra = max(intra, 0.0)               # single-point cluster → 0
        for j in u:
            if i >= j:
                continue
            idx_j = np.where(labels == j)[0]
            inter = min(inter, D[np.ix_(idx_i, idx_j)].min())
    return np.inf if intra == 0 else inter / intra
# ------------------------------------------------------------------
# EXTRA HELPERS for match-gain vs block-entropy gain
# ------------------------------------------------------------------
def delta_k_entropy(global_stream, cluster_streams, k=2):
    """Return H_k(global) − weighted avg H_k(cluster_i)."""
    H_global = compute_kth_entropy(global_stream, k)
    total    = len(global_stream)
    H_weight = sum(len(c) * compute_kth_entropy(c, k) for c in cluster_streams) / total
    return H_global - H_weight          # >0 means decomposition helps

def delta_H0(global_stream, cluster_streams):
    """0-order cross-entropy gain (same as H0 drop)."""
    H0_global = compute_entropy(global_stream)
    total     = len(global_stream)
    H0_weight = sum(len(c) * compute_entropy(c) for c in cluster_streams) / total
    return H0_global - H0_weight
# Rewriting the `test_synthetic_all_modes` function to compute all metrics after reordering only

def test_synthetic_reorder_only(SIZE=1024, ENT=[6, 2, 4, 5], mode="frequency", compress_method=None, comp_name=""):
    if compress_method is None:
        compress_method = fastlz_compress
        comp_name = "FastLZ"

    packed, groups = generate_float_stream(SIZE, ENT)
    byte_groups = groups
    orig_size = len(packed)
    records = []
    global_H1 = round(compute_entropy(packed), 4)
    global_H2 = round(compute_kth_entropy(packed, 2), 4)
    wins_g = [compute_entropy(packed[i:i + 256]) for i in range(0, len(packed), 256)]
    avg_within_global = float(np.mean(wins_g))
    avg_within_std_global = float(np.std(wins_g))
    # Step 1: Collect normalized HClust configurations
    feats = np.vstack([extract_features(g, mode) for g in byte_groups])
    linked = linkage(feats, method='complete')
    hclust_configs = set()

    for k in range(1, 5):
        labels = np.ones(4, int) if k == 1 else fcluster(linked, k, criterion='maxclust')
        clusters = {lab: [i for i, l in enumerate(labels) if l == lab] for lab in sorted(set(labels))}
        hclust_configs.add(normalize_cluster_config(clusters))

    for cluster_config in generate_partitions([0, 1, 2, 3]):
        normalized = normalize_partition(cluster_config)
        cfg = "|".join(f"({','.join(map(str, group))})" for group in cluster_config)
        label_type = "HClust" if normalized in hclust_configs else "Non-HClust"

        all_row_bytes_C, all_row_bytes_F = [], []

        for idxs in cluster_config:
            arr2d = np.stack([byte_groups[i] for i in idxs], axis=1)
            row_bytes_C = transform_data([arr2d], order='C')
            row_bytes_F = transform_data([arr2d], order='F')
            all_row_bytes_C.append(row_bytes_C)
            all_row_bytes_F.append(row_bytes_F)

        reordered_stream_C = np.frombuffer(b"".join(all_row_bytes_C), dtype=np.uint8)
        reordered_stream_F = np.frombuffer(b"".join(all_row_bytes_F), dtype=np.uint8)

        r_std = ratio(orig_size, compress_method(packed.tobytes()))
        r_re_row_C = ratio(orig_size, compress_method(reordered_stream_C.tobytes()))
        r_re_row_F = ratio(orig_size, compress_method(reordered_stream_F.tobytes()))

        # Compute entropy metrics after reordering (C-order)
        HC_H1 = [round(compute_entropy(reordered_stream_C), 4)]

        HC_H2 = [round(compute_kth_entropy(reordered_stream_C, 2), 4)]
        print("H2: ", HC_H2)
        HC_H3 = [round(compute_kth_entropy(reordered_stream_C, 3), 4)]
        HC_H4 = [round(compute_kth_entropy(reordered_stream_C, 4), 4)]

        # HC_H1_F = [round(compute_entropy(reordered_stream_F), 4)]
        # HC_H2_F= [round(compute_kth_entropy(reordered_stream_F, 2), 4)]
        total_bytes = len(reordered_stream_C)

        cluster_streams = [reordered_stream_C]
        jh = compute_joint_entropy(cluster_streams)
        mi = max(compute_entropy(reordered_stream_C) - jh, 0.0)
        kth = HC_H2[0]

        wins = [compute_entropy(reordered_stream_C[i:i + 256]) for i in range(0, len(reordered_stream_C), 256)]
        avg_within = float(np.mean(wins))
        avg_within_std = float(np.std(wins))
        between = 0.0
        #################################
        wins_k2 = [compute_kth_entropy(reordered_stream_C[i:i + 256], k=2)
                   for i in range(0, len(reordered_stream_C), 256)]
        avg_within_k2 = float(np.mean(wins_k2))
        avg_within_k2_std = float(np.std(wins_k2))

        lane_labels = np.zeros(feats.shape[0], dtype=int)
        for cid, idxs in enumerate(cluster_config):
            lane_labels[list(idxs)] = cid
        D_feat = squareform(pdist(feats))
        dunn_val = _dunn_index(D_feat, lane_labels)

        # Divergence
        hists = [_hist256(c) for c in cluster_streams]
        ce_vals, js_vals = [], []
        for i, j in combinations(range(len(hists)), 2):
            p, q = hists[i], hists[j]
            ce_vals.append(-(p * np.log2(q + 1e-12)).sum())
            m = 0.5 * (p + q)
            kl_pm = rel_entr(p, m).sum() / np.log(2)
            kl_qm = rel_entr(q, m).sum() / np.log(2)
            js_vals.append(0.5 * (kl_pm + kl_qm))

        avg_ce = float(np.mean(ce_vals)) if ce_vals else float("nan")
        avg_js = float(np.mean(js_vals)) if js_vals else float("nan")

        num_clusters = len(cluster_config)
        n_samples = feats.shape[0]
        if 1 < num_clusters < n_samples:
            db_index = davies_bouldin_score(feats, lane_labels)
        else:
            db_index = float("nan")

        rec = {
            "Dataset": "synthetic",
            "CompressionTool": comp_name,
            "FeatureMode": mode,
            "ClusterConfig": cfg,
            "ConfigType": label_type,
            "HC_H1": ",".join(map(str, HC_H1)),
            "HC_H2": ",".join(map(str, HC_H2)),
            "HC_H3": ",".join(map(str, HC_H3)),
            "HC_H4": ",".join(map(str, HC_H4)),
            "WithinSTD": avg_within_std,
            "WithinSTD_global": avg_within_std_global,

            "Global_H1": global_H1,
            "Global_H2": global_H2,
            "Delta_H2": global_H2 - HC_H2[0],
            "Delta_H1": global_H1 - HC_H1[0],
            # "HC_H1_F": ",".join(map(str, HC_H1_F)),
            # "HC_H2_F": ",".join(map(str, HC_H2_F)),
            "StandardRatio": r_std,
            "ReorderedRatio_Row_C": r_re_row_C,
            "ReorderedRatio_Row_F": r_re_row_F,
            "JointEntropy": jh,
            "MutualInfo": mi,
            "KthEntropy": float(kth),
            "BetweenSTD": between,
            "DaviesBouldin": db_index,
            "Dunn": dunn_val,
            "WithinSTD_H2": avg_within_k2_std,
            "WithinMean_H2": avg_within_k2,

        }
        records.append(rec)

    df = pd.DataFrame(records)
    out_csv = f"synthetic_reorder_only_{comp_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Written {out_csv}")
    return df

###################################
def plot_corr_to_ratios_re(df,
    metrics=(
        "HC_H2",
        "HC_H3",
       # "HC_H4",
    ),
    ratio_cols=("ReorderedRatio_Row_C",),
    codec_tag="FastLZ",
    save_dir="jamalids/Downloads"):
    """
    Compute Pearson correlations and save a heat-map.
    Returns the PNG path.
    """

    # ---------- Build a tidy DataFrame of correlations ----------
    rows = []
    for m in metrics:
        for r in ratio_cols:
            rho = df[m].corr(df[r])
            rows.append((m, r, 0.0 if np.isnan(rho) else rho))

    corr_df = pd.DataFrame(rows, columns=["Metric", "Ratio", "ρ"])

    # ---------- Pivot to wide format ----------
    wide = corr_df.pivot(index="Metric", columns="Ratio", values="ρ")

    # ---------- Define your aliases ----------
    metric_alias = {
        "HC_H2": "2nd reordered entropy",
        "HC_H3": "3rd reordered entropy",
        #"HC_H4": "4th reordered entropy",
    }
    ratio_alias = {
        "ReorderedRatio_Row_C": "Reordered compression ratio",
    }

    # ---------- Apply renaming ----------
    wide = wide.rename(index=metric_alias, columns=ratio_alias)

    # ---------- Print correlation values ----------
    print(f"\n=== Pearson correlations – {codec_tag} ===")
    print(wide.round(3).to_string())

    # ---------- Heatmap -----------------------------------------
    plt.figure(figsize=(6, 0.6 * len(metrics) + 1))
    sns.heatmap(wide, annot=True, fmt=".2f",
                center=0, cmap="vlag", cbar_kws=dict(label="ρ"))
    plt.title(f"{codec_tag}: Correlation with Reordered Ratios")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join( f"corr_ratios_re_{codec_tag.lower()}.png")
    plt.savefig(f"corr_ratios_{codec_tag.lower()}.png", dpi=300)
    plt.close()
    print("saved →", png_path)

    return png_path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

def plot_corr_and_p_side_by_side(
    dfs, tags,
    metrics=("HC_H2", "HC_H3","Delta_H1","Global_H1","HC_H1"),
    ratio_col="ReorderedRatio_Row_C",
    save_path="/home/jamalids/Documents/corr_and_p_reordering.png"
):
    # human-readable labels for the metrics
    metric_labels = {
        "HC_H2": "2nd order entropy",
        "HC_H3": "3rd order entropy",

    }

    # 1) Build DataFrames of r and p
    r_mat = pd.DataFrame(index=metrics, columns=tags, dtype=float)
    p_mat = pd.DataFrame(index=metrics, columns=tags, dtype=float)
    for df, tag in zip(dfs, tags):
        for m in metrics:
            r, p = pearsonr(df[m].astype(float), df[ratio_col].astype(float))
            r_mat.at[m, tag] = r
            p_mat.at[m, tag] = p

    # 2) Plot side by side
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, 0.6 * len(metrics) + 1),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.4}
    )

    # ρ heatmap
    sns.heatmap(
        r_mat.rename(index=metric_labels),
        annot=True, fmt=".2f",
        cmap="vlag", center=0,
        cbar_kws={"label": "ρ"},
        ax=ax1
    )
    ax1.set_title("Pearson ρ", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # p-value heatmap
    sns.heatmap(
        p_mat.rename(index=metric_labels),
        annot=True, fmt=".2g",
        cmap="YlGnBu", center=0.05,
        cbar_kws={"label": "p-value"},
        ax=ax2
    )
    ax2.set_title("p-value", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    plt.suptitle("Correlation and Significance", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved side-by-side ρ and p-value plot → {save_path}")

if __name__=="__main__":
   # recs = test_synthetic_all_modes()
   df_result = test_synthetic_reorder_only(compress_method=fastlz_compress, comp_name="FastLZ")
   plot_corr_to_ratios_re(df_result, codec_tag="FastLZ")
   # df_result = test_synthetic_reorder_only(compress_method= huffman_compress, comp_name="Huffman")
   # plot_corr_to_ratios_re(df_result, codec_tag="Huffman")
   df_zstd = test_synthetic_reorder_only(compress_method=zstd_comp, comp_name="Zstd")
   plot_corr_to_ratios_re(df_result, codec_tag="Zstd")
   plot_corr_and_p_side_by_side(
       [df_result, df_zstd],
       ["FastLZ", "Zstd"]
   )