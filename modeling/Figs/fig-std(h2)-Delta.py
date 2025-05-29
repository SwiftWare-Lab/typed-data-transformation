import os
import math
import numpy as np
from collections import Counter

from sklearn.metrics import davies_bouldin_score

from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr           # KL divergence


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

def test_synthetic_all_modes(SIZE=1024,  ENT=[6, 2, 4, 5], mode="frequency", compress_method=None, comp_name=""):


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
            flat_stream = transform_data([arr2d], order='C')
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
        # ---- NEW: delta-metrics ------------------------------------------
        delta_match = delta_k_entropy(arr, cluster_streams, k=2)   # k=2 is what you used
        delta_h0    = delta_H0(arr, cluster_streams)

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
        #-------------------------
        cluster_entropy_means_H2, cluster_entropy_stds_H2 = [], []
        for c in cluster_streams:
            wins = [compute_kth_entropy(c[i:i + 256],2) for i in range(0, len(c), 256)]
            cluster_entropy_means_H2.append(np.mean(wins))
            cluster_entropy_stds_H2.append(np.std(wins))

        avg_within_H2 = float(np.mean(cluster_entropy_means_H2))
        avg_within_std_H2 = float(np.mean(cluster_entropy_stds_H2))
        between_H2 = float(np.std(cluster_entropy_means_H2))
        # ---- Davies–Bouldin index for this ClusterConfig -------------------
        num_clusters = len(cluster_config)
        n_samples = feats.shape[0]  # here: 4

        if 1 < num_clusters < n_samples:  # valid range: 2 .. n_samples-1
            lane_labels = np.zeros(n_samples, dtype=int)
            for cid, idxs in enumerate(cluster_config):
                lane_labels[list(idxs)] = cid
            db_index = davies_bouldin_score(feats, lane_labels)
        else:
            db_index = float("nan")  # undefined for 1 or 4 clusters
        # --------------------------------------------------------------------
        # ------------------------------------------------------------------
        #     A) Dunn index  (works for k = n; ∞ when intra = 0)
        # ------------------------------------------------------------------
        # Represent each lane by its feature vector from `feats`
        lane_labels = np.zeros(feats.shape[0], dtype=int)
        for cid, idxs in enumerate(cluster_config):
            lane_labels[list(idxs)] = cid

        # Full Euclidean distance matrix for the 4 feature vectors
        D_feat = squareform(pdist(feats))
        dunn_val = _dunn_index(D_feat, lane_labels)

        # ------------------------------------------------------------------
        #     B) Divergence between cluster byte-distributions
        # ------------------------------------------------------------------
        # Pre-compute histograms
        hists = [_hist256(c) for c in cluster_streams]

        ce_vals, js_vals = [], []
        for i, j in combinations(range(len(hists)), 2):
            p, q = hists[i], hists[j]
            # Cross-entropy H(P,Q) = H(P) + KL(P‖Q)
            ce_vals.append(-(p * np.log2(q + 1e-12)).sum())
            # Jensen–Shannon divergence (base-2): ½ KL(P‖M)+½ KL(Q‖M)
            m = 0.5 * (p + q)
            kl_pm = rel_entr(p, m).sum() / np.log(2)  # convert nats → bits
            kl_qm = rel_entr(q, m).sum() / np.log(2)
            js_vals.append(0.5 * (kl_pm + kl_qm))

        avg_ce = float(np.mean(ce_vals)) if ce_vals else float("nan")
        avg_js = float(np.mean(js_vals)) if js_vals else float("nan")

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
            "BetweenSTD_H2": between,
            "WithinSTD_H2": avg_within_std_H2,
            "BetweenSTD": between_H2,
            "DaviesBouldin": db_index,
            "Dunn": dunn_val,
            "AvgCrossEntropy": avg_ce,
            "AvgJSDivergence": avg_js,
            "DeltaMatchEntropy": delta_match,
            "DeltaH0": delta_h0,
        }
        records.append(rec)



    df = pd.DataFrame(records)

    if comp_name.lower() in {"fastlz", "lz4", "snappy"}:
        df["SeparationScore_final"] = df["DeltaMatchEntropy"]
    else:
        df["SeparationScore_final"] = df["DeltaH0"]

    rho_final = df["SeparationScore_final"].corr(df["DecomposedRatio_Row_F"])
    print(f"ρ(SeparationScore_final, ratio) = {rho_final:.3f}")


    out_csv = f"synthetic_all_partitions_{comp_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Written {out_csv}")
    return df


###################################



# ────────────────────────────────────────────────────────────────
#  0.  put these imports once, near your other matplotlib imports
# ────────────────────────────────────────────────────────────────
import seaborn as sns
import matplotlib.pyplot as plt



def plot_corr_to_ratios(df,
                        metrics=("WithinSTD",
                                 "BetweenSTD",
                                 "WithinSTD_H2",
                                 "BetweenSTD_H2",
                                 "DeltaH0", "DeltaMatchEntropy"),
                        ratio_cols=("DecomposedRatio_Row_C",
                                    ),
                        codec_tag="FastLZ",
                        save_dir=""):
    """
    Compute Pearson correlations and save a  heat-map.
    Returns the PNG path.
    """
    # ---------- build a tidy DF of correlations -------------------
    rows = []
    for m in metrics:
        for r in ratio_cols:
            # guard against constant columns → corr = NaN
            rho = df[m].corr(df[r])
            rows.append((m, r, 0.0 if np.isnan(rho) else rho))
    corr_df = pd.DataFrame(rows, columns=["Metric", "Ratio", "ρ"])

    # ---------- print exact values --------------------------------
    # ---------- print exact values & rename the two ratio columns -----
    wide = corr_df.pivot(index="Metric", columns="Ratio", values="ρ")

    # ✨ NEW – nicer labels for the two ratios

    ratio_alias = {
        "DecomposedRatio_Row_C": "Decomposed compression ratio",
    }

    # ---------- Apply renaming ----------
    wide = wide.rename( columns=ratio_alias)

    # ---------- Print correlation values ----------
    print(f"\n=== Pearson correlations – {codec_tag} ===")
    print(wide.round(3).to_string())

    # ---------- Heatmap -----------------------------------------
    plt.figure(figsize=(6, 0.6 * len(metrics) + 1))
    sns.heatmap(wide, annot=True, fmt=".2f",
                center=0, cmap="vlag", cbar_kws=dict(label="ρ"))
    plt.title(f"{codec_tag}: Correlation ")
    plt.tight_layout()



    plt.savefig(f"corr_ratios_re_{codec_tag.lower()}.png", dpi=300)
    plt.close()


    return "2"

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_three_corr_heatmaps(dfs, tags,
                             metrics=("WithinSTD",
                                      "BetweenSTD",
                                      "WithinSTD_H2",
                                      "BetweenSTD_H2",
                                      "DeltaH0", "DeltaMatchEntropy"),
                             ratio_cols=("DecomposedRatio_Row_C",),
                             save_path="/home/jamalids/Documents/corr_three_methods.png"):
    """
    Given a list of DataFrames [df_result, df_zstd, df_huffman] and matching tags,
    compute the same correlation heatmap for each (using your metrics & ratio_cols),
    and lay them out in a single row.
    """
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 0.6*len(metrics) + 1), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, df, tag in zip(axes, dfs, tags):
        # build tidy DF of correlations
        rows = []
        for m in metrics:
            for r in ratio_cols:
                rho = df[m].corr(df[r])
                rows.append((m, r, 0.0 if np.isnan(rho) else rho))
        corr_df = pd.DataFrame(rows, columns=["Metric", "Ratio", "ρ"])
        wide = corr_df.pivot(index="Metric", columns="Ratio", values="ρ")
        # rename ratio column
        wide = wide.rename(columns={"DecomposedRatio_Row_C": "Decomposed compression ratio"})

        # plot
        sns.heatmap(wide, annot=True, fmt=".2f", center=0, cmap="vlag",
                    cbar_kws=dict(label="ρ"), ax=ax)
        ax.set_title(f"{tag}: Correlation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined heatmap to {save_path}")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

def plot_rho_heatmap(
    dfs, tags,
    metrics=("WithinSTD_H2","BetweenSTD_H2"),
    ratio_col="DecomposedRatio_Row_C",
    save_path="/home/jamalids/Documents/rho_heatmap_vldb.pdf"
):
    """
    VLDB-style Pearson ρ heatmap with figsize=(6.8, 3)
    """
    # Build the r-matrix
    r_mat = pd.DataFrame(index=metrics, columns=tags, dtype=float)
    for df, tag in zip(dfs, tags):
        for m in metrics:
            r, _ = pearsonr(df[m].astype(float), df[ratio_col].astype(float))
            r_mat.at[m, tag] = r

    # VLDB styling
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Plot
    fig, ax = plt.subplots(figsize=(6.8, 3))
    sns.heatmap(
        r_mat,
        annot=True, fmt=".2f",
        cmap="vlag", center=0,
        cbar_kws={"label": "ρ"},
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )
    ax.set_title("Pearson ρ", fontsize=9, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticklabels(metrics, rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved VLDB-format heatmap → {save_path}")

# Usage example:
# plot_rho_heatmap_vldb(
#     [df_result, df_zstd, df_huffman],
#     ["FastLZ", "Zstd", "Huffman"]
# )
def plot_rho_heatmap_min_max(
    dfs, tags,
    metrics=("WithinSTD_H2","BetweenSTD_H2"),
    ratio_col="KthEntropy",
    save_path="/home/jamalids/Documents/rho_min-maxb.pdf"
):
    """
    VLDB-style Pearson ρ heatmap with figsize=(6.8, 3)
    """
    # Build the r-matrix
    r_mat = pd.DataFrame(index=metrics, columns=tags, dtype=float)
    for df, tag in zip(dfs, tags):
        for m in metrics:
            r, _ = pearsonr(df[m].astype(float), df[ratio_col].astype(float))
            r_mat.at[m, tag] = r

    # VLDB styling
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Plot
    fig, ax = plt.subplots(figsize=(6.8, 3))
    sns.heatmap(
        r_mat,
        annot=True, fmt=".2f",
        cmap="vlag", center=0,
        cbar_kws={"label": "ρ"},
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )
    ax.set_title("Pearson ρ", fontsize=9, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticklabels(metrics, rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Saved VLDB-format heatmap → {save_path}")



if __name__=="__main__":
   # recs = test_synthetic_all_modes()
   df_result = test_synthetic_all_modes()

    # Run with Zstd
   df_zstd = test_synthetic_all_modes(compress_method=zstd_comp, comp_name="Zstd")


    # Run with Huffman
   df_huffman = test_synthetic_all_modes(compress_method=huffman_compress, comp_name="Huffman")

   plot_three_corr_heatmaps(
       [ df_huffman,df_result, df_zstd],
       ["Huffman","FastLZ", "Zstd"]
   )



   plot_rho_heatmap(
       [df_result, df_huffman, df_zstd],
       ["FastLZ", "Huffman", "Zstd"]
   )
   plot_rho_heatmap_min_max(
       [df_result, df_huffman, df_zstd],
       ["H2(Huffman)","H2(FastLZ)", "H2(Zstd)"]
   )






