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


# Replace this import with your actual fastlz_compress location
from compressiojn_tools import fastlz_compress, huffman_compress,zstd_comp,zlib_comp,bz2_comp,snappy_comp,lzma_compress

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

def test_synthetic_all_modes(SIZE=65536, ENT=[3, 1,6, 5], mode="frequency", compress_method=None, comp_name=""):


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
            "BetweenSTD": between,
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
    plot_path = plot_path = f"/home/jamalids/Documents/{comp_tool.lower()}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_many_vs_ratio(
    df,
    comp_tool="FastLZ",
    ds_name="Synthetic",
    save_dir="/home/jamalids/Documents",
):
    """
    Draw scatter-and-trend plots of DecomposedRatio_Row_F vs 10 metrics,
    with HClust configs in red and Non-HClust configs in blue.

    Added metrics:
        • AvgCrossEntropy
        • AvgJSDivergence
    """
    ratio_col = "DecomposedRatio_Row_F"
    base_line = df["StandardRatio"].iloc[0]      # identical for all rows
    palette   = {"HClust": "red", "Non-HClust": "blue"}

    # ---- 10 X-metrics ------------------------------------------------------
    x_cols = [
       # ("MutualInfo",          "Mutual Information (bits)"),
        ("JointEntropy",        "Joint Entropy (bits)"),
        ("HC_H1_weighted",      "Weighted H1"),
        ("HC_H2_weighted",      "Weighted H2"),
        ("WithinSTD",           "Within-Cluster STD"),
        ("BetweenSTD",          "Between-Cluster STD"),
        ("HC_H1_num",           "Mean HC_H1"),
        ("HC_H2_num",           "Mean HC_H2"),
        ("AvgCrossEntropy",     "Avg Cross-Entropy (bits)"),
        ("AvgJSDivergence",     "Avg JS Divergence (bits)"),
        ("SeparationScore_final", "Δ-Metric (best for codec)"),

    ]

    # ---- Ensure numeric helper columns for raw HC lists --------------------
    if "HC_H1_num" not in df.columns:
        df = df.copy()
        df["HC_H1_num"] = df["HC_H1"].apply(
            lambda s: np.mean(list(map(float, s.split(",")))))
        df["HC_H2_num"] = df["HC_H2"].apply(
            lambda s: np.mean(list(map(float, s.split(",")))))

    # ---- Layout: 2 rows × 5 columns ---------------------------------------
    fig, axes_grid = plt.subplots(2, 5, figsize=(28, 8), sharey=True)
    axes  = axes_grid.ravel()
    ncols = 5   # for left-most-column check

    # ---- Plot loop ---------------------------------------------------------
    for idx, (ax, (xcol, xlabel)) in enumerate(zip(axes, x_cols)):
        sns.scatterplot(data=df, x=xcol, y=ratio_col,
                        hue="ConfigType", palette=palette, ax=ax,
                        legend=False, s=50)

        sns.regplot(data=df, x=xcol, y=ratio_col,
                    scatter=False, color="black", ax=ax)

        ax.axhline(base_line, ls="--", lw=1, color="grey")

        # annotate hierarchical-clustering points
        for _, row in df[df["ConfigType"] == "HClust"].iterrows():
            ax.text(row[xcol], row[ratio_col], row["ClusterConfig"],
                    fontsize=8, color="black", rotation=20,
                    va="bottom", ha="left")

        # R²
        r2 = stats.linregress(df[xcol], df[ratio_col]).rvalue ** 2
        ax.text(0.02, 0.94, f"R² = {r2:.2f}", transform=ax.transAxes,
                fontsize=8, va="top", ha="left")

        ax.set_title(xlabel, fontsize=11)
        ax.set_xlabel(xlabel)
        if idx % ncols == 0:
            ax.set_ylabel("Decomposed Ratio (Row-F)")
        else:
            ax.set_ylabel(None)

    # ---- Title & save ------------------------------------------------------
    fig.suptitle(
        f"{ds_name} – {comp_tool}: Row-F Compression vs Multiple Metrics",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir,
                            f"rowF_vs_metrics_{comp_tool.lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path

##################################
# Plot with text annotations showing cluster config for each point
def plot_with_cluster_labels(df, comp_tool="FastLZ", ds_name="Synthetic"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ratio_col = "DecomposedRatio_Row_F"

    # Plot 1: MutualInfo vs Compression Ratio
    sns.scatterplot(data=df, x="MutualInfo", y=ratio_col, ax=ax1)
    sns.regplot(data=df, x="MutualInfo", y=ratio_col, scatter=False, color='red', ax=ax1)
    ax1.set_title(f"{ds_name} : {comp_tool} — MutualInfo vs Ratio")
    ax1.set_xlabel("Mutual Information (bits)")
    ax1.set_ylabel("Decomposed Compression Ratio (Row-F)")
    r2_mi = stats.linregress(df["MutualInfo"], df[ratio_col]).rvalue ** 2
    ax1.text(0.05, 0.95, f"R² = {r2_mi:.2f}", transform=ax1.transAxes, fontsize=10, verticalalignment='top', color='red')

    # Add cluster config labels
    for i, row in df.iterrows():
        ax1.annotate(row["ClusterConfig"], (row["MutualInfo"], row[ratio_col]), fontsize=8, alpha=0.7)

    # Plot 2: JointEntropy vs Compression Ratio
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
    plot_path = f"/home/jamalids/Documents/{comp_tool.lower()}.png"

    plt.savefig(plot_path)
    plt.close()
    return plot_path

# ────────────────────────────────────────────────────────────────
#  0.  put these imports once, near your other matplotlib imports
# ────────────────────────────────────────────────────────────────
import seaborn as sns
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
#  1.  helper that adds boolean columns + plot
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
#  correlation heat-map helper  (add after your imports just once)
# ────────────────────────────────────────────────────────────────

def plot_corr_to_ratios(df,
                        metrics=("WithinSTD",
                                 "BetweenSTD",
                                 "HC_H1_weighted",
                                 "HC_H2_weighted",
                                 "DeltaMatchEntropy",
                                 "AvgJSDivergence",
                                 "AvgCrossEntropy",
                                 "JointEntropy",
                                 "MutualInfo",
                                 "DeltaH0"),
                        ratio_cols=("DecomposedRatio_Row_C",
                                    "DecomposedRatio_Row_F"),
                        codec_tag="FastLZ",
                        save_dir="/home/jamalids/Documents"):
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
        "DecomposedRatio_Row_C": "decom-Row ",
        "DecomposedRatio_Row_F": "decom-Col",
    }
    wide = wide.rename(columns=ratio_alias)  # <-- rename for display

    print(f"\n=== Pearson correlations – {codec_tag} ===")
    print(wide.round(3).to_string())

    # ---------- heat-map ------------------------------------------
    plt.figure(figsize=(6, 0.6*len(metrics)+1))
    sns.heatmap(wide, annot=True, fmt=".2f",
                center=0, cmap="vlag", cbar_kws=dict(label="ρ"))
    plt.title(f"{codec_tag}: correlation with decomposed ratios")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir,
                            f"corr_ratios_{codec_tag.lower()}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print("saved →", png_path)
    return png_path

def plot_corr_to_ratios_re(df,
                        metrics=("WithinSTD",
                                 "BetweenSTD",
                                 "HC_H1_weighted",
                                 "HC_H2_weighted",
                                 "DeltaMatchEntropy",
                                 "AvgJSDivergence",
                                 "AvgCrossEntropy",
                                 "JointEntropy",
                                 "MutualInfo",
                                 "DeltaH0"),
                        ratio_cols=("DecomposedRatio_Row_C",
                                    "DecomposedRatio_Row_F"),
                        codec_tag="FastLZ",
                        save_dir="/home/jamalids/Documents"):
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
        "ReorderedRatio_Row_C": "Reordered-Row ",
        "ReorderedRatio_Row_F": "Reordered-Col",
    }
    wide = wide.rename(columns=ratio_alias)  # <-- rename for display

    print(f"\n=== Pearson correlations – {codec_tag} ===")
    print(wide.round(3).to_string())

    # ---------- heat-map ------------------------------------------
    plt.figure(figsize=(6, 0.6*len(metrics)+1))
    sns.heatmap(wide, annot=True, fmt=".2f",
                center=0, cmap="vlag", cbar_kws=dict(label="ρ"))
    plt.title(f"{codec_tag}: correlation with Reordered ratios")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir,
                            f"corr_ratios_re_{codec_tag.lower()}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print("saved →", png_path)
    return png_path


if __name__=="__main__":
   # recs = test_synthetic_all_modes()
   df_result = test_synthetic_all_modes()
  # plot_mutual_and_joint_vs_ratio(df_result)
   plot_mutual_and_joint_vs_ratio(df_result, comp_tool="fastlz")
   plot_many_vs_ratio(df_result, comp_tool="FastLZ", ds_name="Synthetic")

   df_result.to_csv("/home/jamalids/Documents/synthetic_fastlz.csv")
    # Run with Zstd
   df_zstd = test_synthetic_all_modes(compress_method=zstd_comp, comp_name="Zstd")
   df_zstd.to_csv("/home/jamalids/Documents/synthetic_zstd.csv")
   plot_mutual_and_joint_vs_ratio(df_zstd, comp_tool="zstd")
   plot_many_vs_ratio(df_zstd, comp_tool="Zstd", ds_name="Synthetic")
    # Run with Huffman
   df_huffman = test_synthetic_all_modes(compress_method=huffman_compress, comp_name="Huffman")
   df_huffman.to_csv("/home/jamalids/Documents/synthetic_huffman.csv")
   plot_mutual_and_joint_vs_ratio(df_huffman, comp_tool="huffman")
   plot_many_vs_ratio(df_huffman, comp_tool="Huffman", ds_name="Synthetic")
   # #################################
   df_zlib = test_synthetic_all_modes(compress_method=zlib_comp, comp_name="zlib")
   df_zlib.to_csv("/home/jamalids/Documents/synthetic_zlib.csv")
   plot_mutual_and_joint_vs_ratio(df_zlib, comp_tool="zlib")
   plot_many_vs_ratio(df_zlib, comp_tool="Zlib", ds_name="Synthetic")
   #
   # #################################
   df_bzib = test_synthetic_all_modes(compress_method=bz2_comp, comp_name="bzib")
   df_bzib.to_csv("/home/jamalids/Documents/synthetic_bzib.csv")
   plot_mutual_and_joint_vs_ratio(df_bzib, comp_tool="bzib")
   plot_many_vs_ratio(df_bzib, comp_tool="bzib", ds_name="Synthetic")
   #######################################################################3

   df_snappy = test_synthetic_all_modes(compress_method=snappy_comp, comp_name="snappy")
   df_snappy.to_csv("/home/jamalids/Documents/synthetic_snappy.csv")
   plot_mutual_and_joint_vs_ratio(df_snappy, comp_tool="snappy")
   plot_many_vs_ratio(df_snappy, comp_tool="snappy", ds_name="Synthetic")
   ##################################################################################
   df_lzma = test_synthetic_all_modes(compress_method=lzma_compress, comp_name="lzma")
   df_lzma.to_csv("/home/jamalids/Documents/synthetic_lzma.csv")
   plot_mutual_and_joint_vs_ratio(df_lzma, comp_tool="lzma")
   plot_many_vs_ratio(df_lzma, comp_tool="lzma", ds_name="Synthetic")
   #############################################################################
   plot_corr_to_ratios(df_result, codec_tag="FastLZ")
   plot_corr_to_ratios(df_huffman, codec_tag="Huffman")
   plot_corr_to_ratios(df_zstd, codec_tag="ZSTD")
   plot_corr_to_ratios(df_snappy, codec_tag="snappy")
   plot_corr_to_ratios(df_zlib, codec_tag="zlib")
   plot_corr_to_ratios(df_bzib, codec_tag="bzib")
   plot_corr_to_ratios(df_lzma, codec_tag="lzma")
   plot_corr_to_ratios_re(df_result, codec_tag="FastLZ")
   plot_corr_to_ratios_re(df_huffman, codec_tag="Huffman")
   plot_corr_to_ratios_re(df_zstd, codec_tag="ZSTD")
   plot_corr_to_ratios_re(df_snappy, codec_tag="snappy")
   plot_corr_to_ratios_re(df_zlib, codec_tag="zlib")
   plot_corr_to_ratios_re(df_bzib, codec_tag="bzib")
   plot_corr_to_ratios_re(df_lzma, codec_tag="lzma")





