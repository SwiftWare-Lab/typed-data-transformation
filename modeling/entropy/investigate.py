# -*- coding: utf-8 -*-
"""
Complete, ready‑to‑run version of your analysis script with joint‑entropy,
conditional‑entropy and mutual‑information metrics plugged in.

Key additions  ⚙️
----------------
1. `joint_entropy`, `conditional_entropy`, `mutual_information` helpers.
2. A global MI / Hcond matrix computed **before** clustering for every byte
   group.
3. In‑cluster information‑theoretic quality report after clustering, showing
   percentage MI gain and conditional‑entropy drop relative to the baseline.
4. Optional MI heat‑map plotting (turned off by default — flip the flag near
   the top if you want PNGs).

The rest of the workflow (entropy sliding windows, feature extraction,
clustering, compression test, plots) is untouched, so your prior results stay
comparable.  Path variables at the bottom can be edited in place.
"""

# --------------------------- Imports --------------------------- #
import os
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import zlib

# Switch ⬇️ to `True` if you want MI heat‑map PNGs alongside correlation plots
PLOT_MI_HEATMAP = False

# --------------------------- Entropy Helpers --------------------------- #

def compute_entropy(data_window):
    """Shannon entropy of a 1‑D uint8 array."""
    freq = Counter(data_window)
    total = len(data_window)
    H = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            H -= p * math.log2(p)
    return H


def joint_entropy(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    """H(X,Y) for two uint8 arrays using a 2‑D histogram."""
    h2d, _x, _y = np.histogram2d(a, b, bins=[bins, bins])
    p = h2d.ravel()
    p = p / p.sum()
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def conditional_entropy(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    """H(Y | X) = H(X,Y) − H(X)."""
    return joint_entropy(a, b, bins=bins) - compute_entropy(a)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    """I(X;Y) = H(X) + H(Y) − H(X,Y)."""
    return (
        compute_entropy(a)
        + compute_entropy(b)
        - joint_entropy(a, b, bins=bins)
    )


# --------------------------- Sliding‑window Entropy --------------------------- #

def calculate_entropy_over_data(data: np.ndarray, window_size: int = 256):
    entropies = []
    data_len = len(data)
    for start_idx in range(0, data_len - window_size + 1, window_size):
        window = data[start_idx : start_idx + window_size]
        entropies.append(compute_entropy(window))
    return entropies


# --------------------------- Feature Extraction --------------------------- #

def extract_features(byte_group: np.ndarray, window_size: int = 256):
    entropies = calculate_entropy_over_data(byte_group, window_size)
    feats = [np.mean(entropies), np.std(entropies), np.max(entropies), np.min(entropies)]
    freq = Counter(byte_group)
    byte_freq = np.array([freq.get(i, 0) / len(byte_group) for i in range(256)])
    return np.concatenate([feats, byte_freq])


# --------------------------- Byte splitting --------------------------- #

def split_bytes_into_components(byte_array: bytes, component_sizes):
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    components = []
    for i, size in enumerate(component_sizes):
        offset = sum(component_sizes[:i])
        components.append(byte_array[offset :: total_bytes])
    return components


# --------------------------- Clustering helpers --------------------------- #

def perform_hierarchical_clustering(feature_matrix, method="complete"):
    return linkage(feature_matrix, method=method)


def determine_optimal_clusters(linked, feature_matrix, max_clusters=3):
    silhouette_scores = {}
    for k in range(1, max_clusters + 1):
        labels = fcluster(linked, k, criterion="maxclust")
        try:
            score = silhouette_score(feature_matrix, labels)
            silhouette_scores[k] = score
        except Exception:
            silhouette_scores[k] = -1
    return max(silhouette_scores, key=silhouette_scores.get)


def group_and_reorder(byte_groups, cluster_labels):
    info = [dict(group=i + 1, label=lab) for i, lab in enumerate(cluster_labels)]
    ordered = sorted(info, key=lambda d: (d["label"], d["group"]))
    return [d["group"] for d in ordered]


# --------------------------- Compression evaluation --------------------------- #

def compress_and_evaluate(byte_groups, ordered_indices):
    ordered = [byte_groups[i - 1].tobytes() for i in ordered_indices]
    merged = b"".join(ordered)
    comp = zlib.compress(merged)
    return len(comp) / len(merged), len(merged), len(comp)


# --------------------------- Plot helpers --------------------------- #

def plot_entropy_profiles(components_entropy, dataset_name, save_path):
    df = pd.DataFrame(components_entropy).T
    df.columns = [f"Group {i+1}" for i in range(len(components_entropy))]
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(f"Entropy Profiles – {dataset_name}")
    plt.xlabel("Window")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(mat, title, save_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------------- Main workflow --------------------------- #

def run_analysis(folder_path):
    if not os.path.isdir(folder_path):
        print(f"✖ Folder '{folder_path}' not found.")
        return

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tsv")]
    if not tsv_files:
        print("No .tsv files in folder.")
        return

    for tsv_file in tsv_files:
        dataset_path = os.path.join(folder_path, tsv_file)
        name = os.path.splitext(tsv_file)[0]
        out_dir = os.path.join(
            os.path.dirname(folder_path), "results", f"{name}_entropy_clust"
        )
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Processing {tsv_file} ===")

        # ---------- Load & prepare ---------- #
        try:
            df = pd.read_csv(dataset_path, sep="\t", header=None)
        except Exception as e:
            print("  ✖ Failed to load:", e)
            continue

        # Cast to float64 → flatten F‑order → bytes
        values = df.iloc[:, 1:].to_numpy().astype(np.float64)  # skip id col
        byte_data = values.flatten(order="F").tobytes()

        # 4 components of 1 byte each
        comp_sizes = [1, 1, 1, 1]
        byte_groups = split_bytes_into_components(byte_data, comp_sizes)
        n_groups = len(byte_groups)

        # ---------- Global MI / Hcond matrix ---------- #
        print("  Computing MI / conditional‑entropy baseline …")
        mi_mat = np.zeros((n_groups, n_groups))
        hcond_mat = np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                mi = mutual_information(byte_groups[i], byte_groups[j])
                hc = conditional_entropy(byte_groups[i], byte_groups[j])
                mi_mat[i, j] = mi_mat[j, i] = mi
                hcond_mat[i, j] = hcond_mat[j, i] = hc
        base_avg_mi = mi_mat[np.triu_indices(n_groups, 1)].mean()
        base_avg_hc = hcond_mat[np.triu_indices(n_groups, 1)].mean()
        print(f"    Avg pair MI: {base_avg_mi:.3f} bits ; Avg H(Y|X): {base_avg_hc:.3f}")

        if PLOT_MI_HEATMAP:
            plot_heatmap(
                pd.DataFrame(mi_mat),
                f"Mutual Information – {name}",
                os.path.join(out_dir, f"{name}_mi_heatmap.png"),
            )

        # ---------- Sliding‑window entropy per group ---------- #
        window = 65536
        comp_entropy = [calculate_entropy_over_data(g, window) for g in byte_groups]
        plot_entropy_profiles(
            comp_entropy,
            name,
            os.path.join(out_dir, f"{name}_entropy_profiles.png"),
        )

        # ---------- Feature matrix & clustering ---------- #
        feats = np.vstack([extract_features(g, window) for g in byte_groups])
        Z = perform_hierarchical_clustering(feats, method="complete")
        k_opt = determine_optimal_clusters(Z, feats, max_clusters=3)
        labels = fcluster(Z, k_opt, criterion="maxclust")

        # ---------- In‑cluster information gain ---------- #
        intra_mis, intra_hcs = [], []
        for grp in sorted(set(labels)):
            idx = [i for i, lab in enumerate(labels) if lab == grp]
            for p in range(len(idx)):
                for q in range(p + 1, len(idx)):
                    i, j = idx[p], idx[q]
                    intra_mis.append(mi_mat[i, j])
                    intra_hcs.append(hcond_mat[i, j])
        if intra_mis:
            avg_intra_mi = np.mean(intra_mis)
            avg_intra_hc = np.mean(intra_hcs)
            gain_pct = 100 * (avg_intra_mi - base_avg_mi) / base_avg_mi
            drop_pct = 100 * (base_avg_hc - avg_intra_hc) / base_avg_hc
            print(
                f"    In‑cluster MI : {avg_intra_mi:.3f} bits  (∆ {gain_pct:+.1f}%)"
            )
            print(
                f"    In‑cluster Hc : {avg_intra_hc:.3f} bits  (∆ {drop_pct:+.1f}%)"
            )
        else:
            print("    Single‑element clusters; no intra metrics.")

        # ---------- Re‑order groups & compress ---------- #
        order = group_and_reorder(byte_groups, labels)
        best_ratio, orig_size, comp_size = compress_and_evaluate(byte_groups, order)
        print(
            f"    Compression  : {best_ratio:.4f} ({comp_size}/{orig_size} bytes) after re‑order"
        )
        orig_ratio, *_ = compress_and_evaluate(byte_groups, list(range(1, n_groups + 1)))
        print(f"    Baseline     : {orig_ratio:.4f} (original grouping)\n")

    print("All datasets completed. ✅")


# --------------------------- Entry‑point --------------------------- #
if __name__ == "__main__":
    # ✏️ EDIT this to your TSV folder path
    TSV_FOLDER = (
        "/mnt/c/Users/jamalids/Downloads/dataset/HPC"
    )
    run_analysis(TSV_FOLDER)
