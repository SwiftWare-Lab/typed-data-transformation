import os
import math
import numpy as np
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Replace this import with your actual fastlz_compress location
from modeling.compression_tools import fastlz_compress

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

def extract_features(byte_group, mode="all", window=65536):
    if mode=="entropy":   return extract_entropy(byte_group, window)
    if mode=="frequency": return extract_frequency(byte_group)
    if mode=="all":       return extract_all(byte_group, window)
    raise ValueError(f"Unknown mode: {mode}")

# ---------------------- COMPRESSION UTIL ---------------------- #

def compress_with_fastlz(data_bytes):
    return fastlz_compress(data_bytes)

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

# ---------------------- MAIN TEST ---------------------- #
def test_synthetic_all_modes(SIZE=50000, ENT=[0,8, 3, 5]):
    packed, groups = generate_float_stream(SIZE, ENT)
    orig_size = len(packed)

    H1 = [compute_entropy(g) for g in groups]
    H2 = [compute_kth_entropy(g, 2) for g in groups]

    modes = ["entropy", "frequency", "all"]
    records = []

    for mode in modes:
        feats = np.vstack([extract_features(g, mode) for g in groups])
        linked = linkage(feats, method='complete')

        for k in range(1, 5):
            labels = np.ones(4, int) if k == 1 else fcluster(linked, k, criterion='maxclust')
            clusters = {lab: [i for i, l in enumerate(labels) if l == lab] for lab in sorted(set(labels))}
            cfg = "|".join(f"({','.join(map(str, idxs))})" for idxs in clusters.values())

            # Standard compression
            r_std = ratio(orig_size, compress_with_fastlz(packed.tobytes()))

            # Decomposed (Col)
            merged_bs = [b"".join(groups[i].tobytes() for i in idxs) for idxs in clusters.values()]
            dec_sizes_col = [len(compress_with_fastlz(b)) for b in merged_bs]
            r_decomp_col = orig_size / sum(dec_sizes_col)

            # Decomposed (Row_C)
            dec_sizes_row_C = []
            for idxs in clusters.values():
                arr2d = np.stack([groups[i] for i in idxs], axis=1)
                row_bytes_C = arr2d.reshape(-1, order='C').tobytes()
                dec_sizes_row_C.append(len(compress_with_fastlz(row_bytes_C)))
            r_decomp_row_C = orig_size / sum(dec_sizes_row_C)

            # Reordered (Col)
            reordered_bytes_col = b"".join(merged_bs)
            r_reorder_col = ratio(orig_size, compress_with_fastlz(reordered_bytes_col))

            # Reordered (Row_C): use full consistent shape
            arr2d_reordered_C = np.stack([groups[i] for i in range(4)], axis=1)
            row_all_C = arr2d_reordered_C.reshape(-1, order='C').tobytes()
            r_reorder_row_C = ratio(orig_size, compress_with_fastlz(row_all_C))

            # Entropy and mutual information
            merged_np = [np.frombuffer(b, dtype=np.uint8) for b in merged_bs]
            jh = compute_joint_entropy(merged_np)
            mi = max(sum(compute_entropy(g) for g in merged_np) - jh, 0.0)
            kth = np.mean([compute_kth_entropy(g, 2) for g in merged_np])

            # Window-based entropy stats
            means, inners = [], []
            for g in merged_np:
                wins = [compute_entropy(g[i:i + 256]) for i in range(0, len(g), 256)]
                means.append(np.mean(wins))
                inners.append(np.std(wins))
            avg_within = float(np.mean(means))
            between = float(np.std(means))

            rec = {
                "FeatureMode": mode,
                "k": k,
                "ClusterConfig": cfg,
                **{f"H1_c{i}": H1[i] for i in range(4)},
                **{f"H2_c{i}": H2[i] for i in range(4)},
                "AvgWithinEntropy": avg_within,
                "BetweenClusterEntropySTD": between,
                "StandardRatio": r_std,
                "DecomposedRatio_Col": r_decomp_col,
                "DecomposedRatio_Row_C": r_decomp_row_C,
                "ReorderedRatio_Col": r_reorder_col,
                "ReorderedRatio_Row_C": r_reorder_row_C,
                "JointEntropy": jh,
                "MutualInfo": mi,
                "KthEntropy": float(kth),
                "WithinSTD": float(np.mean(inners)),
                "BetweenSTD": between,
            }

            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv("synthetic_all_modes.csv", index=False)
    print("Written synthetic_all_modes.csv")
    return records


######################################################Real###############################
def transform_data(data_set_list, order='C'):
    # view data_set as bytes
    compressed_data = []
    for cmp in data_set_list:
        data_set_bytes = np.frombuffer(cmp.flatten(order).tobytes(), dtype=np.byte)
        compressed_data.append(data_set_bytes)
    # flatten the list
    compressed_data = np.concatenate(compressed_data, axis=0)
    return compressed_data
def test_real_datasets(folder_path):
    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print("No .tsv files found in", folder_path)
        return

    all_records = []

    for fname in tsv_files:
        dataset_name = os.path.splitext(fname)[0]
        fpath = os.path.join(folder_path, fname)
        print(f"Processing: {dataset_name}")

        try:
            df = pd.read_csv(fpath, sep='\t', header=None)
        except Exception as e:
            print("Failed to load", fname, e)
            continue

        # Load float32 column and convert to bytes
        numeric_vals = df.values[1:500000, 1].astype(np.float32)
        arr = np.frombuffer(numeric_vals.tobytes(), dtype=np.uint8)
        byte_groups = [arr[i::4] for i in range(4)]
        orig_size = len(arr)

        # Per-component entropy
        H1 = [compute_entropy(g) for g in byte_groups]
        H2 = [compute_kth_entropy(g, 2) for g in byte_groups]

        for mode in ["entropy", "frequency", "all"]:
            feats = np.vstack([extract_features(g, mode) for g in byte_groups])
            linked = linkage(feats, method='complete')

            for k in range(1, 5):
                labels = np.ones(4, int) if k == 1 else fcluster(linked, k, criterion='maxclust')
                clusters = {lab: [i for i, l in enumerate(labels) if l == lab] for lab in sorted(set(labels))}
                cfg = "|".join(f"({','.join(map(str, idxs))})" for idxs in clusters.values())

                # Standard compression
                r_std = ratio(orig_size, compress_with_fastlz(arr))

                # Decomposed (Col)
                merged_bs = [b"".join(byte_groups[i].tobytes() for i in idxs) for idxs in clusters.values()]
                dec_sizes_col = [len(compress_with_fastlz(b)) for b in merged_bs]
                r_dec_col = orig_size / sum(dec_sizes_col)

                # Decomposed (Row) C and F orders
                dec_sizes_row_C, dec_sizes_row_F = [], []

                for idxs in clusters.values():
                    # Stack byte groups into 2D float32-like layout
                    arr2d = np.stack([byte_groups[i] for i in idxs], axis=1)

                    # Use your transform function
                    row_bytes_C = transform_data([arr2d], order='C')
                    row_bytes_F = transform_data([arr2d], order='F')

                    # Compress transformed byte streams
                    dec_sizes_row_C.append(len(compress_with_fastlz(row_bytes_C)))
                    dec_sizes_row_F.append(len(compress_with_fastlz(row_bytes_F)))

                r_dec_row_C = orig_size / sum(dec_sizes_row_C)
                r_dec_row_F = orig_size / sum(dec_sizes_row_F)

                # Reordered (Col)
                concat_all = b"".join(merged_bs)
                r_reo_col = ratio(orig_size, compress_with_fastlz(concat_all))

                # Reordered (Row) using byte_groups (guaranteed equal length)
                all_arr2d = np.stack([byte_groups[i] for i in range(4)], axis=1)
                row_all_C = all_arr2d.reshape(-1, order='C').tobytes()
                row_all_F = all_arr2d.reshape(-1, order='F').tobytes()
                r_reo_row_C = ratio(orig_size, compress_with_fastlz(row_all_C))
                r_reo_row_F = ratio(orig_size, compress_with_fastlz(row_all_F))

                # Info metrics
                # (merged_np might be variable size — skip stacking and only use for MI, joint entropy, etc.)
                merged_np = [np.frombuffer(b, dtype=np.uint8) for b in merged_bs]
                jh = compute_joint_entropy(merged_np)
                mi = max(sum(compute_entropy(g) for g in merged_np) - jh, 0.0)
                kth = np.mean([compute_kth_entropy(g, 2) for g in merged_np])

                means, inners = [], []
                for g in merged_np:
                    wins = [compute_entropy(g[i:i + 256]) for i in range(0, len(g), 256)]
                    means.append(np.mean(wins))
                    inners.append(np.std(wins))
                avg_within = float(np.mean(means))
                between = float(np.std(means))

                rec = {
                    "Dataset": dataset_name,
                    "FeatureMode": mode,
                    "k": k,
                    "ClusterConfig": cfg,
                    **{f"H1_c{i}": H1[i] for i in range(4)},
                    **{f"H2_c{i}": H2[i] for i in range(4)},
                    "AvgWithinEntropy": avg_within,
                    "BetweenClusterEntropySTD": between,
                    "StandardRatio": r_std,
                    "DecomposedRatio_Col": r_dec_col,
                    "DecomposedRatio_Row_C": r_dec_row_C,
                    "DecomposedRatio_Row_F": r_dec_row_F,
                    "ReorderedRatio_Col": r_reo_col,
                    "ReorderedRatio_Row_C": r_reo_row_C,
                    "ReorderedRatio_Row_F": r_reo_row_F,
                    "JointEntropy": jh,
                    "MutualInfo": mi,
                    "KthEntropy": float(kth),
                    "WithinSTD": float(np.mean(inners)),
                    "BetweenSTD": between,
                }
                all_records.append(rec)

    df_real = pd.DataFrame(all_records)
    df_real.to_csv("/home/jamalids/Documents/real_dataset_results.csv", index=False)
    print("Written real_dataset_results.csv")
    return all_records

# ---------------------- PLOT ---------------------- #

def plot_ratios_and_kth(records, kth_order=2):
    # just plot the "all" mode as example
    all_rec = [r for r in records if r["FeatureMode"]=="all"]
    ks  = [r['k'] for r in all_rec]
    std = [r['StandardRatio'] for r in all_rec]
    dec = [r['DecomposedRatio_Col'] for r in all_rec]
    reo = [r['ReorderedRatio_Col'] for r in all_rec]
    kth = [r['KthEntropy'] for r in all_rec]

    x, w = np.arange(len(ks)), 0.25
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.bar(x-w, std, w, label='Std')
    ax1.bar(x  , dec, w, label='Dec')
    ax1.bar(x+w, reo, w, label='Reo')
    ax1.set_xticks(x); ax1.set_xticklabels(ks)
    ax1.set_xlabel('k'); ax1.set_ylabel('Compression Ratio')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(x, kth, marker='o', color='k', label=f'H$_{{{kth_order}}}$')
    ax2.set_ylabel(f'H$_{{{kth_order}}}$ Entropy')
    ax2.legend(loc='upper right')

    plt.title('All‐features mode: Ratios & k‐th Entropy vs k')
    plt.tight_layout()
    plt.savefig('synthetic_all_mode_plot.png')
    plt.close()

def plot_decomp_col_vs_mi(records, mode="all"):
    # filter by feature mode
    recs = [r for r in records if r["FeatureMode"] == mode]
    ks   = [r["k"]                for r in recs]
    dec  = [r["ReorderedRatio_Col"] for r in recs]
    mi   = [r["MutualInfo"]       for r in recs]

    x = np.arange(len(ks))

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(x, dec, marker="o", linestyle="-", label="Dec_Col Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ks)
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("ReorderedRatio_Col", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, mi, marker="s", linestyle="--", color="C1", label="MutualInfo")
    ax2.set_ylabel("MutualInfo (bits)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.legend(loc="upper right")

    plt.title(f"ReorderedRatio_Col and MutualInfo vs k (mode={mode})")
    plt.tight_layout()
    plt.savefig(f"synthetic_ReorderedRatio_Col_vs_mi_{mode}.png")
    plt.close()
    print(f"Saved synthetic_ReorderedRatio_Col_vs_mi_{mode}.png")
def plot_kth_entropy_vs_k(records, mode="all", kth_order=2):
    # filter records by mode
    recs = [r for r in records if r["FeatureMode"] == mode]
    ks   = [r["k"]           for r in recs]
    kth  = [r["KthEntropy"]  for r in recs]

    x = np.arange(len(ks))

    plt.figure(figsize=(8,5))
    plt.plot(x, kth, marker="o", linestyle="-", label=f"H$_{{{kth_order}}}$")
    plt.xticks(x, ks)
    plt.xlabel("k (clusters)")
    plt.ylabel(f"H$_{{{kth_order}}}$ Entropy (bits)")
    plt.title(f"2-th Order Entropy vs k (mode={mode})")
    plt.grid(True)
    plt.tight_layout()
    fname = f"synthetic_kthEntropy_vs_k_{mode}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

def plot_decomp_reo_vs_mi(records, mode="all"):
    # filter records by mode
    recs = [r for r in records if r["FeatureMode"] == mode]
    ks   = [r["k"]                   for r in recs]
    dec  = [r["DecomposedRatio_Col"] for r in recs]
    reo  = [r["ReorderedRatio_Col"]  for r in recs]
    mi   = [r["MutualInfo"]          for r in recs]

    x = np.arange(len(ks))

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(x, dec, marker="o", linestyle="-", label="Dec_Col")
    ax1.plot(x, reo, marker="s", linestyle="--", label="Reo_Col")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ks)
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("Compression Ratio")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, mi, marker="^", linestyle=":", color="C2", label="MutualInfo")
    ax2.set_ylabel("MutualInfo (bits)")
    ax2.legend(loc="upper right")

    plt.title(f"Decomp & Reo Ratios vs MutualInfo (mode={mode})")
    plt.tight_layout()
    fname = f"synthetic_decompReo_vs_mi_{mode}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def plot_decomp_row_vs_mutual_info(recs, mode="all"):
    """
    Plot correlation between Mutual Information and DecomposedRatio_Row using linear scale.
    Removes k=1 (no clustering) and zooms the Y-axis to show small differences clearly.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Filter for mode and exclude k=1
    recs_filtered = [r for r in recs if r["FeatureMode"] == mode and r["k"] in [2, 3, 4]]
    if not recs_filtered:
        print(f"No records found for mode: {mode}")
        return

    df = pd.DataFrame(recs_filtered)
    df = df[df["ReorderedRatio_Col"] > 0]

    # Plot styling
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # Regression line (clean trend)
    sns.regplot(
        data=df,
        x="MutualInfo",
        y="ReorderedRatio_Col",
        scatter=False,
        color="black",
        line_kws={"linewidth": 2, "label": "Regression Line"}
    )

    # Scatter points (k=2,3,4)
    sns.scatterplot(
        data=df,
        x="MutualInfo",
        y="ReorderedRatio_Col",
        hue="k",
        palette="viridis",
        s=80
    )

    # Tight Y-axis limits
    ymin = df["ReorderedRatio_Col"].min() * 0.95
    ymax = df["ReorderedRatio_Col"].max() * 1.05
    plt.ylim(ymin, ymax)

    # Labels and layout
    plt.xlabel("Mutual Information (bits)")
    plt.ylabel("Reordered Compression Ratio (Col-wise)")
    plt.title(f"Synthetic Data: Col Compression Ratio vs. Mutual Info (mode={mode})")
    plt.legend(title="k (clusters)", loc="best")
    plt.tight_layout()

    # Save
    fname = f"synthetic_mi_vs_decomposed_row_ratio_{mode}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def plot_avg_within_entropy_vs_global(csv_path="synthetic_all_modes.csv", mode="all", save_path=None):
    """
    Plots the average within-cluster entropy for different values of k and compares it to global entropy H(X).

    Parameters:
    - csv_path: Path to the CSV file containing synthetic results.
    - mode: Feature mode to filter by (default: "all").
    - save_path: Optional path to save the plot. If None, saves as 'synthetic_avg_within_entropy_vs_Hx.png'.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        df_all = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return

    df_mode = df_all[(df_all["FeatureMode"] == mode)]

    if df_mode.empty:
        print(f"No records found for mode='{mode}' in {csv_path}")
        return

    # Get global entropy (k=1)
    if not df_mode[df_mode["k"] == 1].empty:
        H_global = df_mode[df_mode["k"] == 1]["AvgWithinEntropy"].values[0]
    else:
        H_global = None

    # Filter only k > 1 to visualize clustering impact
    df = df_mode[df_mode["k"] > 1]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["AvgWithinEntropy"], marker="o", linestyle="-", label="AvgWithinEntropy (Clustered)")

    if H_global:
        plt.axhline(H_global, color='red', linestyle='--', label=f"H(X) ≈ {H_global:.3f}")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Within-Cluster Entropy")
    plt.title("Clustering Effect: AvgWithinEntropy vs Global Entropy H(X)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = "synthetic_avg_within_entropy_vs_Hx.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_avg_within_entropy_vs_global_from_df(df, mode="entropy", save_path="avg_within_vs_global_entropy.png"):
    """
    Plot AvgWithinEntropy for different k vs. H(X), using preloaded DataFrame.
    """
    import matplotlib.pyplot as plt

    df_mode = df[df["FeatureMode"] == mode]

    if df_mode.empty:
        print(f"No data for FeatureMode = {mode}")
        return

    # Extract global entropy from k=1
    global_entropy = df_mode[df_mode["k"] == 1]["AvgWithinEntropy"].values[0]
    df_k_gt1 = df_mode[df_mode["k"] > 1]

    plt.figure(figsize=(8, 5))
    plt.plot(df_k_gt1["k"], df_k_gt1["AvgWithinEntropy"], marker='o', label='AvgWithinEntropy (Clustered)')
    plt.axhline(global_entropy, color='red', linestyle='--', label=f"Global Entropy H(X) ≈ {global_entropy:.3f}")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Within-Cluster Entropy")
    plt.title(f"Clustering Effect: AvgWithinEntropy vs H(X) [{mode} mode]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

    print(f"Saved plot to {save_path}")

if __name__=="__main__":
    recs = test_synthetic_all_modes()
    plot_ratios_and_kth(recs)
    for m in ["all", "entropy", "frequency"]:
        plot_decomp_col_vs_mi(recs, mode=m)
        plot_kth_entropy_vs_k(recs, mode=m, kth_order=2)
        plot_decomp_reo_vs_mi(recs, mode=m)
        plot_decomp_row_vs_mutual_info(recs, mode=m)  # 👈 Just add this line
        plot_avg_within_entropy_vs_global(csv_path="/home/jamalids/Documents/frame/new3/big-data-compression/modeling/cross-entropy/synthetic_all_modes.csv", mode="all", save_path=None)
        df = pd.DataFrame(recs)
        plot_avg_within_entropy_vs_global_from_df(df, mode="entropy", save_path="avg_within_vs_global_entropy.png")



    # Process your real datasets
    # folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/test"
    # recs=test_real_datasets(folder_path)
    # plot_ratios_and_kth(recs)
    # for m in ["all", "entropy", "frequency"]:
    #     plot_decomp_col_vs_mi(recs, mode=m)
    #     plot_kth_entropy_vs_k(recs, mode=m, kth_order=2)
    #     plot_decomp_reo_vs_mi(recs, mode=m)
    #     plot_decomp_row_vs_mutual_info(recs, mode=m)  # 👈 Just add this line