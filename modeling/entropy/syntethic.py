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

def test_synthetic_all_modes():
    SIZE, ENT    = 50000, [7,6.5,1,0.2]
    packed, groups = generate_float_stream(SIZE, ENT)
    orig_size     = len(packed)

    # Precompute H1/H2 once
    H1 = [compute_entropy(g) for g in groups]
    H2 = [compute_kth_entropy(g,2) for g in groups]

    modes = ["entropy", "frequency", "all"]
    records = []

    for mode in modes:
        # build feature matrix
        feats  = np.vstack([extract_features(g, mode) for g in groups])
        linked = linkage(feats, method='complete')

        for k in range(1,5):
            labels   = np.ones(4,int) if k==1 else fcluster(linked, k, criterion='maxclust')
            clusters = {lab:[i for i,l in enumerate(labels) if l==lab] for lab in sorted(set(labels))}
            cfg      = "|".join(f"({','.join(map(str,idxs))})" for idxs in clusters.values())

            # Standard
            r_std = ratio(orig_size, compress_with_fastlz(packed.tobytes()))

            # per-cluster merged byte-strings
            merged_bs = [b"".join(groups[i].tobytes() for i in idxs) for idxs in clusters.values()]

            # Decomposed Col
            dec_sizes_col = [len(compress_with_fastlz(b)) for b in merged_bs]
            r_dec_col     = orig_size / sum(dec_sizes_col)

            # Decomposed Row
            dec_sizes_row = []
            for idxs in clusters.values():
                arrs = [groups[i] for i in idxs]
                dec_sizes_row.append(len(compress_with_fastlz(interleave_bytes(arrs))))
            r_dec_row = orig_size / sum(dec_sizes_row)

            # Reordered Col
            concat_all   = b"".join(merged_bs)
            r_reo_col    = ratio(orig_size, compress_with_fastlz(concat_all))

            # Reordered Row
            merged_np    = [np.frombuffer(b, dtype=np.uint8) for b in merged_bs]
            row_all      = interleave_bytes(merged_np)
            r_reo_row    = ratio(orig_size, compress_with_fastlz(row_all))

            # Info metrics on merged_np
            jh   = compute_joint_entropy(merged_np)
            mi   = max(sum(compute_entropy(g) for g in merged_np) - jh, 0.0)
            kth  = np.mean([compute_kth_entropy(g,2) for g in merged_np])

            # Within/Between entropies
            means, inners = [], []
            for g in merged_np:
                wins = [compute_entropy(g[i:i+256]) for i in range(0,len(g),256)]
                means.append(np.mean(wins)); inners.append(np.std(wins))
            avg_within = float(np.mean(means))
            between   = float(np.std(means))

            # Pairwise MI
            pairs = []
            for idxs in clusters.values():
                for i,j in itertools.combinations(idxs,2):
                    pairs.append(f"{i}-{j}={mutual_information_pair(groups[i],groups[j]):.4f}")
            pairstr = "|".join(pairs)

            rec = {
                "FeatureMode":            mode,
                "k":                      k,
                "ClusterConfig":          cfg,
                **{f"H1_c{i}":H1[i] for i in range(4)},
                **{f"H2_c{i}":H2[i] for i in range(4)},
                "AvgWithinEntropy":       avg_within,
                "BetweenClusterEntropySTD": between,
                "StandardRatio":          r_std,
                "DecomposedRatio_Col":    r_dec_col,
                "DecomposedRatio_Row":    r_dec_row,
                "ReorderedRatio_Col":     r_reo_col,
                "ReorderedRatio_Row":     r_reo_row,
                "JointEntropy":           jh,
                "MutualInfo":             mi,
                "KthEntropy":             float(kth),
                "WithinSTD":              float(np.mean(inners)),
                "BetweenSTD":             between,
                "PairwiseMI":             pairstr
            }
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv("synthetic_all_modes.csv", index=False)
    print("Written synthetic_all_modes.csv")
    return records

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
    dec  = [r["DecomposedRatio_Col"] for r in recs]
    mi   = [r["MutualInfo"]       for r in recs]

    x = np.arange(len(ks))

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(x, dec, marker="o", linestyle="-", label="Dec_Col Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ks)
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("DecomposedRatio_Col", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, mi, marker="s", linestyle="--", color="C1", label="MutualInfo")
    ax2.set_ylabel("MutualInfo (bits)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.legend(loc="upper right")

    plt.title(f"DecomposedRatio_Col and MutualInfo vs k (mode={mode})")
    plt.tight_layout()
    plt.savefig(f"synthetic_decompCol_vs_mi_{mode}.png")
    plt.close()
    print(f"Saved synthetic_decompCol_vs_mi_{mode}.png")
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

if __name__=="__main__":
    recs = test_synthetic_all_modes()
    plot_ratios_and_kth(recs)
    plot_decomp_col_vs_mi(recs, mode="all")
    plot_decomp_col_vs_mi(recs, mode="entropy")
    plot_decomp_col_vs_mi(recs, mode="frequency")
    # new plots
    for m in ["all", "entropy", "frequency"]:
        plot_kth_entropy_vs_k(recs, mode=m, kth_order=2)
        plot_decomp_reo_vs_mi(recs, mode=m)

