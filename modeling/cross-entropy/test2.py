#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
#  Exhaustive lane-clustering → compression study
#  Prints linkage table, saves dendrogram, lists every partition’s CR
# ================================================================
import numpy as np, matplotlib.pyplot as plt
from collections import Counter
from enum import Enum
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import pandas as pd

# ----------------------------------------------------------------
# 1.  choose compression backend
# ----------------------------------------------------------------
from modeling.compression_tools import fastlz_compress   # adjust if needed

class CompressionTool(Enum):
    FASTLZ = "fastlz"

compression_tool = CompressionTool.FASTLZ

def compress(buf: np.ndarray) -> bytes:
    """Wrap the chosen compressor (FastLZ here)."""
    return fastlz_compress(np.ascontiguousarray(buf))

# ----------------------------------------------------------------
SAVE_DENDROGRAM = True            # False → skip plotting
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 2.  information-theory helpers
# ----------------------------------------------------------------
def cross_relative_entropy(a: np.ndarray, b: np.ndarray, eps=1e-12):
    """KL(P‖Q) and cross-entropy H(P,Q) in bits, ε-smoothed."""
    h1 = np.bincount(a, minlength=256); p = h1 / h1.sum()
    h2 = np.bincount(b, minlength=256); q = (h2 + eps); q /= q.sum()
    kl = stats.entropy(p, q, base=2)
    ce = stats.entropy(p, base=2) + kl
    return kl, ce

def build_linkage(components, eps=1e-12):
    """Return (SciPy linkage, symmetric KL distance matrix)."""
    n = len(components)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d  = cross_relative_entropy(components[i], components[j])[0] \
               + cross_relative_entropy(components[j], components[i])[0]
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D), method="single")   # single-link on KL
    return Z, D

# ----------------------------------------------------------------
# 3.  compression-ratio helpers
# ----------------------------------------------------------------
def comp_ratio(raw):                 # one-shot compression ratio
    return len(raw.tobytes()) / len(compress(raw))

def decomposed_ratio(clusters, total_bytes):
    comp_sz = sum(len(compress(c)) for c in clusters)
    return total_bytes / comp_sz

# ----------------------------------------------------------------
# 4.  exhaustive analysis
# ----------------------------------------------------------------
def analyse_components(components, ds_name):
    n = len(components)
    Z, _ = build_linkage(components)

    # ---- linkage table ------------------------------------------
    print("\n=== LINKAGE SEQUENCE (single-link, sym-KL) ===")
    print(pd.DataFrame(Z, columns=["cl1","cl2","dist","n"])
            .to_string(index=False))

    # ---- dendrogram ---------------------------------------------
    if SAVE_DENDROGRAM:
        plt.figure(figsize=(6,4))
        dendrogram(Z, labels=[f"lane{i}" for i in range(n)],
                   color_threshold=0)
        plt.title("Symmetric-KL dendrogram"); plt.tight_layout()
        plt.savefig("kl_dendrogram.png", dpi=300)
        #plt.show(block=False)
        print("Saved → kl_dendrogram.png")

    # ---- enumerate every partition ------------------------------
    def all_parts(idx):
        if not idx: yield []; return
        f,*r = idx
        for p in all_parts(r):
            for k in range(len(p)):
                yield p[:k]+[[f]+p[k]]+p[k+1:]
            yield [[f]]+p

    total = sum(c.size for c in components)
    rows=[]
    for part in all_parts(list(range(n))):
        merged=[np.concatenate([components[i] for i in grp]) for grp in part]
        dec_cr = decomposed_ratio(merged,total)
        reo_cr = comp_ratio(np.concatenate(merged))
        rows.append(dict(k=len(part), Part=part,
                         DecompCR=dec_cr, ReordCR=reo_cr))

    df = pd.DataFrame(rows).sort_values("DecompCR",ascending=False)
    print("\n=== ALL PARTITIONS sorted by DecompCR ===")
    print(df[["k","Part","DecompCR"]].to_string(index=False))
    best = df.iloc[0]
    print(f"\n→ BEST: k={best.k}  CR={best.DecompCR:.3f}\n   {best.Part}")
    return df

# ----------------------------------------------------------------
# 5.  synthetic stream generator  (fixed syntax)
# ----------------------------------------------------------------
def gen_float_stream(n, Hs):
    """Generate 4 lanes with entropies Hs (tuple/list of 4 ints)."""
    lanes=[]
    for H in Hs:
        k = int(2**H)
        p = np.ones(k)/k
        lanes.append(np.random.choice(k, size=n, p=p).astype(np.uint8))
    return lanes

# ----------------------------------------------------------------
if __name__ == "__main__":
    lanes = gen_float_stream(1024, (3,5,2,1))
    df = analyse_components(lanes, "Synthetic")
    df.to_csv("all_partitions.csv", index=False)
    print("\nCSV saved → all_partitions.csv")
