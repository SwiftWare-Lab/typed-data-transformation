import os
import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
import zlib
from compression_tools import fastlz_compress, huffman_compress, zstd_comp

# --------------------------- Entropy & Delta Helpers --------------------------- #

def compute_entropy(arr):
    freq, total = Counter(arr), len(arr)
    return -sum((cnt/total) * math.log2(cnt/total) for cnt in freq.values() if cnt > 0)

def compute_kth_entropy(arr, k=2):
    if len(arr) < k:
        return 0.0
    grams = [tuple(arr[i:i+k]) for i in range(len(arr)-k+1)]
    freq, total = Counter(grams), len(grams)
    return -sum((cnt/total) * math.log2(cnt/total) for cnt in freq.values())

def delta_k_entropy(global_stream, cluster_streams, k=2):
    H_gl = compute_kth_entropy(global_stream, k)
    total = len(global_stream)
    Hw = sum(len(c)*compute_kth_entropy(c, k) for c in cluster_streams)/total
    return H_gl - Hw

def delta_H0(global_stream, cluster_streams):
    H0 = compute_entropy(global_stream)
    total = len(global_stream)
    Hw0 = sum(len(c)*compute_entropy(c) for c in cluster_streams)/total
    return H0 - Hw0

# --------------------------- Feature Extractors --------------------------- #

def extract_freq(g, **_):
    freq = Counter(g)
    probs = np.array([freq.get(b,0)/len(g) for b in range(256)])
    return np.array([probs.std()])

def extract_ent(g, window_size=256, **_):
    ents = [compute_entropy(g[i:i+window_size]) for i in range(0, len(g)-window_size+1, window_size)]
    return np.array([np.mean(ents)])

def extract_all(g, global_stream=None, window_size=256, **_):
    return np.concatenate((extract_freq(g), extract_ent(g, window_size=window_size)))

def extract_delta(g, global_stream, **_):
    arr = np.frombuffer(g.tobytes(), dtype=np.uint8)
    return np.array([delta_k_entropy(global_stream,[arr],k=2), delta_H0(global_stream,[arr])])

# --------------------------- Data Splitting --------------------------- #

def split_groups(byte_data, sizes):
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    stride = sum(sizes)
    return [arr[offset::stride] for offset in np.cumsum([0]+sizes[:-1])]

# --------------------------- Core Analysis --------------------------- #

def run_analysis(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Compressor mappings
    compressors = {
        'fastlz': fastlz_compress,
        'huffman': huffman_compress,
        'zstd': zstd_comp
    }

    tsvs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.tsv'))
    for tsv in tsvs:
        base = os.path.splitext(tsv)[0]
        df = pd.read_csv(os.path.join(folder_path, tsv), sep='\t', header=None)
        data = df.iloc[:,1:].to_numpy(dtype=np.uint8).flatten(order='F').tobytes()
        groups = split_groups(data, [1,1,1,1])
        full_bytes = data
        global_stream = np.frombuffer(full_bytes, dtype=np.uint8)

        # Feature modes
        modes = {
            'frequency': extract_freq,
            'entropy': extract_ent,
            'all': extract_all,
            'delta': extract_delta
        }

        for comp_name, comp_func in compressors.items():
            hc_records = []
            for mode, feat_func in modes.items():
                # build features
                feats = np.vstack([feat_func(g, global_stream=global_stream) for g in groups])
                feats += np.random.default_rng(0).normal(scale=1e-8, size=feats.shape)
                L = linkage(feats, method='complete')
                n = len(groups)

                # record HClust configs and decomposed ratios
                for k in range(1, n+1):
                    labs = fcluster(L, k, criterion='maxclust') if k>1 else np.ones(n,int)
                    part = tuple(tuple(sorted(np.where(labs==c)[0].tolist())) for c in sorted(set(labs)))
                    # decomposed ratio using this compressor
                    dec = []
                    for blk in part:
                        arr2d = np.stack([groups[i] for i in blk], axis=1)
                        bts   = arr2d.flatten(order='F').tobytes()
                        dec.append((comp_func(bts)))
                    ratio = len(full_bytes) / sum(dec)
                    hc_records.append({'Mode': mode,
                                       'K': k,
                                       'Partition': str(part),
                                       'DecomposedRatio': ratio})

            out_df = pd.DataFrame(hc_records)
            out_csv = os.path.join(folder_path, f"{base}_hc_{comp_name}.csv")
            out_df.to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")

if __name__ == '__main__':
    folder = '/mnt/c/Users/jamalids/Downloads/dataset/OBS/test'
    run_analysis(folder)
