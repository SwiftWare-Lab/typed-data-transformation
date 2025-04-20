import os
import math
import numpy as np
import pandas as pd
from collections import Counter

# === Entropy calculation for a sequence of float32 values ===
def compute_entropy_float(float_array):
    # Convert float32 array to raw bytes
    raw_bytes = float_array.tobytes()
    # Convert to uint8 array to count frequency
    byte_values = np.frombuffer(raw_bytes, dtype=np.uint8)
    freq = Counter(byte_values)
    total = len(byte_values)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return entropy

# === Main entropy measurement function ===
def run_entropy_only(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder:", folder_path)
        return

    results = []

    tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
    if not tsv_files:
        print("No .tsv files found in", folder_path)
        return

    for fname in tsv_files:
        dataset_name = os.path.splitext(fname)[0]
        fpath = os.path.join(folder_path, fname)
        print(f"Processing: {dataset_name}")

        try:
            df = pd.read_csv(fpath, sep='\t', header=None)
            float_vals = df.values[:, 1].astype(np.float32)  # Assuming 2nd column
            entropy = compute_entropy_float(float_vals)
            results.append({"Dataset": dataset_name, "Entropy": entropy})
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    out_csv = os.path.join("/home/jamalids/Documents", "float_level_entropy_results32.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Entropy results saved to: {out_csv}")

# === Run directly ===
if __name__ == "__main__":
    folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
    run_entropy_only(folder_path)
