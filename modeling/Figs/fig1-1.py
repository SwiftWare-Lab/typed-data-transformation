import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

####################
# VLDB-style Format
####################
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

####################
# Entropy Functions
####################
def compute_entropy(window):
    """
    Compute the entropy (in bits) of a numpy array 'window' of bytes.
    """
    freq = Counter(window)
    total = len(window)
    return -sum((count / total) * math.log2(count / total)
                for count in freq.values() if count > 0)

def sliding_window_entropy(byte_array, window_size):
    """
    Compute sliding-window entropy for a byte array.
    Divides 'byte_array' into non-overlapping windows of size 'window_size'
    and computes the entropy for each window.
    Returns a list of entropy values.
    """
    entropies = []
    num_windows = len(byte_array) // window_size
    for i in range(num_windows):
        start = i * window_size
        window = byte_array[start:start + window_size]
        entropies.append(compute_entropy(window))
    # Optionally process the last partial window (if any)
    remainder = len(byte_array) % window_size
    if remainder:
        window = byte_array[-remainder:]
        entropies.append(compute_entropy(window))
    return entropies

####################
# Data Extraction
####################
folder_path = "/mnt/c/Users/jamalids/Downloads/dataset/OBS/fig1"
tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
if not tsv_files:
    raise ValueError("No TSV files found in folder: " + folder_path)

# Use the first TSV file
tsv_file = tsv_files[0]
file_path = os.path.join(folder_path, tsv_file)
print("Processing file:", file_path)

try:
    # Assume the TSV file has no header, and float32 data is in column 1
    df = pd.read_csv(file_path, sep='\t', header=None)
except Exception as e:
    raise RuntimeError("Failed to read TSV file: " + str(e))

# Extract float32 column
float_data = df.values[:, 1].astype(np.float32)
# Convert float array to bytes
byte_array = np.frombuffer(float_data.tobytes(), dtype=np.uint8)

#############################
# Sliding-Window Entropy Calc
#############################
full_window_size = 4*1024*1024 # full: 64 KB window
entropy_full = sliding_window_entropy(byte_array, full_window_size)

# For components: each float32 is 4 bytes
component_window_size = full_window_size // 4  # = 16384

# Split into 4 byte-level components
components = [byte_array[i::4] for i in range(4)]
entropy_components = [
    sliding_window_entropy(comp, component_window_size)
    for comp in components
]

####################
# Plotting
####################
plt.figure(figsize=(6.8, 3))

# Plot full entropy
plt.plot(entropy_full, marker='o', linestyle='-', label="Full Data", linewidth=1)

# Plot components
for i, ent in enumerate(entropy_components):
    plt.plot(ent, marker='o', linestyle='-', label=f"Component {i+1}", linewidth=1)

#plt.xlabel("Window Index (Block Number)")
plt.ylabel("Entropy (bits)")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/entropy_aligned_plot.pdf", bbox_inches='tight')
plt.close()
