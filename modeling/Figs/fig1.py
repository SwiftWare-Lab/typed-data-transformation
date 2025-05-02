import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------
# VLDB Plot Formatting
# ------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 22,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------
# Dataset and Preparation
# ------------------------------
dataset_path = "/mnt/c/Users/jamalids/Downloads/dataset/OBS/jw_mirimage_f32.tsv"
dataset_name = os.path.basename(dataset_path).split('.')[0]

# Load dataset
data_set = pd.read_csv(dataset_path, sep='\t')

# Slice and cast to float32
sliced_data = data_set.values[61640:100000, 1].astype(np.float32)

# Define chunk size
element_size = np.dtype(np.float32).itemsize  # 4 bytes
chunk_size_bytes = 128 * 1024  # 128KB
elements_per_chunk = chunk_size_bytes // element_size

# Determine number of chunks
num_chunks = int(np.ceil(len(sliced_data) / elements_per_chunk))

# ------------------------------
# Generate one plot per chunk
# ------------------------------
for i in range(num_chunks):
    start_idx = i * elements_per_chunk
    end_idx = min((i + 1) * elements_per_chunk, len(sliced_data))
    chunk_data = sliced_data[start_idx:end_idx]

    # Create separate VLDB-style plot
    plt.figure(figsize=(6.8, 3))
    plt.plot(chunk_data, marker='o', linestyle='-', color='blue', markersize=2)
   # plt.xlabel("Index")
   # plt.ylabel("Value")
   # plt.grid(True)
   # plt.title(f"{dataset_name}: Values {start_idx}–{end_idx}")

    # Save each chunk to its own file
    output_path = f"/mnt/c/Users/jamalids/Downloads/M{i+1}.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print("Saved VLDB-style plots for each chunk.")
#######################################################################################
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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

# Set your folder path containing your TSV files (adjust if needed)
folder_path = "/mnt/c/Users/jamalids/Downloads/dataset/OBS/fig1"
tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
if not tsv_files:
    raise ValueError("No TSV files found in folder: " + folder_path)

# Use the first TSV file for this example.
tsv_file = tsv_files[0]
file_path = os.path.join(folder_path, tsv_file)
print("Processing file:", file_path)

try:
    # Assume the TSV file has no header and that the float32 data is in column index 1.
    df = pd.read_csv(file_path, sep='\t', header=None)
except Exception as e:
    raise RuntimeError("Failed to read TSV file: " + str(e))

# Extract the float32 data from the second column.
float_data = df.values[:, 1].astype(np.float32)
# Convert the float32 array to bytes.
flattened_bytes = float_data.flatten().tobytes()
# Interpret these bytes as an array of uint8.
byte_array = np.frombuffer(flattened_bytes, dtype=np.uint8)

####################
# Entropy Computation
####################

# For the full data, use a window size of 65,536 bytes.
full_window_size = 65536
entropy_full = sliding_window_entropy(byte_array, full_window_size)

# For 32-bit floats, each float is 4 bytes.
# To align the windows, compute the window size for each component as (full_window_size / 4)
component_window_size = full_window_size // 4  # i.e., 16384 bytes

# Split the byte array into 4 components.
components = [ byte_array[i::4] for i in range(4) ]
# Compute sliding-window entropy for each component using component_window_size.
entropy_components = [ sliding_window_entropy(comp, component_window_size) for comp in components ]

####################
# Plotting: All Curves in One Plot
####################

#plt.figure(figsize=(12, 6))

# Plot the full data entropy curve.
plt.plot(entropy_full, marker='o', linestyle='-', label="Full Data")
# Plot each of the 4 components.
for i, ent in enumerate(entropy_components):
    plt.plot(ent, marker='o', linestyle='-', label=f"Component {i+1}")

plt.xlabel("Window Index (Block Number)")
plt.ylabel("Entropy (bytes)")
plt.title("Sliding-Window Entropy (Aligned Windows)")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/entropy_aligned_plot.png")
plt.show()
