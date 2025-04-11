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
    return -sum((count/total) * math.log2(count/total)
                for count in freq.values() if count > 0)

def sliding_window_entropy(byte_array, window_size):
    """
    Compute sliding-window entropy for a byte array.
    Divides 'byte_array' into non-overlapping windows of size 'window_size'
    and computes the entropy for each window.
    """
    entropies = []
    num_windows = len(byte_array) // window_size
    for i in range(num_windows):
        start = i * window_size
        window = byte_array[start:start + window_size]
        entropies.append(compute_entropy(window))
    # Optionally compute entropy for the last partial window:
    remainder = len(byte_array) % window_size
    if remainder:
        window = byte_array[-remainder:]
        entropies.append(compute_entropy(window))
    return entropies

####################
# Data Extraction
####################

# Set the folder path containing your TSV files
folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
tsv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tsv')]
if not tsv_files:
    raise ValueError("No TSV files found in folder: " + folder_path)

# Use the first TSV file for this example
tsv_file = tsv_files[0]
file_path = os.path.join(folder_path, tsv_file)
print("Processing file:", file_path)

try:
    # Assume the TSV file has no header and the float32 data is in column index 1.
    df = pd.read_csv(file_path, sep='\t', header=None)
except Exception as e:
    raise RuntimeError("Failed to read TSV file: " + str(e))

# Extract the float32 data from column 1 and convert to a numpy array
float_data = df.values[:, 1].astype(np.float32)
# Flatten the array and convert it to bytes
flattened_bytes = float_data.flatten().tobytes()
# Interpret these bytes as an array of uint8 values
byte_array = np.frombuffer(flattened_bytes, dtype=np.uint8)

####################
# Entropy Computation
####################

# Set the window size to 64 KB (65,536 bytes)
window_size = 65536

# Compute sliding-window entropy for the full dataset (all bytes)
entropy_full = sliding_window_entropy(byte_array, window_size)

# For 32-bit floats, there are 4 bytes per number.
# Divide the entire byte array into 4 components:
components = []
for i in range(4):
    # Extract every 4th byte, starting at position i.
    comp_i = byte_array[i::4]
    components.append(comp_i)

# Compute sliding-window entropy for each component
entropy_components = [sliding_window_entropy(comp, window_size) for comp in components]

####################
# Plotting
####################

# Create 5 subplots arranged horizontally with a shared y-axis.
fig, axs = plt.subplots(1, 5, sharey=True, figsize=(25, 5))

# Subplot 1: Entropy for the full dataset.
axs[0].plot(entropy_full, marker='o', linestyle='-')
axs[0].set_title("Full Data")
axs[0].set_xlabel("Window Index")
axs[0].set_ylabel("Entropy (bytes)")

# Subplots 2-5: Entropy for each of the 4 components.
for i in range(4):
    axs[i+1].plot(entropy_components[i], marker='o', linestyle='-')
    axs[i+1].set_title(f"bytes {i+1}")
    axs[i+1].set_xlabel("Window Index")

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy-plot")
#########################
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
    return -sum((count/total) * math.log2(count/total) for count in freq.values() if count > 0)

def sliding_window_entropy(byte_array, window_size):
    """
    Compute sliding-window entropy for a byte array.
    Divides 'byte_array' into non-overlapping windows of size 'window_size'
    and computes the entropy for each window.
    """
    entropies = []
    num_windows = len(byte_array) // window_size
    for i in range(num_windows):
        start = i * window_size
        window = byte_array[start:start+window_size]
        entropies.append(compute_entropy(window))
    # Optionally process the last partial window:
    remainder = len(byte_array) % window_size
    if remainder:
        window = byte_array[-remainder:]
        entropies.append(compute_entropy(window))
    return entropies

####################
# Data Extraction
####################

# Set your folder path (adjust if needed)
folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
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
# Convert the float32 data to a byte array.
flattened_bytes = float_data.flatten().tobytes()
byte_array = np.frombuffer(flattened_bytes, dtype=np.uint8)

####################
# Entropy Computation
####################

# Set window size to 64 KB (65,536 bytes)
window_size = 65536

# Compute sliding-window entropy for the full byte array.
entropy_full = sliding_window_entropy(byte_array, window_size)

# For 32-bit floats, split the byte array into four components.
components = [ byte_array[i::4] for i in range(4) ]
# Compute sliding-window entropy for each component.
entropy_components = [ sliding_window_entropy(comp, window_size) for comp in components ]

####################
# Plotting: All Curves in One Plot
####################

plt.figure(figsize=(12, 6))

# Plot entropy for the full dataset.
plt.plot(entropy_full, marker='o', linestyle='-', label="Full Data")

# Plot entropy for each of the 4 components.
for i, ent in enumerate(entropy_components):
    plt.plot(ent, marker='o', linestyle='-', label=f"Component {i+1}")

plt.xlabel("Window Index")
plt.ylabel("Entropy (bits)")
plt.title("Sliding-Window Entropy (Full Data and Components)")
plt.legend()
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy1-plot.png")
######################################
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
folder_path = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
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

plt.figure(figsize=(12, 6))

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
plt.savefig("/home/jamalids/Documents/entropy_aligned_plot.png")
plt.show()
