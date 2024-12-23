import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/citytemp_f32_decomposition_stats.csv"
data = pd.read_csv(file_path)

# Define Huffman compression methods
huffman_methods = {
    'decomposed huffman_compress compressed size (B)': 'Decomposed Huffman',
    'reordered huffman_compress compressed size (B)': 'Reordered Huffman',
    'standard huffman_compress compressed size (B)': 'Standard Huffman'
}

# Calculate compression ratios
for method in huffman_methods.keys():
    data[method.replace(' size (B)', ' ratio')] = data['original size'] / data[method]

# Find the best Huffman method decomposition
best_ratios = data[[m.replace(' size (B)', ' ratio') for m in huffman_methods.keys()]]
data['Huffman best ratio'] = best_ratios.max(axis=1)
data['Huffman best method'] = best_ratios.idxmax(axis=1)
data['Huffman best decomposition'] = data['Huffman best method'].apply(lambda x: x.split()[0])

# Extract specific compression ratios for the best overall decomposition method
best_idx = data['Huffman best ratio'].idxmax()
best_decomposition = data.loc[best_idx, 'decomposition']
huffman_ratios = {name: data.loc[best_idx, method.replace(' size (B)', ' ratio')] for method, name in huffman_methods.items()}

# Plotting Huffman Compression Ratios
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(huffman_ratios.keys(), huffman_ratios.values(), color=['red', 'green', 'blue'])
ax.set_xlabel('Huffman Compression Methods')
ax.set_ylabel('Compression Ratio')
ax.set_title(f'Huffman Compression Ratios (Best Decomposition: {best_decomposition})')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', color='black')
plt.show()
############################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/citytemp_f32_decomposition_stats.csv"
data = pd.read_csv(file_path)

# Define Fast LZ compression methods
fastlz_methods = {
    'decomposed fastlz compressed size (B)': 'Decomposed Fast LZ',
    'reordered fastlz compressed size (B)': 'Reordered Fast LZ',
    'standard fastlz compressed size (B)': 'Standard Fast LZ'
}

# Calculate compression ratios
for method in fastlz_methods.keys():
    data[method.replace(' size (B)', ' ratio')] = data['original size'] / data[method]

# Find the best Fast LZ method decomposition
best_ratios = data[[m.replace(' size (B)', ' ratio') for m in fastlz_methods.keys()]]
data['FastLZ best ratio'] = best_ratios.max(axis=1)
data['FastLZ best method'] = best_ratios.idxmax(axis=1)
data['FastLZ best decomposition'] = data['FastLZ best method'].apply(lambda x: x.split()[0])

# Extract specific compression ratios for the best overall decomposition method
best_idx = data['FastLZ best ratio'].idxmax()
best_decomposition = data.loc[best_idx, 'decomposition']
fastlz_ratios = {name: data.loc[best_idx, method.replace(' size (B)', ' ratio')] for method, name in fastlz_methods.items()}

# Plotting Fast LZ Compression Ratios
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(fastlz_ratios.keys(), fastlz_ratios.values(), color=['red', 'green', 'blue'])
ax.set_xlabel('Fast LZ Compression Methods')
ax.set_ylabel('Compression Ratio')
ax.set_title(f'Fast LZ Compression Ratios (Best Decomposition: {best_decomposition})')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', color='black')
plt.show()

###########################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path =  "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/citytemp_f32_decomposition_stats.csv"
data = pd.read_csv(file_path)

# Define RLE compression methods
rle_methods = {
    'decomposed rle compressed size (B)': 'Decomposed RLE',
    'reordered rle compressed size (B)': 'Reordered RLE',
    'standard rle compressed size (B)': 'Standard RLE'
}

# Calculate compression ratios
for method in rle_methods.keys():
    data[method.replace(' size (B)', ' ratio')] = data['original size'] / data[method]

# Find the best RLE method decomposition
best_ratios = data[[m.replace(' size (B)', ' ratio') for m in rle_methods.keys()]]
data['RLE best ratio'] = best_ratios.max(axis=1)
data['RLE best method'] = best_ratios.idxmax(axis=1)
data['RLE best decomposition'] = data['RLE best method'].apply(lambda x: x.split()[0])

# Extract specific compression ratios for the best overall decomposition method
best_idx = data['RLE best ratio'].idxmax()
best_decomposition = data.loc[best_idx, 'decomposition']
rle_ratios = {name: data.loc[best_idx, method.replace(' size (B)', ' ratio')] for method, name in rle_methods.items()}

# Plotting RLE Compression Ratios
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(rle_ratios.keys(), rle_ratios.values(), color=['red', 'green', 'blue'])
ax.set_xlabel('RLE Compression Methods')
ax.set_ylabel('Compression Ratio')
ax.set_title(f'RLE Compression Ratios (Best Decomposition: {best_decomposition})')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', color='black')
plt.show()
