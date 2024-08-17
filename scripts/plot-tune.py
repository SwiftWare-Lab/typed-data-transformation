# Let's create plots for different compression ratios based on M and N, and another plot to show different sizes.
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results11.csv')

# Let's generate multiple plots based on the available data

# Plot 1: Compression Ratios based on M and N
plt.figure(figsize=(12, 6))
for n_value in df['N'].unique():
    subset = df[df['N'] == n_value]
    plt.plot(subset['M'], subset['com_ratio_b1'], marker='o', label=f'com_ratio_b1 (N={n_value})')
    plt.plot(subset['M'], subset['com_ratio_b2'], marker='o', label=f'com_ratio_b2 (N={n_value})')
    plt.plot(subset['M'], subset['com_ratio_b3'], marker='o', label=f'com_ratio_b3 (N={n_value})')

plt.xlabel('M')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratios based on  pattern Sizes ( M and N )')
plt.legend()
plt.grid(True)
plt.savefig('1')
plt.show()

# Plot 2: Sizes based on M and N
plt.figure(figsize=(12, 6))
for n_value in df['N'].unique():
    subset = df[df['N'] == n_value]
    plt.plot(subset['M'], subset['Original Size (bits)'], marker='o', label=f'Original Size (N={n_value})')
    plt.plot(subset['M'], subset['b1_leading_encoded_size'] + subset['b1_content_encoded_size'] + subset['b1_trailing_encoded_size'],
             marker='o', label=f'Total Encoded Size for b1 (N={n_value})')
    plt.plot(subset['M'], subset['b2_leading_encoded_size'] + subset['b2_content_encoded_size'] + subset['b2_trailing_encoded_size'],
             marker='o', label=f'Total Encoded Size for b2 (N={n_value})')
    plt.plot(subset['M'], subset['b3_leading_encoded_size'] + subset['b3_content_encoded_size'] + subset['b3_trailing_encoded_size'],
             marker='o', label=f'Total Encoded Size for b3 (N={n_value})')

plt.xlabel('M')
plt.ylabel('Size (bits)')
plt.title('pattern Sizes ( M and N )and different Tune size b')
plt.legend()
plt.grid(True)

plt.savefig('2')
plt.show()


