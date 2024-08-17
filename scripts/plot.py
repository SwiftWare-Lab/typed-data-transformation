
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('results12.csv')

# Assigning a single dataset name for the whole dataset
df['dataset_name'] = 'ACSF1'

# Plot 1: Compression Ratio with and without Dictionary
plt.figure(figsize=(12, 6))

# Adjusting the bar width for comparison
bar_width = 0.1
index = range(len(df))

plt.bar(index, df['com_ratio'], bar_width, label='Com Ratio without Dictionary')
plt.bar([i + bar_width for i in index], df['com_ratio_dict'], bar_width, label='Com Ratio with Dictionary')
plt.bar([i + 2 * bar_width for i in index], df['com_ratio_tot_d '], bar_width, label='Normal Com Ratio with Dictionary')
plt.bar([i + 3 * bar_width for i in index], df['com_ratio_tot'], bar_width, label='Normal Com Ratio ')
plt.bar([i + 4 * bar_width for i in index], df['comp_ratio_zstd_default'], bar_width, label='comp_ratio_zstd_default')
plt.bar([i + 5 * bar_width for i in index], df['comp_ratio_l22'], bar_width, label='comp_ratio_l22')

plt.title('Compression Ratio ')
plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio')
plt.xticks([i + bar_width / 2 for i in index], df['dataset_name'])
plt.legend()
plt.show()
plt.savefig('CompressionRatio.png')
