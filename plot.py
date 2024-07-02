
import pandas as pd
import matplotlib.pyplot as plt
df_results=pd.read_csv('results.csv')

plt.figure(figsize=(10, 6))


plt.plot(df_results['Original Size (bytes)'], df_results['pattern_comp'], marker='o', label="pattern+dict")
plt.plot(df_results['Original Size (bytes)'], df_results['pattern_comp_dict_zstd'], marker='s', label="pattern+zstd-dict")
plt.plot(df_results['Original Size (bytes)'], df_results['zstd_comp_size'] ,marker='^', label="zstd-size")
plt.plot(df_results['Original Size (bytes)'], df_results['pattern_comp_dict_int_zstd'], marker='d', label="pattern+zstd-arraydict")

plt.xlabel("Original Size (bytes)")
plt.ylabel("Decompressed Size (bytes)")
plt.title("Decompressed Size vs Original Size")


plt.legend()


plt.grid(True)
plt.show()
