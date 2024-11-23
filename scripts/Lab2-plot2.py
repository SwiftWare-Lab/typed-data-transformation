#############################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/jamalids/Documents/complier/TA/lab2-4sp4-test/result256.csv'
df = pd.read_csv(file_path)

# Clean up the DataFrame by renaming columns and filtering out unnecessary rows
df.columns = ["name", "density", "iterations", "real_time", "cpu_time", "time_unit",
              "bytes_per_second", "items_per_second", "label", "error_occurred", "error_message"]
df = df[df['name'].str.contains("BM_", na=False)].reset_index(drop=True)

# Extract the benchmark type and density from 'name' column
df['type'] = df['name'].str.extract(r'(BM_SPMM/baseline_spmm|BM_GEMM/spmm_skipping|BM_GEMM/gemm_optimized)')[0]
df['density'] = df['density'].astype(int)

# Filter the DataFrame to include only "baseline_spmm" type benchmarks
spmm_df = df[df['type'] == 'BM_SPMM/baseline_spmm']

# Extract NNZ from the 'label' column
spmm_df['NNZ'] = spmm_df['label'].str.extract(r'NNZ=(\d+)').astype(float)

# Plot Real Time vs Density
plt.figure(figsize=(12, 6))

# Real Time vs Density plot
plt.subplot(1, 2, 1)
plt.plot(spmm_df['density'], spmm_df['real_time'], marker='o', color='b', label="Real Time (ns)")
plt.xlabel('Density')
plt.ylabel('Real Time (ns)')
plt.title('Real Time vs. Density for Baseline SPMM')
plt.legend()
plt.grid(True)

# NNZ vs Density plot
plt.subplot(1, 2, 2)
plt.plot(spmm_df['density'], spmm_df['NNZ'], marker='o', color='g', label="NNZ")
plt.xlabel('Density')
plt.ylabel('NNZ')
plt.title('NNZ vs. Density for Baseline SPMM')
plt.legend()
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()
