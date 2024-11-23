import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/jamalids/Documents/complier/TA/lab2-4sp4-test/result128.csv'
df = pd.read_csv(file_path)

## Clean up the DataFrame by renaming columns and filtering out unnecessary rows
df.columns = ["name", "density", "iterations", "real_time", "cpu_time", "time_unit",
              "bytes_per_second", "items_per_second", "label", "error_occurred", "error_message"]
df = df[df['name'].str.contains("BM_", na=False)].reset_index(drop=True)

# Extract the benchmark type and density from 'name' column
df['type'] = df['name'].str.extract(r'(BM_SPMM/baseline_spmm|BM_GEMM/spmm_skipping|BM_GEMM/gemm_optimized)')[0]
df['density'] = df['density'].astype(int)

# Legend labels for display
legend_labels = {
    "BM_SPMM/baseline_spmm": "baseline_spmm",
    "BM_GEMM/spmm_skipping": "spmm_skipping",
    "BM_GEMM/gemm_optimized": "gemm"
}

# Plotting with adjusted legend labels
plt.figure(figsize=(10, 6))
for benchmark_type in df['type'].unique():
    subset = df[df['type'] == benchmark_type]
    label = legend_labels.get(benchmark_type, benchmark_type)  # Adjust label for legend only
    plt.plot(subset['density'], subset['real_time'], marker='o', label=label)

# Customize the plot
plt.xlabel('Density')
plt.ylabel('Real Time (ns)')
plt.title('Real Time vs. Density for Different Benchmark Types')
plt.legend(title="Benchmark Type")
plt.grid(True)
plt.show()
