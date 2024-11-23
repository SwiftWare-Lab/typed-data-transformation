import zstandard as zstd
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import time

def compress_with_zstd(data, level=3, num_threads=1):
    start_time = time.time()
    cctx = zstd.ZstdCompressor(level=level, threads=num_threads)
    compressed = cctx.compress(data)

    original_size = len(data.tobytes())
    compressed_size = len(compressed)
    comp_ratio = original_size / compressed_size

    end_time = time.time()
    runtime = end_time - start_time

    return compressed_size, comp_ratio, runtime

def process_dataset(data, group_name, backend_type, num_threads=10):
    # Compress with Zstd using specified level and threads
    zstd_compressed_ts, comp_ratio_zstd_default, time_default = compress_with_zstd(data, num_threads=num_threads)

    return {
        "Group Name": group_name,
        "Backend": backend_type,
        "Compressed Size (Level 3)": zstd_compressed_ts,
        "Compression Ratio (Level 3)": comp_ratio_zstd_default,
        "Compression Time (Level 3)": time_default
    }

def run_and_collect_data(dataset_path, backend_type, num_threads=1, num_runs=1000):
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    group = ts_data1.drop(ts_data1.columns[0], axis=1)
    group = group.iloc[:2700000, :].T

    # Split into subgroups
    group1 = group.iloc[:, 0:900000]
    group2 = group.iloc[:, 900000:1800000]
    group3 = group.iloc[:, 1800000:2700000]

    # Convert to float32 and reshape
    group = group.astype(np.float32).to_numpy().reshape(-1)
    group1 = group1.astype(np.float32).to_numpy().reshape(-1)
    group2 = group2.astype(np.float32).to_numpy().reshape(-1)
    group3 = group3.astype(np.float32).to_numpy().reshape(-1)

    datasets = [
        (group1, 'group1'),
        (group2, 'group2'),
        (group3, 'group3')
    ]

    df_results = pd.DataFrame()  # Initialize a DataFrame to store results for all runs

    print(f"Running with backend: {backend_type}, using {num_threads} threads for Zstd compression")

    # Run the compression num_runs times
    for run in range(1, num_runs + 1):
        print(f"Run {run} of {num_runs}")

        compressed_size, comp_ratio, runtime = compress_with_zstd(group, level=3, num_threads=1)
        group_result = {
            "Run": run,
            "Group Name": 'group',
            "Backend": 'default',
            "Compressed Size (Level 3)": compressed_size,
            "Compression Ratio (Level 3)": comp_ratio,
            "Compression Time (Level 3)": runtime
        }

        # Parallel processing of subgroups
        start_time = time.time()
        results = Parallel(n_jobs=3, backend=backend_type)(
            delayed(process_dataset)(data, group_name, backend_type, num_threads) for data, group_name in datasets
        )
        end_time = time.time()

        total_runtime = end_time - start_time
        print(f"Total runtime with {backend_type}: {total_runtime:.4f} seconds\n")

        results.append(group_result)  # Add the group result to the results list

        # Convert results to a DataFrame and append to the main DataFrame
        df_run_results = pd.DataFrame(results)
        df_run_results['Run'] = run  # Add a 'Run' column
        df_results = pd.concat([df_results, df_run_results], ignore_index=True)

    return df_results

# Example usage
dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/TS/L/citytemp_f32.tsv"

df_combined = pd.DataFrame()

for backend in ['loky', 'multiprocessing', 'threading']:
    df_backend = run_and_collect_data(dataset_path, backend, num_threads=4, num_runs=1000)
    df_combined = pd.concat([df_combined, df_backend], ignore_index=True)

print("Combined Results DataFrame:")
print(df_combined)

# Save to CSV
output_csv = 'combined_zstd_compression_results_1000_runs.csv'
df_combined.to_csv(output_csv, index=False)
print(f"Results saved to '{output_csv}'")
