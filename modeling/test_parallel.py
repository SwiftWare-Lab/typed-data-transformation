import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time

# Define a matrix multiplication function
def matmul_task(A, B):
    return np.dot(A, B)

# Function to run and time matrix multiplication with different backends
def run_matmul_parallel(backend, n_jobs=4, size=1000):
    # Create random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Start timing
    start_time = time.time()

    # Perform matrix multiplication in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(matmul_task)(A, B) for _ in range(n_jobs)
    )

    # End timing
    end_time = time.time()

    return end_time - start_time

# List of backends to test
backends = ['loky', 'threading', 'multiprocessing']

# Run matrix multiplication for each backend and collect results
times = [run_matmul_parallel(backend) for backend in backends]

# Create a DataFrame to store the results
df_results = pd.DataFrame({
    'Backend': backends,
    'Execution Time (seconds)': times
})

# Display the DataFrame
print(df_results)
################################################################
from joblib import Parallel, delayed
import numpy as np


# Assuming all helper functions are already defined:
# decompose_array_three, calculate_entropy, bits_to_float32, compress_with_zstd, compress_with_gzip

def run_compression(backend, image_ts, leading_zero_pos, tail_zero_pos, funct_name):
    # Run the decomposition_based_compression1 function
    return decomposition_based_compression1(image_ts, leading_zero_pos, tail_zero_pos, funct_name)


# Define a function to run parallel jobs for leading, content, and trailing components
def parallel_compression(image_ts, leading_zero_pos, tail_zero_pos):
    backends = ['loky', 'threading', 'multiprocessing']
    funct_names = ['zstd', 'zstd_22', 'gzip']  # Compression functions to use

    results = {}

    # Run the parallel processing for each backend
    for backend in backends:
        print(f"Running with backend: {backend}")
        results[backend] = Parallel(n_jobs=3, backend=backend)(
            delayed(run_compression)(image_ts, leading_zero_pos, tail_zero_pos, funct_name)
            for funct_name in funct_names
        )

    return results


# Example input (replace these with actual values)
image_ts = np.random.rand(100, 32)  # Replace with actual time-series data
leading_zero_pos = np.random.randint(0, 32, size=100)  # Replace with actual leading zero positions
tail_zero_pos = np.random.randint(0, 32, size=100)  # Replace with actual tail zero positions

# Run the function
results = parallel_compression(image_ts, leading_zero_pos, tail_zero_pos)

# Display results for each backend
for backend, backend_results in results.items():
    print(f"Backend: {backend}")
    for funct_name, result in zip(['zstd', 'zstd_22', 'gzip'], backend_results):
        leading, content, tailing, leading_R, content_R, tailing_R, lead_entropy, tail_entropy, content_entropy, leading_time, content_time, trailing_time = result
        print(f"\nFunction: {funct_name}")
        print(f"Leading Compression Ratio: {leading_R}, Time: {leading_time}")
        print(f"Content Compression Ratio: {content_R}, Time: {content_time}")
        print(f"Trailing Compression Ratio: {tailing_R}, Time: {trailing_time}")
