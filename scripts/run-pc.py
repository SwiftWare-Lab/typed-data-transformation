import os
import time
import subprocess

DATASET_DIR = "home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"  # Change to your dataset folder
RESULTS_DIR = "home/jamalids/Documents"  # Change to where you want to save results

os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results folder if it doesn't exist

for dataset in sorted(os.listdir(DATASET_DIR)):  # Sort files alphabetically
    dataset_path = os.path.join(DATASET_DIR, dataset)

    if os.path.isfile(dataset_path):  # Ensure it's a file
        print(f"Processing {dataset_path}...")

        start_time = time.time()
        result = subprocess.run(["/home/jamalids/Documents/2D/final results/big-data-compression/external_tools/parallel-test", dataset_path], capture_output=True, text=True)
        end_time = time.time()

        elapsed_time = end_time - start_time
        result_file = os.path.join(RESULTS_DIR, f"{dataset}.txt")

        with open(result_file, "w") as f:
            f.write(result.stdout)  # Save program output
            f.write(f"\nDataset: {dataset_path}\n")
            f.write(f"Processing Time: {elapsed_time:.4f} sec\n")

        print(f"Results saved in: {result_file}")
