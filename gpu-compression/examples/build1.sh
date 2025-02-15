#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --export=ALL
#SBATCH --job-name="gdeflate_cpu_compression"
#SBATCH --mail-type=begin  # Email when the job starts
#SBATCH --mail-type=end    # Email when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="gdeflate_cpu_compression.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00

module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7

# Define paths
DATASET_DIR="/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/64"
EXECUTABLE="/home/jamalids/development/big-data-compression/gpu-compression/examples/cmake-build-debug/gdeflate_cpu_compression"
RESULTS_DIR="/home/jamalids/Documents/results1"

mkdir -p "$RESULTS_DIR"

echo "=====> Starting gdeflate_cpu_compression execution for all datasets <====="

# Loop over every TSV file in the dataset directory.
for dataset in "$DATASET_DIR"/*.tsv; do
    if [ -f "$dataset" ]; then
        dataset_name=$(basename "$dataset" .tsv)
        log_file="$RESULTS_DIR/${dataset_name}_run.log"

        echo "Processing dataset: $dataset"
        echo "Dataset: $dataset" > "$log_file"

        start_time=$(date +%s.%N)

        # Run the executable with arguments: dataset, 64, 262144, 1
        "$EXECUTABLE" "$dataset" 64 262144 1

        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)

        echo "Execution Time: $elapsed_time seconds" >> "$log_file"
        echo "Results saved in: $log_file"

        echo "Processed dataset: $dataset (Elapsed Time: $elapsed_time seconds)"
    fi
done

echo "=====> All datasets processed successfully <====="
