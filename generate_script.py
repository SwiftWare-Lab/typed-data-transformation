import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate SBATCH job scripts for each dataset.")
parser.add_argument("--dataset", required=True, help="Path to the dataset directory containing .tsv files.")
parser.add_argument("--outcsv", required=True, help="Directory where output CSV files will be stored.")
parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for processing each dataset.")
parser.add_argument("--bits", type=int, default=32, help="Number of bits for  each dataset.")
# Parse arguments
args = parser.parse_args()
DATASET_DIR = os.path.abspath(args.dataset)  # Use absolute path
OUTPUT_DIR = os.path.abspath(args.outcsv)  # Use absolute path
NUM_THREADS = args.threads
bits = args.bits
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

for dataset_file in os.listdir(DATASET_DIR):
    if dataset_file.endswith(".tsv"):
        dataset_name = os.path.splitext(dataset_file)[0]
        outcsv = os.path.join(OUTPUT_DIR, f"{dataset_name}.csv")
        job_script = os.path.join(CURRENT_DIR, f"{dataset_name}_job.sh")
        dataset_path = os.path.join(DATASET_DIR, dataset_file)

        # Write the job script
        with open(job_script, "w") as script_file:
            script_file.write(f"""#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="compression"
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="compression.%j.%N.out"
#SBATCH --time 20:59:00
# Load necessary modules
module load StdEnv/.2022a
module load gcc/13.2
module load cmake

# Debugging info
echo "Running on dataset: {dataset_path}"
echo "Output CSV will be saved to: {outcsv}"

# Run the program with the dataset
./build/external_tools/parallel-test "{dataset_path}" "{outcsv}" {NUM_THREADS} {bits}
./build/external_tools/parallel-test --dataset "{dataset_path}" --outcsv "{outcsv}" --threads {NUM_THREADS} --bits {bits} 
""")

