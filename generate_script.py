import os

# Define directories and parameters
DATASET_DIR = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H"
OUTPUT_DIR = "/home/jamalids/Documents/compression-part4/new1/new/resultba1"
NUM_THREADS = 10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all .tsv files in the dataset directory
for dataset_file in os.listdir(DATASET_DIR):
    if dataset_file.endswith(".tsv"):
        dataset_name = os.path.splitext(dataset_file)[0]
        outcsv = os.path.join(OUTPUT_DIR, f"{dataset_name}.csv")
        job_script = os.path.join(OUTPUT_DIR, f"{dataset_name}_job.sh")
        dataset_path = os.path.join(DATASET_DIR, dataset_file)

        # Write the job script
        with open(job_script, "w") as script_file:
            script_file.write(f"""#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --export=ALL
#SBATCH --job-name="compression"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="compression.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00

#module load StdEnv/2023
#module load gcc/13.3
#module load cmake/3.27.7

# Run the program with the dataset
./build/external_tools/parallel-test "{dataset_path}" "{outcsv}" {NUM_THREADS}
./build/external_tools/parallel-test --dataset "{dataset_path}" --outcsv "{outcsv}" --threads {NUM_THREADS}
""")

        # Make the job script executable
        os.chmod(job_script, 0o755)

print("Job scripts created successfully.")
