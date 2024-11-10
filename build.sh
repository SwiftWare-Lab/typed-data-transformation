#!/bin/bash

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


mkdir -p build
cd build
/home/jamalids/programs/cmake-3.31.0-rc3-linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

# Set directories
DATASET_DIR="/home/jamalids/Documents/2D/data1/Fcbench/HPC/H"
OUTPUT_DIR="/home/jamalids/Documents/compression-part4/new1/new/resultba"
NUM_THREADS=10


mkdir -p "$OUTPUT_DIR"

# TSV file
for dataset_file in "$DATASET_DIR"/*.tsv; do
    dataset_name=$(basename "$dataset_file" .tsv)
    outcsv="$OUTPUT_DIR/${dataset_name}.csv"
    job_script="$OUTPUT_DIR/${dataset_name}_job.sh"

    # Create a job script for each dataset
    cat <<EOT > "$job_script"
#!/bin/bash
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
./build/external_tools/parallel-test "$dataset_file" "$outcsv" $NUM_THREADS
./build/external_tools/parallel-test --dataset "$dataset_file" --outcsv "$outcsv"  --threads $NUM_THREADS
EOT

    # Make job script executable
    chmod +x "$job_script"

done

