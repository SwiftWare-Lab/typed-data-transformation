#!/bin/bash


#BATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="compression"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="compression.%j.%N.out"
#SBATCH --constraint=cascade
#SBATCH -t 7:59:00

#module load StdEnv/2023
#module load gcc/13.3
#module load cmake/3.27.7

module load NiaEnv/.2022a
module load gcc
module load cmake

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

