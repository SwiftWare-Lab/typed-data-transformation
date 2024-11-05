#!/bin/bash

#SBATCH --cpus-per-task=64
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=msalehi20@gmail.com
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00


module load StdEnv/2023

# build the program
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# run the program
./parallel-test 




