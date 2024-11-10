#!/bin/bash

#SBATCH --cpus-per-task=64
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00


module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7



# build the program
mkdir build
cd build
#/home/jamalids/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
/home/jamalids/programs/cmake-3.31.0-rc3-linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ..

# run the program
./build/external_tools/parallel-test




