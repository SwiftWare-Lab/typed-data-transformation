#!/bin/bash
#SBATCH --job-name=Compression
#SBATCH --output=Compression.%j.%N.out
#SBATCH --error=Compression.%j.%N.err
#SBATCH --time=05:00:00
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=samira@mcmaster.ca
#SBATCH --export=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64000M

                module load StdEnv

module load python/3.10

source venv/bin/activate

