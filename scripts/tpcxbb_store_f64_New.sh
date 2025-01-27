#!/bin/bash
#SBATCH --job-name=Compression
#SBATCH --output=Compression.%j.%N.out
#SBATCH --error=Compression.%j.%N.err
#SBATCH --time=23:00:00
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=samira@mcmaster.ca
#SBATCH --export=ALL
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1


                module load StdEnv

module load python/3.10

source venv/bin/activate

echo 'Processing /scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/64/tpcxbb_store_f64.tsv'
python3 /scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/decomposition.py --dataset=/scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/64/tpcxbb_store_f64.tsv --outcsv=/scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/logL-32/output_tpcxbb_store_f64.csv --mode=8

