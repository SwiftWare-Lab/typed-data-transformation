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

echo 'Processing /scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/32/rsim_f32.tsv'
python3 /scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/decomposition-h.py --dataset=/scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/32/rsim_f32.tsv --outcsv=/scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/logL-32/output_rsim_f32.csv --mode=4

echo 'Processing /scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/32/citytemp_f32.tsv'
python3 /scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/decomposition-h.py --dataset=/scratch/k/kazem/samiraj/big-data-compression/Low-Entropy/32/citytemp_f32.tsv --outcsv=/scratch/k/kazem/samiraj/python-script/big-data-compression/modeling/logL-32/output_citytemp_f32.csv --mode=4

