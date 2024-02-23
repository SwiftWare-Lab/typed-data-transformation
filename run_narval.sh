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



# Loading module
module load StdEnv
module load python/3.10
module load cmake

echo "-- Installing --"
python3 --version
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

echo "running the code"
python3 Timeseries_final.py
