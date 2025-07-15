#!/bin/bash
#SBATCH --job-name=WDR_MSVM_pump_4
#SBATCH -A gts-ngebraeel3
#SBATCH -N 1 --ntasks-per-node=24
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -q embers
#SBATCH --output=WDR_MSVM_pump_4_%a.out
#SBATCH --array=1-50

module load anaconda3/2022.05
module load gurobi/10.0.1
conda activate gurobi_env_new

python WDR_MSVM_pump_4.py