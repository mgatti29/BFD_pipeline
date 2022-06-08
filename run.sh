#!/bin/bash 
#SBATCH -A des 
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 2:00:00 
#SBATCH --nodes=20
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=200
module load python
source activate perlmutter_env
cd /global/homes/m/mgatti/BFD_pipeline
srun python run_pipeline_3s.py config_tile.yaml