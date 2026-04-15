#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=random_seed_parallel
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation


cd /home/noedia/projects/def-lplevass/noedia/crl/astro-ml-peek/astro_peek/runner

MIN_SEED=37
MAX_SEED=47
for SEED in $(seq $MIN_SEED $MAX_SEED); do 
    python training.py\
    trainer.seed=$SEED
    
done