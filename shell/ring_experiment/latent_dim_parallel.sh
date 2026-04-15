#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-12:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=1
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation


cd /home/noedia/projects/def-lplevass/noedia/crl/astro-ml-peek/astro_peek/runner

MIN_LATENT_DIM=1
MAX_LATENT_DIM=20
for LATENT_DIM in $(seq $MIN_LATENT_DIM $MAX_LATENT_DIM); do 
    python training.py\
    encoder_features.backbone_cfg.output_dim=$LATENT_DIM\
    encoder_labels.backbone_cfg.output_dim=$LATENT_DIM
done