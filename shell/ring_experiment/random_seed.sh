#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=1
#SBATCH --output=jobout/%x_%A_%a.out



module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/links/projects/rrg-lplevass/noedia/classes/crl/astro-ml-peek/astro_peek

GLOBAL_OUTPUT_DIR=/home/noedia/links/scratch/crl/experiments
EXP=rings
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
SEED=42

python runner/training.py\
    trainer.seed=$SEED\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/"    
