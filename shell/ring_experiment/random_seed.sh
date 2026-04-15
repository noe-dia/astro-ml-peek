#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=1
#SBATCH --output=jobout/%x_%A_%a.out



module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/projects/def-lplevass/noedia/crl/astro-ml-peek/astro_peek/runner

GLOBAL_OUTPUT_DIR=/home/noedia/scratch/crl/ring_experiment
EXP=rings
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
SEED=42
python training.py\
    trainer.seed=$SEED\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/seed_$SEED.pt"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/seed_$SEED.pt"


    
