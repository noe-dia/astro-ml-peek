#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=0-2
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/links/projects/rrg-lplevass/noedia/classes/crl/astro-ml-peek/astro_peek

GLOBAL_OUTPUT_DIR=/home/noedia/links/scratch/crl/experiments 
EXP=lognormal 
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
SEEDS=(40 42 28)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo $CURRENT_SEED
python runner/training.py --config-name=$EXP\
    trainer.seed=$CURRENT_SEED\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/"
