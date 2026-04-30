#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00         # time (DD-HH:MM)
#SBATCH --account=ctb-lplevass
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=0-4
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/projects/def-lplevass/noedia/crl/astro-ml-peek/astro_peek

GLOBAL_OUTPUT_DIR=/home/noedia/scratch/crl/experiments 
EXP=strong_lenses 
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
MIN_SEED=50
MAX_SEED=54
SEEDS=($(seq $MIN_SEED $MAX_SEED))
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo $CURRENT_SEED
python runner/training.py --config-name=$EXP\
    trainer.seed=$CURRENT_SEED\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/"
