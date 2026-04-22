#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=01-00:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=random_seed_parallel
#SBATCH --array=0-2
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/model_training/bin/activate # Environment activation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/ssalhi/astro-ml-peek

GLOBAL_OUTPUT_DIR=/home/ssalhi/scratch/crl/experiments
EXP=cifar10
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
MIN_SEED=37
MAX_SEED=47
SEEDS=($(seq $MIN_SEED $MAX_SEED))
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo $CURRENT_SEED
python -m astro_peek.runner.training --config-name=cifar10\
    trainer.seed=$CURRENT_SEED\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/"

