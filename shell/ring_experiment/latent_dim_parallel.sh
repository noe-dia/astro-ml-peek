#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=latent_dim_parallel
#SBATCH --array=0-3
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/links/projects/rrg-lplevass/noedia/classes/crl/astro-ml-peek/astro_peek

GLOBAL_OUTPUT_DIR=/home/noedia/links/scratch/crl/experiments
EXP=rings
OUTPUT_DIR="$GLOBAL_OUTPUT_DIR/$EXP"
MIN_LATENT_DIM=1
MAX_LATENT_DIM=10
LATENT_DIMS=(1 2 10 50)
CURRENT_LATENT_DIM=${LATENT_DIMS[$SLURM_ARRAY_TASK_ID]}

python runner/training.py\
    trainer.seed=42\
    encoder_features.backbone_cfg.output_dim=$CURRENT_LATENT_DIM\
    encoder_labels.backbone_cfg.output_dim=$CURRENT_LATENT_DIM\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/latent_dims"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/latent_dims"

python runner/training.py\
    trainer.seed=43\
    encoder_features.backbone_cfg.output_dim=$CURRENT_LATENT_DIM\
    encoder_labels.backbone_cfg.output_dim=$CURRENT_LATENT_DIM\
    encoder_features.save_dir="$OUTPUT_DIR/models/encoder_features/latent_dims"\
    encoder_labels.save_dir="$OUTPUT_DIR/models/encoder_labels/latent_dims"
