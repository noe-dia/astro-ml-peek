#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=create_lognormal
#SBATCH --array=0-10
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/links/projects/rrg-lplevass/noedia/classes/crl/astro-ml-peek/astro_peek/datasets

OUTPUT_DIR=/home/noedia/links/scratch/crl/experiments/lognormal_bis

python make_lognormal.py --output_dir=$OUTPUT_DIR/$SLURM_ARRAY_TASK_ID --num_cosmo=100