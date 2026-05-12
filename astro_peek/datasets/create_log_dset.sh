#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=create_lognormal
#SBATCH --array=0-49
#SBATCH --output=jobout/%x_%A_%a.out

module load arrow/16 cuda/12.6
source $HOME/causal_env/bin/activate # Environment activation

cd /home/noedia/links/projects/rrg-lplevass/noedia/classes/crl/astro-ml-peek/astro_peek/datasets

OUTPUT_DIR=/home/noedia/links/scratch/crl/experiments/lognormal_bis_2
SEEDS=( $(seq 0 200 9999) )
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Worker seed: $CURRENT_SEED" 
python make_lognormal.py --output_dir=$OUTPUT_DIR/$SLURM_ARRAY_TASK_ID --num_cosmo=20 --seed_init=$CURRENT_SEED