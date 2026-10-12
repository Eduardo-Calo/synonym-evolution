#!/bin/bash
#SBATCH -A NAISS2025-5-227
#SBATCH -p alvis
#SBATCH -t 160:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_err_%j.out

#SBATCH --nodes=1-20
#SBATCH --gres=gpu:A100fat:1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=closest

echo "Job $SLURM_JOB_ID starting"

srun python compute_similarities.py \
    --pos VERB \
    --block-size 16384