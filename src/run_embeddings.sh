#!/bin/bash
#SBATCH -A NAISS2025-5-227 -p alvis
#SBATCH -t 160:00:00
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_err_%j.out

python3 embed_definitions.py \
    --input-dir ../data/parsed \
    --output-dir ../data/embedding \
    --pos VERB