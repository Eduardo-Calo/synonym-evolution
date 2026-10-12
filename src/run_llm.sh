#!/bin/bash
#SBATCH -A NAISS2025-5-227 -p alvis
#SBATCH -t 6:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_err_%j.out

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT_FILE="$1"

python3 hf_annotate_similarity.py \
    --input "$INPUT_FILE" \
    --model models--Qwen--Qwen3.5-27B \
    --trust-remote-code