#!/bin/bash
# Submits four separate SBATCH jobs that each run batch_compute_change.py
# Modes: change, control, lemma (syn), lemma (non-syn)

echo "Submitting job: change"
sbatch run_change.sh --mode change

echo "Submitting job: control"
sbatch run_change.sh --mode control

echo "Submitting job: lemma (syn)"
sbatch run_change.sh --mode lemma --lemma_mode syn

echo "Submitting job: lemma (non-syn)"
sbatch run_change.sh --mode lemma --lemma_mode non-syn

echo "All 4 submission commands issued."
