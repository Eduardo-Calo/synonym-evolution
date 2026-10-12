#!/bin/bash
# Submit a separate SLURM job for each JSONL file in a folder.
# Usage: ./run_llm_all.sh <folder_path>

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <folder_path>"
    echo "Example: $0 ../data/llm_eval/ADJ_thr0.85_1.1"
    exit 1
fi

FOLDER="$1"

if [[ ! -d "$FOLDER" ]]; then
    echo "Error: folder not found: $FOLDER"
    exit 1
fi

JSONL_COUNT=0
SUBMITTED_COUNT=0

for jsonl_file in "$FOLDER"/*.jsonl; do
    if [[ ! -f "$jsonl_file" ]]; then
        continue
    fi

    JSONL_COUNT=$((JSONL_COUNT + 1))
    echo "Submitting job for: $jsonl_file"
    sbatch run_llm.sh "$jsonl_file"
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
done

if [[ $JSONL_COUNT -eq 0 ]]; then
    echo "No JSONL files found in $FOLDER"
    exit 1
fi

echo "Submitted $SUBMITTED_COUNT jobs for $JSONL_COUNT JSONL files"
