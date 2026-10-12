#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_prolific_all.sh /path/to/jsonl_folder [log_dir]
# Example: ./run_prolific_all.sh ../data/hum_eval/ADJ logs/prolific

DIR="${1:-./data/hum_eval/ADJ}"
LOG_DIR="${2:-./logs/prolific}"

mkdir -p "$LOG_DIR"
shopt -s nullglob

for f in "$DIR"/*.jsonl; do
  base=$(basename "$f" .jsonl)
  echo "==== Running: $f ====" | tee "$LOG_DIR/${base}.txt"
  python prolific_publish.py "$f" --publish-prolific 2>&1 | tee -a "$LOG_DIR/${base}.txt"
  echo "==== Finished: $f ====" | tee -a "$LOG_DIR/${base}.txt"
done

echo "All done. Logs saved to $LOG_DIR"
