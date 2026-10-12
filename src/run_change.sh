#!/bin/bash
#SBATCH -A NAISS2026-3-306 -p alvis
#SBATCH -t 01:00:00
#SBATCH -C NOGPU
#SBATCH --cpus-per-task=16
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out

# Activate virtualenv if present
if [ -f venv/bin/activate ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

# Forward all arguments to the Python batch script
python3 batch_compute_change.py "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "batch_compute_change.py exited with code $EXIT_CODE"
fi

echo "Batch job finished with exit code $EXIT_CODE"
exit $EXIT_CODE
