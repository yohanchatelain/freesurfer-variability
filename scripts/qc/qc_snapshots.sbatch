#!/bin/bash
#SBATCH --job-name=fsvv-qc
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%A_%a.out
# One array task = one participant of outputs/qc/images.csv (python scripts/qc/qc_snapshots.py list first).
# Usage (from the repository root): sbatch --account=<account> --array=0-<n-1>%40 --chdir=outputs/qc/slurm \
#        --export=ALL,FSVV_ROOT=$PWD scripts/qc/qc_snapshots.sbatch
set -euo pipefail
"${FSVV_PYTHON:-python}" "$FSVV_ROOT/scripts/qc/qc_snapshots.py" "$SLURM_ARRAY_TASK_ID"
