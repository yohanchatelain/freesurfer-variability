#!/bin/bash
#SBATCH --job-name=fsvv-qpn
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
# One array task = one QPN participant (row of lists/qpn_jobs.csv), cross-sectional run for one FreeSurfer version.
# Submit with submit.sh.
set -euo pipefail
module load apptainer
"${FSVV_VENV:-$QPN_SCRIPTS/../../data/venv}/bin/python" "$QPN_SCRIPTS/run_cross.py" "$FS_VERSION" "$SLURM_ARRAY_TASK_ID"
