#!/bin/bash
#SBATCH --job-name=fsvv-mb
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
# One array task = one Mindboggle-101 subject (row of subjects.csv), one FreeSurfer version. Submit with submit.sh.
set -euo pipefail
module load apptainer
export FSVV_MB_WORK=${FSVV_MB_WORK:-$SCRATCH/fsvv/mindboggle}
"${FSVV_VENV:-$EXP2_DIR/../../data/venv}/bin/python" "$EXP2_DIR/run_recon.py" "$FS_VERSION" "$SLURM_ARRAY_TASK_ID"
