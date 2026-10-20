#!/bin/bash
#SBATCH --account=rrg-glatard_cpu
#SBATCH --job-name=fsvv-new
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
# One array task = one participant (row of lists/new_participants.csv), full chain for one FreeSurfer version.
# Submit with submit.sh (sets the version, the log folder and the throttle).
set -euo pipefail
module load apptainer
export FSVV_NEW_WORK=${FSVV_NEW_WORK:-$SCRATCH/fsvv_newpart}
/lustre09/project/6049200/vinuyans/.venv/bin/python "$CHAIN_DIR/run_chain.py" "$FS_VERSION" "$SLURM_ARRAY_TASK_ID"
