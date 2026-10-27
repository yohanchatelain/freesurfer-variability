#!/bin/bash
#SBATCH --job-name=fsvv-mriqc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%A_%a.out
# MRIQC 22.0.1 (the paper's container, mriqc_patch.simg) on one participant of make_mriqc_bids.py's list per array
# task, with the paper's options (participant level, --no-sub). Compute nodes have no internet: the TemplateFlow
# files (MNI152NLin2009cAsym res-01/02) must be in $TEMPLATEFLOW_HOME first (see scripts/mriqc/README.md).
# Usage (from the repository root): sbatch --account=<account> --array=0-<n-1>%<k> scripts/mriqc/run_mriqc.sbatch
#        MRIQC_DIR (default outputs/mriqc) holds bids/ and participants.txt; results go to MRIQC_DIR/output
set -euo pipefail
module load apptainer
root=${FSVV_ROOT:-$SLURM_SUBMIT_DIR}
dir=$(realpath "${MRIQC_DIR:-$root/outputs/mriqc}")
label=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$dir/participants.txt")
work=${SCRATCH}/fsvv_mriqc_work/$label
export APPTAINERENV_TEMPLATEFLOW_HOME=${TEMPLATEFLOW_HOME:-$HOME/.cache/templateflow}
mkdir -p "$dir/output" "$work"
# the BIDS images are symlinks: bind where they point
inputs=$(dirname "$(dirname "$(dirname "$(dirname "$(readlink -f "$(ls "$dir"/bids/sub-"$label"/ses-*/anat/*_T1w.nii* | head -1)")")")")")
if [ -f "$dir/output/sub-$label/"ses-*/anat/"sub-${label}_T1w.json" ] 2>/dev/null; then
    echo "sub-$label: already done"; exit 0
fi
apptainer run -B "$dir" -B "$inputs" -B "$work" -B "$APPTAINERENV_TEMPLATEFLOW_HOME" \
    "$(realpath "$root/data/source_files/clinical/mriqc/mriqc_patch.simg")" \
    "$dir/bids" "$dir/output" participant --participant-label "$label" --no-sub \
    -w "$work" --nprocs "$SLURM_CPUS_PER_TASK" --omp-nthreads "$SLURM_CPUS_PER_TASK"
ls "$dir/output/sub-$label/"ses-*/anat/"sub-${label}_T1w.json"
rm -rf "$work"
