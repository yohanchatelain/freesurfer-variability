#!/bin/bash
#SBATCH --job-name=fsvv-dice
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%A_%a.out
# Segmentation Dice between FreeSurfer versions (mri_seg_overlap of FreeSurfer 8, as for the paper's Dice), for
# the lines of the list written by make_dice_list.py (segmentation 1, segmentation 2, output JSON).
# Array task i of n handles lines i, i+n, i+2n, ... in one container; existing outputs are skipped.
# Usage (from the repository root): sbatch --account=<account> --array=0-<n-1> scripts/dice/run_dice.sbatch
#        DICE_LIST (default outputs/dice/todo.tsv)
set -euo pipefail
module load apptainer
root=${FSVV_ROOT:-$SLURM_SUBMIT_DIR}
list=$(realpath "${DICE_LIST:-$root/outputs/dice/todo.tsv}")
n=${SLURM_ARRAY_TASK_COUNT:-1}
i=${SLURM_ARRAY_TASK_ID:-0}
license=$(realpath "$root/data/FS_license/license.txt")
binds=(-B "$(dirname "$list")" -B "$license:$license" -B "$(realpath "$root/data/new_participants")")
for v in FS5 FS6 FS7 FS8; do binds+=(-B "$(realpath "$root/data/$v")"); done

awk -F'\t' -v n="$n" -v i="$i" '(NR - 1) % n == i' "$list" |
    apptainer exec --cleanenv "${binds[@]}" --env FS_LICENSE="$license" \
        "$(realpath "$root/data/containers/freesurfer_8.2.0.sif")" bash -c '
    status=0
    while IFS=$'"'"'\t'"'"' read -r a b out; do
        [ -s "$out" ] && continue
        mri_seg_overlap "$a" "$b" -o "$out" > /dev/null || { echo "failed: $out"; status=1; }
    done
    exit $status'
echo "task $i/$n done"
