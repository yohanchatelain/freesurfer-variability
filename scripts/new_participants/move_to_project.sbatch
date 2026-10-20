#!/bin/bash
#SBATCH --account=rrg-glatard_cpu
#SBATCH --job-name=fsvv-move
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# Move the finished participants (archive + light subset) from scratch to the project; safe to re-run.
# A participant is finished when its archive exists and its subject folders are gone (last step of run_chain.py),
# so participants still running or failed stay on scratch.
# Usage: sbatch --chdir=$SCRATCH/fsvv_newpart/slurm move_to_project.sbatch
set -euo pipefail
work=${FSVV_NEW_WORK:-$SCRATCH/fsvv_newpart}
dest=${FSVV_NEW_DEST:-/lustre09/project/6049200/ychatel/fsvv_new_participants}

move_version() {
    local v=$1 list
    list=$(mktemp -p "$work")
    for t in "$work/archives/$v"/*.tar; do
        b=$(basename "$t" .tar)
        [ -e "$work/$v/$b" ] || echo "$b"
    done > "$list"
    mkdir -p "$dest/archives/$v" "$dest/light/$v"
    # no owner/group: files take the project group from the setgid folder; rsync checks each file after transfer
    sed 's/$/.tar/' "$list" | rsync -rlpt --remove-source-files --files-from=- "$work/archives/$v/" "$dest/archives/$v/"
    (cd "$work/light/$v" && sed 's/_base$//' "$list" | while read -r p; do ls -d "${p}"_*; done) |
        rsync -rlpt --remove-source-files --files-from=- "$work/light/$v/" "$dest/light/$v/"
    find "$work/light/$v" -mindepth 1 -type d -empty -delete
    echo "$v: moved $(wc -l < "$list") participants;" \
         "project has $(ls "$dest/archives/$v" | wc -l) archives, $(ls "$dest/light/$v" | wc -l) light folders"
    rm "$list"
}

for v in FS5 FS6 FS7 FS8; do move_version "$v" & done
wait
