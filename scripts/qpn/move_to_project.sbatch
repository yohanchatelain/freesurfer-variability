#!/bin/bash
#SBATCH --job-name=fsvv-qpn-move
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# Move the finished QPN FreeSurfer archives from scratch to the project space; safe to re-run.
# The light subset (stats, segmentations) is already written to the project by run_cross.py.
# An archive is moved only when the participant's light subset exists (run_cross.py writes both at the end).
# Logs (recon-all, Slurm) are copied, not moved.
# Usage (from the repo root): sbatch --account=<account> --chdir=$SCRATCH/fsvv_qpn/slurm move_to_project.sbatch
#        FSVV_QPN_DIR (default data/qpn of the submit folder): project folder; archives go to $FSVV_QPN_DIR/fs_archives/FSx
set -euo pipefail
shopt -s nullglob
work=${FSVV_QPN_WORK:-$SCRATCH/fsvv_qpn}
qpn=$(readlink -f "${FSVV_QPN_DIR:-$SLURM_SUBMIT_DIR/data/qpn}")
[ -d "$qpn/stats" ] || { echo "no stats/ in $qpn: set FSVV_QPN_DIR" >&2; exit 1; }
dest=$qpn/fs_archives

move_version() {
    local v=$1 list
    list=$(mktemp -p "$work")
    for t in "$work/archives/$v"/*.tar; do
        b=$(basename "$t" .tar)
        [ -f "$qpn/stats/$v/$b/stats/aseg.stats" ] && echo "$b.tar"
    done > "$list"
    mkdir -p "$dest/$v"
    # no owner/group: files take the project group from the setgid folder; rsync checks each file after transfer
    rsync -rlpt --remove-source-files --files-from="$list" "$work/archives/$v/" "$dest/$v/"
    echo "$v: moved $(wc -l < "$list") archives; project has $(ls "$dest/$v" | wc -l)," \
         "scratch has $(ls "$work/archives/$v" | wc -l) left"
    rm "$list"
}

for v in FS5 FS6 FS7 FS8; do move_version "$v" & done
wait
mkdir -p "$qpn/logs"
rsync -rlpt "$work/logs/" "$qpn/logs/recon-all/"
rsync -rlpt "$work/slurm/" "$qpn/logs/slurm/"
echo "logs copied to $qpn/logs"
