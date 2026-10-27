#!/bin/bash
#SBATCH --job-name=fsvv-mb-eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
# Native-space DKT and DK labels of every finished subject of one version, then Dice/volume vs the manual DKT31 labels.
# Usage: sbatch --account=<account> --export=ALL,FS_VERSION=FS7,EXP2_DIR=$PWD eval.sbatch
set -euo pipefail
module load apptainer
work=${FSVV_MB_WORK:-$SCRATCH/fsvv/mindboggle}
root=$EXP2_DIR/../..
case $FS_VERSION in
  FS5) img=$root/data/source_files/freesurfer/ansokol-freesurfer_5.3.simg; dkt=aparc.DKTatlas40+aseg ;;
  FS6) img=$root/data/source_files/freesurfer/ansokol-freesurfer_6.0.1.simg; dkt=aparc.DKTatlas+aseg ;;
  FS7) img=$root/data/source_files/freesurfer/ansokol-freesurfer_7.3.2.simg; dkt=aparc.DKTatlas+aseg ;;
  FS8) img=$root/data/containers/freesurfer_8.2.0.sif; dkt=aparc.DKTatlas+aseg ;;
esac
sd=$work/$FS_VERSION
out=$work/eval/$FS_VERSION
mkdir -p "$out"
for s in $(tail -n +2 "$EXP2_DIR/subjects.csv" | cut -d, -f2); do
  m=$sd/$s/mri
  [ -f "$sd/$s/stats/aseg.stats" ] && [ ! -f "$sd/$s/scripts/recon-all.error" ] || { echo "skip $s (not finished)"; continue; }
  # FS5 makes the DKTatlas40 annotation but not its volume: build it as recon-all does for aparc+aseg
  if [ "$FS_VERSION" = FS5 ] && [ ! -f "$m/$dkt.mgz" ]; then
    apptainer exec -B "$work" -B "$sd/license.txt:/usr/local/freesurfer/license.txt" --env SUBJECTS_DIR="$sd" "$img" \
      mri_aparc2aseg --s "$s" --annot aparc.DKTatlas40 --volmask > "$out/$s.aparc2aseg.log" 2>&1
  fi
  for seg in "dkt:$dkt" "dk:aparc+aseg"; do
    name=${seg%%:*}; file=$m/${seg#*:}.mgz
    [ -f "$out/$s.$name.nii.gz" ] && continue
    # FreeSurfer conformed space -> native space of the input T1w (rawavg), nearest neighbour
    apptainer exec -B "$work" --env FS_LICENSE="$sd/license.txt" "$img" \
      mri_label2vol --seg "$file" --temp "$m/rawavg.mgz" --o "$out/$s.$name.nii.gz" --regheader "$file" > /dev/null
  done
done
"${FSVV_VENV:-$root/data/venv}/bin/python" "$EXP2_DIR/eval_metrics.py" "$FS_VERSION"
