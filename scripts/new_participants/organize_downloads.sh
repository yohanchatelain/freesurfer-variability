#!/bin/bash
# Convert the PPMI images downloaded from LONI IDA (DICOM zip(s) of the image IDs in lists/loni_image_ids.txt)
# to NIfTI in $FSVV_NEW_WORK/inputs/sub-<PATNO>/ses-<VISIT>/anat/PPMI_<PATNO>_I<ImageID>.nii (layout of data/inputs).
# Usage (compute node or salloc, it runs dcm2niix ~600 times): ./organize_downloads.sh <download.zip>...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${FSVV_NEW_WORK:-$SCRATCH/fsvv_newpart}
module load StdEnv/2023 dcm2niix/1.0.20230411
raw=$work/downloads_unzipped
mkdir -p "$raw"
for z in "$@"; do unzip -q -n "$z" -d "$raw"; done
# LONI layout: PPMI/<PATNO>/<series>/<date>/I<ImageID>/*.dcm
tail -n +2 "$here/lists/new_participants.csv" | while IFS=, read -r task patno dx v1 v2 i1 i2 rest; do
  for pair in "$v1:$i1" "$v2:$i2"; do
    visit=${pair%%:*}; id=${pair##*:}
    out=$work/inputs/sub-$patno/ses-$visit/anat
    ls "$out"/*I"$id".nii* >/dev/null 2>&1 && continue
    src=$(ls -d "$raw"/PPMI/"$patno"/*/*/I"$id" 2>/dev/null | head -1)
    if [ -z "$src" ]; then echo "MISSING sub-$patno ses-$visit I$id"; continue; fi
    mkdir -p "$out"
    dcm2niix -z n -b n -f "PPMI_${patno}_I${id}" -o "$out" "$src" > /dev/null
  done
done
echo "images ready: $(find "$work/inputs" -name '*.nii*' | wc -l) / $(( 2 * ($(wc -l < "$here/lists/new_participants.csv") - 1) ))"
