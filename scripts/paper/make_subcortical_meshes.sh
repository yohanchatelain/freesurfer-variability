#!/bin/bash
# Surface meshes of fsaverage's subcortical structures (ENIGMA set: thalamus, caudate, putamen, pallidum,
# hippocampus, amygdala, accumbens, lateral ventricle; left and right) for the 3D subcortical figures.
# FreeSurfer 7.3.2 container: mri_mc (marching cubes on one aseg label) + mris_smooth. Template data only.
# Usage (compute node, from the repo root): bash scripts/paper/make_subcortical_meshes.sh [output dir]
set -euo pipefail
module load apptainer
img=$(realpath data/source_files/freesurfer/ansokol-freesurfer_7.3.2.simg)
lic=$(realpath data/FS_license/license.txt)
out=$(realpath -m "${1:-outputs/fsaverage/subcortical}")
mkdir -p "$out"
declare -A ids=([Left-Thalamus]=10 [Left-Caudate]=11 [Left-Putamen]=12 [Left-Pallidum]=13 [Left-Hippocampus]=17
  [Left-Amygdala]=18 [Left-Accumbens-area]=26 [Left-Lateral-Ventricle]=4 [Right-Thalamus]=49 [Right-Caudate]=50
  [Right-Putamen]=51 [Right-Pallidum]=52 [Right-Hippocampus]=53 [Right-Amygdala]=54 [Right-Accumbens-area]=58
  [Right-Lateral-Ventricle]=43)
for name in "${!ids[@]}"; do
  apptainer exec -B "$out" -B "$lic" --env FS_LICENSE="$lic" "$img" bash -c "
    mri_mc /usr/local/freesurfer/subjects/fsaverage/mri/aseg.mgz ${ids[$name]} $out/$name.raw.surf >/dev/null &&
    mris_smooth -n 10 -nw $out/$name.raw.surf $out/$name.surf >/dev/null"
  rm -f "$out/$name.raw.surf"
  echo "$name ok"
done
