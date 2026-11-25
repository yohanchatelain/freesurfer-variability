#!/bin/bash
#SBATCH --job-name=fsvv-glm
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
# Vertex-wise GLMs of the paper (cortical thickness, fsaverage, fwhm10) for one FreeSurfer version per array task
# (0-3 = FS5-FS8), on the inputs of make_glm_inputs.py. Same commands as the paper (logs of the FS7/FS8 runs):
#   mris_preproc --fsgd F --cache-in C --target fsaverage --hemi H --out STACK
#   mri_glmfit --fsgd F --glmdir DIR --y STACK --surf fsaverage H --C CON
#   mri_glmfit-sim --glmdir DIR --cache 1.3 abs --cwp 0.05 --2spaces
# A step is skipped when its output exists, so a failed task can be resubmitted as is.
# Usage (from the repository root): sbatch --account=<account> scripts/glm/run_glm.sbatch
#        GLM_DIR (default outputs/glm) holds fsgd/, con/, subjects/FSx; results go to GLM_DIR/FSx/
set -euo pipefail
module load apptainer
root=${FSVV_ROOT:-$SLURM_SUBMIT_DIR}
glm=$(realpath "${GLM_DIR:-$root/outputs/glm}")
versions=(FS5 FS6 FS7 FS8)
v=${versions[$SLURM_ARRAY_TASK_ID]}
declare -A image=(
    [FS5]=data/source_files/freesurfer/ansokol-freesurfer_5.3.simg
    [FS6]=data/source_files/freesurfer/ansokol-freesurfer_6.0.1.simg
    [FS7]=data/source_files/freesurfer/ansokol-freesurfer_7.3.2.simg
    [FS8]=data/containers/freesurfer_8.2.0.sif
)
license=$(realpath "$root/data/FS_license/license.txt")
# the subject links point into the paper's runs and the new participants' runs: bind their real locations
binds=(-B "$glm" -B "$(realpath "$root/data/$v")" -B "$(realpath "$root/data/new_participants")")
if [ "$v" = FS5 ]; then  # FS5.3 reads the license from FREESURFER_HOME
    binds+=(-B "$license:/usr/local/freesurfer/license.txt")
else
    binds+=(-B "$license:$license")
fi
run() {
    apptainer exec --cleanenv "${binds[@]}" --env SUBJECTS_DIR="$glm/subjects/$v" --env FS_LICENSE="$license" \
        --pwd "$glm/$v" "$(realpath "$root/${image[$v]}")" "$@"
}
mkdir -p "$glm/$v"

#        name                 fsgd                                    cache-in                              contrast                          stack (H = hemi)
analyses="group_base_HC_PDnonMCI fsgd_cort_group_HC_PDnonMCI_baseline.fsgd thickness.fwhm10.fsaverage            con_group_HC_PDnonMCI_base.mtx   stack.H.group.HC_PDnonMCI.thickness.10.baseline.mgh
group_long_HC_PDnonMCI fsgd_cort_group_HC_PDnonMCI_long.fsgd     long.thickness-rate.fwhm10.fsaverage  con_group_HC_PDnonMCI_long.mtx   stack.H.group.HC_PDnonMCI.thickness.rate.10.long.mgh
corr_base_PDnonMCI     fsgd_corr_PDnonMCI_group.fsgd             thickness.fwhm10.fsaverage            con_corr_UPDRS_base.mtx          stack.H.corr.PDnonMCI.thickness.10.baseline.mgh
corr_long_PDnonMCI     fsgd_corr_PDnonMCI_group_long.fsgd        long.thickness-rate.fwhm10.fsaverage  con_corr_UPDRS_long.mtx          stack.H.corr.PDnonMCI.thickness.rate.10.mgh"

while read -r name fsgd cache con stack; do
    for hemi in lh rh; do
        y=${stack/H/$hemi}
        dir=results_${name}_$hemi
        [ -f "$glm/$v/$y" ] ||
            run mris_preproc --fsgd "$glm/fsgd/$fsgd" --cache-in "$cache" --target fsaverage --hemi "$hemi" --out "$y"
        [ -f "$glm/$v/$dir/${con%.mtx}/sig.mgh" ] ||
            run mri_glmfit --fsgd "$glm/fsgd/$fsgd" --glmdir "$dir" --y "$y" --surf fsaverage "$hemi" --C "$glm/con/$con"
        [ -f "$glm/$v/$dir/${con%.mtx}/cache.th13.abs.sig.cluster.summary" ] ||
            run mri_glmfit-sim --glmdir "$dir" --cache 1.3 abs --cwp 0.05 --2spaces
        echo "$v $dir: done"
    done
done <<< "$analyses"
