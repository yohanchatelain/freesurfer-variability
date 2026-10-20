#!/bin/bash
#SBATCH --job-name=dice_overlap
#SBATCH --output=logs/dice_%x_%j.out
#SBATCH --error=logs/dice_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=rrg-glatard

# Usage:
# sbatch run_mri_seg_overlap.sbatch FS8 FS7

module load apptainer
sif_path=/home/vinuyans/links/projects/rrg-glatard/vinuyans/freesurfer_8.2.0.sif
WORKDIR=$(pwd)

if [ $# -ne 2 ]; then
    echo "Usage: sbatch $0 <version1> <version2>"
    exit 1
fi

version1=$1
version2=$2
tag=$(echo "${version1}${version2}" | sed 's/FS//g')

echo "Comparing $version1 vs $version2"

mkdir -p stats/dice
mkdir -p logs

mapfile -t subj < orig.txt


for sub in "${subj[@]}";
do

    echo "Processing $sub: $version1 vs $version2"

    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aparc.a2009s+aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aparc.a2009s+aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aparc2009.json


    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aparc+aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aparc+aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aparc.json


    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aseg.json


done
