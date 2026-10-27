#!/bin/bash
# Usage: ./submit.sh <FS5|FS6|FS7|FS8> [array spec, default all subjects] [max running tasks, default 50]
# Example test: ./submit.sh FS7 0,50        Full run: for v in FS5 FS6 FS7 FS8; do ./submit.sh $v; done
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
version=$1
n=$(( $(wc -l < "$here/subjects.csv") - 2 ))
array=${2:-0-$n}
throttle=${3:-50}
work=${FSVV_MB_WORK:-$SCRATCH/fsvv/mindboggle}
mkdir -p "$work/slurm"
mem=8G; [ "$version" = FS8 ] && mem=24G  # FS8 SynthStrip/SynthSeg need ~24G
mem=${FSVV_MEM:-$mem}
time=${FSVV_TIME:-24:00:00}
sbatch --account="${FSVV_ACCOUNT:?set FSVV_ACCOUNT to your Slurm account}" --array="$array%$throttle" --mem="$mem" --time="$time" \
       --job-name="fsvv-mb-$version" --chdir="$work/slurm" --export=ALL,FS_VERSION="$version",EXP2_DIR="$here" "$here/run_recon.sbatch"
