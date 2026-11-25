#!/bin/bash
# Usage: ./submit.sh <FS5|FS6|FS7|FS8> [array spec, default all] [max running tasks, default 40] [extra sbatch options...]
# Example: ./submit.sh FS8 0   (test)   ./submit.sh FS8 1-269 40 --dependency=afterok:<test job id>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
version=$1
qpn=${FSVV_QPN_DIR:-$here/../../data/qpn}
n=$(( $(wc -l < "$qpn/lists/qpn_jobs.csv") - 2 ))
array=${2:-0-$n}
throttle=${3:-40}
shift $(( $# < 3 ? $# : 3 ))
work=${FSVV_QPN_WORK:-$SCRATCH/fsvv_qpn}
mkdir -p "$work/slurm"
# FS8 (SynthStrip, SynthSeg) needs ~24G per recon-all, as in the original FS8 jobs
mem=6G; [ "$version" = FS8 ] && mem=24G
mem=${FSVV_MEM:-$mem}
time=${FSVV_TIME:-48:00:00}
sbatch --parsable --account="${FSVV_ACCOUNT:?set FSVV_ACCOUNT to your Slurm account}" --array="$array%$throttle" \
       --mem="$mem" --time="$time" --job-name="fsvv-qpn-$version" --chdir="$work/slurm" \
       --export=ALL,FS_VERSION="$version",QPN_SCRIPTS="$here" "$@" "$here/run_cross.sbatch"
