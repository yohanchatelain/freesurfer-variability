#!/bin/bash
# Usage: ./submit.sh <FS5|FS6|FS7|FS8> [array spec, default all participants] [max running tasks, default 40]
# Example test: ./submit.sh FS8 0        Full run: for v in FS5 FS6 FS7 FS8; do ./submit.sh $v; done
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
version=$1
lists=${FSVV_NEW_LISTS:-$here/lists}
n=$(( $(wc -l < "$lists/new_participants.csv") - 2 ))
array=${2:-0-$n}
throttle=${3:-40}
work=${FSVV_NEW_WORK:-$SCRATCH/fsvv_newpart}
mkdir -p "$work/slurm"
# FS8 (SynthStrip, SynthSeg) needs ~24G per recon-all, as in vinuyans' FS8 jobs; the chain runs 2 visits at once
mem=10G; [ "$version" = FS8 ] && mem=48G
mem=${FSVV_MEM:-$mem}  # override for OOM resubmissions, e.g. FSVV_MEM=16G
time=${FSVV_TIME:-48:00:00}  # override for timeouts, e.g. FSVV_TIME=7-00:00:00
sbatch --array="$array%$throttle" --mem="$mem" --time="$time" --job-name="fsvv-new-$version" --chdir="$work/slurm" \
       --export=ALL,FS_VERSION="$version",CHAIN_DIR="$here" "$here/run_chain.sbatch"
