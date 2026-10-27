"""Cross-sectional recon-all of one Mindboggle-101 subject with one FreeSurfer version (Exp 2).

Same containers and Boutiques descriptors as the PPMI runs (scripts/new_participants/boutiques/fsN_cross.json),
with recon-all -all and no -qcache (no vertex-wise analysis on Mindboggle).
A finished subject (stats/aseg.stats, no recon-all.error) is skipped, an unfinished one is restarted.

Usage: python run_recon.py <FS5|FS6|FS7|FS8> <task index in subjects.csv>
Env:   FSVV_MB_WORK (default $SCRATCH/fsvv/mindboggle) holds inputs/, FSx/ (SUBJECTS_DIR), logs/
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAIN = ROOT / "scripts/new_participants"
WORK = Path(os.environ.get("FSVV_MB_WORK", Path(os.environ["SCRATCH"]) / "fsvv/mindboggle"))
VENV = Path(os.environ.get("FSVV_VENV", ROOT / "data/venv"))
CONTAINERS = ROOT / "data/source_files/freesurfer"
IMAGES = {
    "FS5": CONTAINERS / "ansokol-freesurfer_5.3.simg",
    "FS6": CONTAINERS / "ansokol-freesurfer_6.0.1.simg",
    "FS7": CONTAINERS / "ansokol-freesurfer_7.3.2.simg",
    "FS8": ROOT / "data/containers/freesurfer_8.2.0.sif",
}
LICENSE = ROOT / "data/FS_license/license.txt"

version, task = sys.argv[1], int(sys.argv[2])
subject = pd.read_csv(HERE / "subjects.csv", index_col="task").loc[task, "subject"]
sd = WORK / version
sd.mkdir(parents=True, exist_ok=True)
os.chdir(sd)  # SUBJECTS_DIR = `pwd` in every descriptor
# Concurrent tasks read this file: install it atomically, only when missing or different
if not (sd / "license.txt").exists() or (sd / "license.txt").read_bytes() != LICENSE.read_bytes():
    tmp = sd / f".license.txt.{os.getpid()}"
    shutil.copy(LICENSE, tmp)
    os.replace(tmp, sd / "license.txt")

s = sd / subject
if (s / "stats/aseg.stats").exists() and not (s / "scripts/recon-all.error").exists():
    sys.exit(print(f"{version} {subject}: already done") or 0)
if s.exists():
    print(f"{version} {subject}: removing incomplete run", flush=True)
    shutil.rmtree(s)

inputs = {"subjid": subject, "input": str(WORK / "inputs" / f"{subject}.nii.gz"), "qcache_flag": False}
if version != "FS5":
    inputs["license"] = "license.txt"
logdir = WORK / "logs" / version
logdir.mkdir(parents=True, exist_ok=True)
inv = logdir / f"{subject}.json"
inv.write_text(json.dumps(inputs))
cmd = [str(VENV / "bin/bosh"), "exec", "launch", "-s", "--skip-data-collection", "--no-pull",
       "--imagepath", str(IMAGES[version]), "-v", f"{WORK}:{WORK}"]
if version == "FS5":  # FS5.3 reads the license from FREESURFER_HOME
    cmd += ["-v", f"{sd / 'license.txt'}:/usr/local/freesurfer/license.txt"]
cmd += [str(CHAIN / "boutiques" / f"{version.lower()}_cross.json"), str(inv)]
with open(logdir / f"{subject}.log", "w") as out:
    rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
ok = rc == 0 and (s / "stats/aseg.stats").exists() and not (s / "scripts/recon-all.error").exists()
print(f"{version} {subject}: exit {rc}, {'ok' if ok else 'FAILED'}", flush=True)
sys.exit(0 if ok else 1)
