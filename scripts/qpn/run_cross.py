"""Cross-sectional FreeSurfer run for one QPN participant and one version, then archive and clean up.

QPN has one session per participant, so only the cross-sectional stage of the PPMI chain is run:
recon-all -all -qcache, with the same Boutiques descriptors and containers as the paper and the new PPMI
participants (scripts/new_participants/boutiques, run_chain.py).

Usage: python run_cross.py <FS5|FS6|FS7|FS8> <task index in lists/qpn_jobs.csv>
Env:   FSVV_QPN_DIR  (default <repo>/data/qpn): lists/qpn_jobs.csv; the light subset goes to stats/FSx/<subject>/
       FSVV_QPN_WORK (default $SCRATCH/fsvv_qpn): FSx/ (SUBJECTS_DIR), archives/FSx/<subject>.tar, logs/
       FSVV_VENV     (default <repo>/data/venv): Python environment with bosh
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAIN = ROOT / "scripts/new_participants"
QPN = Path(os.environ.get("FSVV_QPN_DIR", ROOT / "data/qpn"))
WORK = Path(os.environ.get("FSVV_QPN_WORK", Path(os.environ["SCRATCH"]) / "fsvv_qpn"))
VENV = Path(os.environ.get("FSVV_VENV", ROOT / "data/venv"))
BOSH = str(VENV / "bin/bosh")
CONTAINERS = ROOT / "data/source_files/freesurfer"
IMAGES = {
    "FS5": CONTAINERS / "ansokol-freesurfer_5.3.simg",
    "FS6": CONTAINERS / "ansokol-freesurfer_6.0.1.simg",
    "FS7": CONTAINERS / "ansokol-freesurfer_7.3.2.simg",
    "FS8": ROOT / "data/containers/freesurfer_8.2.0.sif",
}
LICENSE = ROOT / "data/FS_license/license.txt"
LIGHT_FILES = ["mri/aseg.mgz", "mri/aparc+aseg.mgz", "mri/aparc.a2009s+aseg.mgz",
               "surf/lh.thickness.fwhm10.fsaverage.mgh", "surf/rh.thickness.fwhm10.fsaverage.mgh"]

version, task = sys.argv[1], int(sys.argv[2])
job = pd.read_csv(QPN / "lists/qpn_jobs.csv", index_col="task").loc[task]
subject, t1w = job["subject"], job["t1w"]
sd = WORK / version
sd.mkdir(parents=True, exist_ok=True)
os.chdir(sd)  # SUBJECTS_DIR = `pwd` in every descriptor
# Concurrent tasks read this file: install it atomically, never rewrite it in place.
if not (sd / "license.txt").exists() or (sd / "license.txt").read_bytes() != LICENSE.read_bytes():
    tmp = sd / f".license.txt.{os.getpid()}"
    shutil.copy(LICENSE, tmp)
    os.replace(tmp, sd / "license.txt")
logdir = WORK / "logs" / version
logdir.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {version} task {task}: {msg}", flush=True)


archive = WORK / "archives" / version / f"{subject}.tar"
light = QPN / "stats" / version / subject
if archive.exists() and (light / "stats/aseg.stats").exists():
    log("already archived")
    sys.exit(0)

s = sd / subject
if not ((s / "stats/aseg.stats").exists() and not (s / "scripts/recon-all.error").exists()):
    if s.exists():
        log("removing incomplete run")
        shutil.rmtree(s)
    inputs = {"subjid": subject, "input": t1w, "qcache_flag": True}
    if version != "FS5":
        inputs["license"] = "license.txt"
    inv = logdir / f"{subject}.json"
    inv.write_text(json.dumps(inputs))
    cmd = [BOSH, "exec", "launch", "-s", "--skip-data-collection", "--no-pull",
           "--imagepath", str(IMAGES[version]), "-v", f"{WORK}:{WORK}", "-v", f"{Path(t1w).parent}:{Path(t1w).parent}"]
    if version == "FS5":  # FS5.3 reads the license from FREESURFER_HOME (as in preprocess_FS5.py)
        cmd += ["-v", f"{sd / 'license.txt'}:/usr/local/freesurfer/license.txt"]
    cmd += [str(CHAIN / "boutiques" / f"{version.lower()}_cross.json"), str(inv)]
    with open(logdir / f"{subject}.log", "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"cross: exit {rc}")
    if rc != 0 or not (s / "stats/aseg.stats").exists() or (s / "scripts/recon-all.error").exists():
        sys.exit("cross-sectional failed")

# Archive the subject folder, check it, copy the light subset, delete the folder
archive.parent.mkdir(parents=True, exist_ok=True)
tmp = archive.with_suffix(".tar.part")
with tarfile.open(tmp, "w") as tar:
    tar.add(subject)
n = sum(1 for _ in tarfile.open(tmp))
n_files = sum(len(fs) + len(ds) for _, ds, fs in os.walk(subject)) + 1
if n != n_files:
    sys.exit(f"archive check failed: {n} members, {n_files} files")
tmp.rename(archive)
for rel in ["stats", *LIGHT_FILES]:
    src, dst = s / rel, light / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)
shutil.rmtree(s)
log(f"done: {archive.stat().st_size / 1e9:.2f} GB archive, light subset in stats/{version}")
