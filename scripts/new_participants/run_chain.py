"""Full FreeSurfer chain for one participant and one version, then archive and clean up.

cross (2 visits, in parallel) -> base -> long (2 visits) -> long qcache (2 visits) -> long_mris_slopes (lh, rh)
-> tar of the 5 subject folders -> extract the light subset used by the analyses -> delete the folders.

Same Boutiques descriptors and options as the paper (FS5-7: Zenodo records of preprocess_FS{5,6,7}*.py;
FS8: vinuyans/boutiques with SUBJECTS_DIR=`pwd` instead of `pwd`/data/FS8).
A stage is skipped when its outputs exist, so a failed task can be resubmitted as is.

Usage: python run_chain.py <FS5|FS6|FS7|FS8> <task index in lists/new_participants.csv>
Env:   FSVV_NEW_LISTS (default lists/) holds new_participants.csv and qdec/
       FSVV_NEW_WORK (default $SCRATCH/fsvv_newpart) holds inputs/, FSx/ (SUBJECTS_DIR), archives/, light/
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = Path(os.environ.get("FSVV_NEW_WORK", Path(os.environ["SCRATCH"]) / "fsvv_newpart"))
LISTS = Path(os.environ.get("FSVV_NEW_LISTS", HERE / "lists"))
BOSH = "/lustre09/project/6049200/vinuyans/.venv/bin/bosh"
VINUYANS = Path("/lustre09/project/6049200/vinuyans")
IMAGES = {
    "FS5": VINUYANS / "source_files/freesurfer/ansokol-freesurfer_5.3.simg",
    "FS6": VINUYANS / "source_files/freesurfer/ansokol-freesurfer_6.0.1.simg",
    "FS7": VINUYANS / "source_files/freesurfer/ansokol-freesurfer_7.3.2.simg",
    "FS8": VINUYANS / "freesurfer_8.2.0.sif",
}
LICENSE = ROOT / "data/FS_license/license.txt"

version, task = sys.argv[1], int(sys.argv[2])
job = pd.read_csv(LISTS / "new_participants.csv", index_col="task").loc[task]
patno, tp1, tp2, base = job["PATNO"], job["tp1"], job["tp2"], job["base"]
longs = [f"{tp}.long.{base}" for tp in (tp1, tp2)]
sd = WORK / version
sd.mkdir(parents=True, exist_ok=True)
os.chdir(sd)  # SUBJECTS_DIR = `pwd` in every descriptor
# Concurrent tasks read this file: never rewrite it in place (a truncated read fails with
# "error parsing license file"), only install it atomically when missing or different.
if not (sd / "license.txt").exists() or (sd / "license.txt").read_bytes() != LICENSE.read_bytes():
    tmp = sd / f".license.txt.{os.getpid()}"
    shutil.copy(LICENSE, tmp)
    os.replace(tmp, sd / "license.txt")
logdir = WORK / "logs" / version / base
logdir.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {version} {base}: {msg}", flush=True)


def bosh(stage, name, **inputs):
    """Run one descriptor through bosh/apptainer; return True on success."""
    if version != "FS5":
        inputs["license"] = "license.txt"
    inv = logdir / f"{stage}_{name}.json"
    inv.write_text(json.dumps(inputs))
    cmd = [BOSH, "exec", "launch", "-s", "--skip-data-collection", "--no-pull",
           "--imagepath", str(IMAGES[version]), "-v", f"{WORK}:{WORK}"]
    if version == "FS5":  # FS5.3 reads the license from FREESURFER_HOME (as in preprocess_FS5.py)
        cmd += ["-v", f"{sd / 'license.txt'}:/usr/local/freesurfer/license.txt"]
    cmd += [str(HERE / "boutiques" / f"{version.lower()}_{stage}.json"), str(inv)]
    with open(logdir / f"{stage}_{name}.log", "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"{stage} {name}: exit {rc}")
    return rc == 0


def recon_ok(subject):
    s = sd / subject
    return (s / "stats/aseg.stats").exists() and not (s / "scripts/recon-all.error").exists()


def parallel(fn, items):
    with ThreadPoolExecutor(len(items)) as ex:
        return all(ex.map(fn, items))


def fresh(subject):
    if (sd / subject).exists():
        log(f"removing incomplete {subject}")
        shutil.rmtree(sd / subject)


def image(visit, image_id):
    found = sorted((WORK / "inputs" / f"sub-{patno}" / f"ses-{visit}" / "anat").glob(f"*I{image_id}*.nii*"))
    if not found:
        sys.exit(f"missing T1w image I{image_id} for {base} ({visit})")
    return str(found[0])


archive = WORK / "archives" / version / f"{base}.tar"
if archive.exists():
    log(f"already archived: {archive}")
    sys.exit(0)

# 1. Cross-sectional (with -qcache, as for the paper)
todo = [(tp, image(v, i)) for tp, v, i in [(tp1, job["visit1"], job["image_id1"]), (tp2, job["visit2"], job["image_id2"])]
        if not recon_ok(tp)]
for tp, _ in todo:
    fresh(tp)
if todo and not parallel(lambda a: bosh("cross", a[0], subjid=a[0], input=a[1], qcache_flag=True), todo):
    sys.exit("cross-sectional failed")
if not (recon_ok(tp1) and recon_ok(tp2)):
    sys.exit("cross-sectional outputs incomplete")

# 2. Base template
if not recon_ok(base):
    fresh(base)
    if not bosh("base", base, tp1=tp1, tp2=tp2, outputdir=base) or not recon_ok(base):
        sys.exit("base failed")

# 3. Longitudinal runs
todo = [(tp, lg) for tp, lg in zip((tp1, tp2), longs) if not recon_ok(lg)]
for _, lg in todo:
    fresh(lg)
if todo and not parallel(lambda a: bosh("long", a[1], tp=a[0], base=base), todo):
    sys.exit("long failed")

# 4. qcache of the longitudinal runs
todo = [(tp, lg) for tp, lg in zip((tp1, tp2), longs)
        if not (sd / lg / "surf/rh.thickness.fwhm10.fsaverage.mgh").exists()]
if todo and not parallel(lambda a: bosh("long_qcache", a[1], tp=a[0], base=base), todo):
    sys.exit("long qcache failed")

# 5. Thickness slopes (per participant; cohort stacks are rebuilt later with mri_concat/mris_preproc)
qdec = sd / f"{base}.qdec.dat"
shutil.copy(LISTS / "qdec" / f"{patno}.dat", qdec)
(sd / base / "slopes").mkdir(exist_ok=True)
todo = [h for h in ("lh", "rh") if not (sd / base / f"surf/{h}.long.thickness-rate.fwhm10.fsaverage.mgh").exists()]
stacks = {k: f"{base}/slopes/{{h}}.long.thickness-{k}.stack.mgh" for k in ("avg", "rate", "pc1fit", "pc1", "spc")}
if todo and not parallel(lambda h: bosh("slopes", h, qdec=qdec.name, meas="thickness", hemi=h, time="years",
                                        **{f"stack_{k}": v.format(h=h) for k, v in stacks.items()}), todo):
    sys.exit("long_mris_slopes failed")
qdec.unlink()

# 6. Archive, extract the light subset, delete the subject folders
folders = [tp1, tp2, base, *longs]
archive.parent.mkdir(parents=True, exist_ok=True)
tmp = archive.with_suffix(".tar.part")
with tarfile.open(tmp, "w") as tar:
    for f in folders:
        tar.add(f)
n = sum(1 for _ in tarfile.open(tmp))
n_files = sum(len(fs) + len(ds) for f in folders for _, ds, fs in os.walk(f)) + len(folders)
if n != n_files:
    sys.exit(f"archive check failed: {n} members, {n_files} files")
tmp.rename(archive)

light = WORK / "light" / version
LIGHT_FILES = {
    tp1: ["mri/aseg.mgz", "mri/aparc+aseg.mgz", "mri/aparc.a2009s+aseg.mgz",
          "surf/lh.thickness.fwhm10.fsaverage.mgh", "surf/rh.thickness.fwhm10.fsaverage.mgh"],
    base: [f"surf/{h}.long.thickness-{k}.fwhm10.fsaverage.mgh" for h in ("lh", "rh") for k in stacks],
}
LIGHT_FILES[tp2] = LIGHT_FILES[tp1]
for f in folders:
    for rel in ["stats", *LIGHT_FILES.get(f, [])]:
        src, dst = sd / f / rel, light / f / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, dst)
for f in folders:
    shutil.rmtree(sd / f)
log(f"done: {archive} ({archive.stat().st_size / 1e9:.1f} GB, {n} members); light subset in {light}")
