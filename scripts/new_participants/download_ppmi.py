"""Download the T1w images of the new participants from PPMI (LONI IDA) with livingpark_utils, as for the paper.

Runs on the login node (compute nodes have no internet). Chrome 114 + chromedriver come from the
selenium/standalone-chrome:114.0 image (apptainer); livingpark_utils/ppmi_downloader drive it as a remote WebDriver.
DICOMs are converted by image ID to $FSVV_NEW_WORK/inputs/sub-<PATNO>/ses-<VISIT>/anat/PPMI_<PATNO>_I<ImageID>.nii
(dcm2niix), the layout read by run_chain.py. Already downloaded images are skipped, so the script can be re-run after a failure.

Usage: bash -lc 'module load apptainer StdEnv/2023 dcm2niix/1.0.20230411; python download_ppmi.py [n_participants]'
Credentials: .ppmi_config of living-park (linked into the work folder, never copied).
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("FSVV_NEW_WORK", Path(os.environ["SCRATCH"]) / "fsvv_newpart"))
LISTS = Path(os.environ.get("FSVV_NEW_LISTS", HERE / "lists"))
CHROME_SIF = Path(os.environ.get("CHROME_SIF", Path(os.environ["SCRATCH"]) / "chrome/chrome114.sif"))
PPMI_CONFIG = Path("/home/ychatel/links/projects/rrg-glatard/ychatel/living-park/notebooks/.ppmi_config")
COHORT = HERE.parents[1] / "data/tables/stage_two_clinical.csv"

DL = WORK / "download"  # livingpark cwd: .cache/ (DICOM zips, study files) and inputs/
(DL / "tmp").mkdir(parents=True, exist_ok=True)
os.chdir(DL)
if not Path(".ppmi_config").exists():
    Path(".ppmi_config").symlink_to(PPMI_CONFIG)
if not (WORK / "inputs").exists():
    (WORK / "inputs").symlink_to(DL / "inputs")

# Query: one row per image, with the columns livingpark matches on (PATNO, EVENT_ID, Description).
jobs = pd.read_csv(LISTS / "new_participants.csv")
cohort = pd.read_csv(COHORT)
cohort = cohort[cohort["PATNO"].isin(jobs["PATNO"])]
query = pd.concat([
    cohort[["PATNO", "EVENT_ID", "Description", "Image ID"]],
    cohort[["PATNO", "EVENT_ID_NX", "Description_NX", "Image ID_NX"]].set_axis(
        ["PATNO", "EVENT_ID", "Description", "Image ID"], axis=1),
], ignore_index=True)
if len(sys.argv) > 1:
    query = query[query["PATNO"].isin(sorted(query["PATNO"].unique())[: int(sys.argv[1])])]


def have(r):
    return any((DL / "inputs" / f"sub-{r.PATNO}" / f"ses-{r.EVENT_ID}" / "anat").glob(f"*I{int(r._4)}.nii*"))


todo = query[[not have(r) for r in query.itertuples()]].reset_index(drop=True)
print(f"{len(query)} images requested, {len(todo)} to download", flush=True)
if todo.empty:
    sys.exit(0)

# chromedriver from the container, on a free local port
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
driver = subprocess.Popen(["apptainer", "exec", "-B", str(WORK), str(CHROME_SIF), "/usr/bin/chromedriver",
                           f"--port={port}", "--allowed-ips=127.0.0.1"],
                          stdout=open(DL / "chromedriver.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
os.environ["PPMI_SINGULARITY_SELENIUM_REMOTE"] = f"http://127.0.0.1:{port}"

import ppmi_downloader.ppmi_downloader as ppmi_dl  # noqa: E402

DICOM = DL / "dicom"  # LONI zip layout: PPMI/<PATNO>/<series>/<date>/I<ImageID>/*.dcm


def convert(r):
    """DICOM folder I<ImageID> -> inputs/sub-<PATNO>/ses-<VISIT>/anat/PPMI_<PATNO>_I<ImageID>.nii"""
    if have(r):
        return True
    image_id = int(r._4)
    src = next(DICOM.glob(f"PPMI/{r.PATNO}/*/*/I{image_id}"), None)
    if src is None:
        return False
    out = DL / "inputs" / f"sub-{r.PATNO}" / f"ses-{r.EVENT_ID}" / "anat"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dcm2niix", "-z", "n", "-b", "n", "-f", f"PPMI_{r.PATNO}_I{image_id}", "-o", str(out), str(src)],
                   check=True, stdout=subprocess.DEVNULL)
    return have(r)


try:
    patnos = sorted(todo["PATNO"].unique())
    for i in range(0, len(patnos), 10):
        batch = todo[todo["PATNO"].isin(patnos[i:i + 10])]
        if not all(convert(r) for r in batch.itertuples()):  # download only what is not already there
            try:
                dl = ppmi_dl.PPMIDownloader(remote=os.environ["PPMI_SINGULARITY_SELENIUM_REMOTE"], tempdir=str(DL / "tmp"))
                dl.download_imaging_data(batch, timeout=600, destination_dir=str(DICOM))
            except Exception as e:  # keep going; missing images are reported at the end
                print(f"batch {i // 10}: {type(e).__name__}: {e}", flush=True)
            finally:
                if "dl" in locals():
                    dl.quit()
                    del dl
        n_ok = sum(convert(r) for r in batch.itertuples())
        print(f"batch {i // 10}: {n_ok}/{len(batch)} images converted", flush=True)
finally:
    driver.terminate()

done = query[[have(r) for r in query.itertuples()]]
print(f"images ready: {len(done)}/{len(query)}")
if len(done) < len(query):
    query[[not have(r) for r in query.itertuples()]].to_csv(DL / "missing_images.csv", index=False)
    print(f"missing images listed in {DL / 'missing_images.csv'}")
