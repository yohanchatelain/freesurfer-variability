"""Build the QPN job list and QPN_subjects.csv (participant-level: written under data/qpn, never commit).

One T1w per participant: run-1, or the next run when run-1 is not a full 3D T1w (fewer than 100 slices
along any axis or voxels larger than 1.5 mm). Participants without a usable T1w are left out.
All participants with an image are kept (PD and controls).

Usage: python make_lists.py
Env:   FSVV_QPN_DIR (default <repo>/data/qpn) holds bids/, tabular/; writes lists/qpn_jobs.csv and QPN_subjects.csv
"""
import os
import re
from pathlib import Path

import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QPN = Path(os.environ.get("FSVV_QPN_DIR", ROOT / "data/qpn"))
SEX = {"Male/Masculin": "M", "Female/Féminin": "F"}


def usable(path):
    img = nib.load(path)
    return min(img.shape[:3]) >= 100 and max(img.header.get_zooms()[:3]) <= 1.5


rows, skipped = [], 0
for anat in sorted(QPN.glob("bids/*/sub-*/ses-01/anat")):
    sub = anat.parts[-3]
    runs = sorted(anat.glob(f"{sub}_ses-01_run-*_T1w.nii.gz"),
                  key=lambda p: int(re.search(r"run-(\d+)", p.name).group(1)))
    chosen = next((p for p in runs if usable(p)), None)
    if chosen is None:
        skipped += 1
        continue
    rows.append({"subject": sub, "t1w": str(chosen.resolve()), "run": int(re.search(r"run-(\d+)", chosen.name).group(1))})
jobs = pd.DataFrame(rows)

dem = pd.read_csv(QPN / "tabular/demographics.csv").drop_duplicates("participant_id").set_index("participant_id")
ses = pd.read_csv(QPN / "tabular/mri_sessions.csv")
age = ses[ses["session"] == "ses-01"].drop_duplicates("participant_id").set_index("participant_id")["MRI_age"]
ids = jobs["subject"].str.removeprefix("sub-")
jobs["dx_group"] = dem["diagnosis_group_for_analysis"].reindex(ids).map({"PD": "PD", "control": "HC"}).values
subjects = pd.DataFrame({
    "PATNO": jobs["subject"],
    "sex": dem["sex"].reindex(ids).map(SEX).values,
    "age": age.reindex(ids).values,
    "dx_group": jobs["dx_group"],
})

(QPN / "lists").mkdir(exist_ok=True)
jobs.rename_axis("task").to_csv(QPN / "lists/qpn_jobs.csv")
subjects.to_csv(QPN / "QPN_subjects.csv", index=False)
# Aggregate summary only (no identifiers)
print(f"{len(jobs)} participants with a usable T1w, {skipped} without; "
      f"{(jobs['run'] > 1).sum()} use a run other than run-1")
print(subjects.groupby("dx_group").agg(n=("PATNO", "size"), age=("age", "mean"),
                                       women=("sex", lambda s: (s == "F").sum())).round(1))
print("missing sex/age/dx:", subjects[["sex", "age", "dx_group"]].isna().sum().to_dict())
