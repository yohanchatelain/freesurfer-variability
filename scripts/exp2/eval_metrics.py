"""Dice and volume of the FreeSurfer DKT and DK cortical labels vs the manual DKT31 labels of Mindboggle-101 (Exp 2).

Reads the native-space labels written by eval.sbatch ($FSVV_MB_WORK/eval/FSx/<subject>.{dkt,dk}.nii.gz) and
data/mindboggle/subjects/<subject>/labels.DKT31.manual.nii.gz (same voxel grid as the T1w given to recon-all).
Labels are FreeSurfer codes (1000 + region lh, 2000 + region rh); only the 62 DKT31 regions are scored (DK regions
absent from DKT31, i.e. banks of the STS, frontal and temporal poles, are not scored).
Writes $FSVV_MB_WORK/eval/dice_<FSx>.csv: one row per subject, parcellation and label.
Usage: python eval_metrics.py <FSx>
"""
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MB = HERE.parents[1] / "data/mindboggle/subjects"
WORK = Path(os.environ.get("FSVV_MB_WORK", Path(os.environ["SCRATCH"]) / "fsvv/mindboggle"))
version = sys.argv[1]

rows = []
for subject in pd.read_csv(HERE / "subjects.csv")["subject"]:
    manual_img = nib.load(MB / subject / "labels.DKT31.manual.nii.gz")
    manual = np.asarray(manual_img.dataobj).astype(np.int32)
    labels = [int(l) for l in np.unique(manual) if l >= 1000]
    voxel = float(np.prod(manual_img.header.get_zooms()[:3]))
    for parc in ("dkt", "dk"):
        f = WORK / "eval" / version / f"{subject}.{parc}.nii.gz"
        if not f.exists():
            continue
        auto_img = nib.load(f)
        if auto_img.shape != manual.shape or not np.allclose(auto_img.affine, manual_img.affine, atol=1e-3):
            print(f"{subject} {parc}: grid differs from the manual labels, skipped", flush=True)
            continue
        auto = np.asarray(auto_img.dataobj).astype(np.int32)
        for l in labels:
            a, m = auto == l, manual == l
            na, nm = int(a.sum()), int(m.sum())
            rows.append((subject, version, parc, l, 2 * int((a & m).sum()) / (na + nm) if na + nm else np.nan,
                         na * voxel, nm * voxel))
out = pd.DataFrame(rows, columns=["subject", "version", "parcellation", "label", "dice", "vol_auto", "vol_manual"])
out.to_csv(WORK / "eval" / f"dice_{version}.csv", index=False)
print(f"{version}: {out['subject'].nunique()} subjects, {len(out)} rows -> {WORK / 'eval' / f'dice_{version}.csv'}")
