"""Visual QC snapshots of the images added for the revision (participants not in the paper's cohort).

`list`: writes outputs/qc/images.csv (one row per participant with images analysed in 02) and an empty
outputs/qc/ratings.csv to fill in. `<task>`: for the participant of row <task>, extracts T1.mgz and aparc+aseg.mgz
of each cross-sectional visit from the four version archives (data/new_participants/archives/FSx/<base>.tar) into a
temporary folder and writes outputs/qc/png/<image>.png: one row per version, five slices (3 coronal, axial,
sagittal) of T1.mgz with aparc+aseg outlines. Participant-level images: outputs/ only, never published.

Usage: python scripts/qc/qc_snapshots.py list
       python scripts/qc/qc_snapshots.py <task>      (qc_snapshots.sbatch runs one task per participant)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

QC = fsvv.OUT / "qc"
ARCHIVES = fsvv.DATA / "new_participants" / "archives"
FILES = ["mri/T1.mgz", "mri/aparc+aseg.mgz"]


def make_list():
    images = pd.read_csv(fsvv.OUT / "aseg_table4.csv", usecols=["PATNO_id"])["PATNO_id"]
    patno = images.str.extract(r"sub-(\d+)_ses", expand=False).astype(int)
    new = images[~patno.isin(fsvv.paper_participants())]
    t = pd.DataFrame({"image": new, "base": new.str.replace(r"_ses-.*", "_base", regex=True)})
    t = t.groupby("base")["image"].apply(" ".join).reset_index()
    t.to_csv(QC / "images.csv", index_label="task")
    ratings = QC / "ratings.csv"
    if not ratings.exists():  # never overwrite ratings already entered
        pd.DataFrame({"image": sorted(new), "rating": "", "version": "", "comment": ""}).to_csv(ratings, index=False)
    print(f"{len(new)} images of {len(t)} participants -> {QC / 'images.csv'}; tasks 0-{len(t) - 1}")


def lut():
    """FreeSurferColorLUT.txt (copied from the FS7 container) -> {label: rgb}."""
    colors = {}
    for line in open(QC / "FreeSurferColorLUT.txt"):
        f = line.split()
        if len(f) >= 5 and f[0].isdigit():
            colors[int(f[0])] = np.array([int(c) for c in f[2:5]]) / 255
    return colors


def outlines(seg, colors):
    """RGBA image of the label boundaries of a 2D label slice."""
    edge = np.zeros(seg.shape, bool)
    edge[:-1] |= seg[:-1] != seg[1:]
    edge[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    edge &= seg > 0
    rgba = np.zeros(seg.shape + (4,))
    for label in np.unique(seg[edge]):
        m = edge & (seg == label)
        rgba[m, :3] = colors.get(int(label), (1, 1, 0))
        rgba[m, 3] = 1
    return rgba


def slices(vol):
    """Five slices of a 256^3 conformed volume (LIA): 3 coronal, 1 axial, 1 sagittal, displayed upright."""
    c = np.array(vol.shape) // 2
    return ([vol[:, :, k].T for k in (c[2] - 30, c[2], c[2] + 30)]
            + [np.flipud(vol[:, c[1] - 10, :].T), vol[c[0] + 30, :, :]])


def snapshot(task):
    row = pd.read_csv(QC / "images.csv", index_col="task").loc[task]
    images = row["image"].split()
    colors = lut()
    out = QC / "png"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR")) as tmp:
        for v in fsvv.VERSIONS:
            members = [f"{im}/{f}" for im in images for f in FILES]
            subprocess.run(["tar", "-xf", str(ARCHIVES / v / f"{row['base']}.tar"), "-C", tmp, "--one-top-level=" + v,
                            *members], check=True)
        for im in images:
            fig, axes = plt.subplots(len(fsvv.VERSIONS), 5, figsize=(12.5, 2.6 * len(fsvv.VERSIONS)))
            for ax_row, v in zip(axes, fsvv.VERSIONS):
                t1 = np.asarray(nib.load(Path(tmp) / v / im / FILES[0]).dataobj)
                seg = np.asarray(nib.load(Path(tmp) / v / im / FILES[1]).dataobj).astype(int)
                for ax, s_t1, s_seg in zip(ax_row, slices(t1), slices(seg)):
                    ax.imshow(s_t1, cmap="gray", vmin=0, vmax=np.percentile(t1, 99.5))
                    ax.imshow(outlines(s_seg, colors), interpolation="nearest")
                    ax.set_axis_off()
                ax_row[0].text(-10, s_t1.shape[0] / 2, v, rotation=90, va="center", ha="right", fontsize=11)
            fig.suptitle(im, fontsize=11)
            fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.01, wspace=0.02, hspace=0.04)
            fig.savefig(out / f"{im}.png", dpi=90)
            plt.close(fig)
            print("wrote", out / f"{im}.png", flush=True)


if __name__ == "__main__":
    make_list() if sys.argv[1] == "list" else snapshot(int(sys.argv[1]))
