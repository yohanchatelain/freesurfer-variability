"""Job lists of the cross-cluster / repeat-run reproducibility test (Reviewer 1, within-version variability).

Draws N baseline images of the paper's participants (half PD, half HC; fixed seed) processed in all four versions,
and copies the input each version actually received in the paper's run (mri/orig/001.mgz of data/FSx/<image>) to
<dir>/inputs/FSx/<image>.mgz. Each image is re-run twice on the current cluster with the paper's cross-sectional
pipeline (scripts/qpn/run_cross.py, same containers and descriptors), so we can compare:
  rep1 vs rep2 (same cluster, same input), rep1 vs the paper's run (other cluster for FS5-7), and the between-version
  differences on the same images.
Writes <dir>/rep1/lists/qpn_jobs.csv and <dir>/rep2/lists/qpn_jobs.csv (same rows; run_cross.py reads this name).
Participant-level: <dir> must be gitignored (default data/crosscluster).

Usage: python scripts/crosscluster/make_list.py [--n 20] [--dir data/crosscluster]
Then:  for r in rep1 rep2; do for v in FS5 FS6 FS7 FS8; do FSVV_QPN_DIR=<dir>/$r FSVV_QPN_WORK=$SCRATCH/fsvv_cc/$r \
           FSVV_ACCOUNT=<account> scripts/qpn/submit.sh $v; done; done
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dir", type=Path, default=fsvv.DATA / "crosscluster")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    download = pd.read_csv(fsvv.DATA / "tables" / "download_data.csv", index_col=0)  # the paper's cohort
    bl = download[download["EVENT_ID"] == "BL"].drop_duplicates("PATNO").copy()
    bl["image"] = "sub-" + bl["PATNO"].astype(str) + "_ses-BL"
    ok = bl["image"].map(lambda i: all((fsvv.PAPER_ROOT / v / i / "mri/orig/001.mgz").exists()
                                       and (fsvv.PAPER_ROOT / v / i / "stats/aseg.stats").exists()
                                       for v in fsvv.VERSIONS))
    bl = bl[ok]
    bl["group"] = bl["dx_group"].map(lambda g: "HC" if g == "HC" else "PD")
    pick = pd.concat([g.sample(args.n // 2, random_state=args.seed) for _, g in bl.groupby("group")])

    rows = []
    for image in pick["image"]:
        row = {"subject": image}
        for v in fsvv.VERSIONS:
            dst = args.dir / "inputs" / v / f"{image}.mgz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fsvv.PAPER_ROOT / v / image / "mri/orig/001.mgz", dst)
            row[f"t1w_{v}"] = str(dst.resolve())
        rows.append(row)
    jobs = pd.DataFrame(rows).rename_axis("task")
    for rep in ("rep1", "rep2"):
        (args.dir / rep / "lists").mkdir(parents=True, exist_ok=True)
        jobs.to_csv(args.dir / rep / "lists" / "qpn_jobs.csv")
    print(f"{len(bl)} eligible baseline images; picked {len(jobs)} ({pick['group'].value_counts().to_dict()}) "
          f"-> {args.dir}")


if __name__ == "__main__":
    main()
