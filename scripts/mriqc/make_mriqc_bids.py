"""BIDS input of MRIQC for the first visits of the cross-sectional cohort that have no MRIQC output yet.

As for the paper, MRIQC runs on the first visit of each participant (the visit 02_software_variability attaches the
metrics to), with the layout sub-<PATNO>/ses-<visit>/anat/sub-<PATNO>_T1w.nii. Participants with a run missing in
any version (not_processed.csv) are skipped, and so are first visits that already have an output in
fsvv.MRIQC_ROOTS (the paper's). Images are symlinks to the T1w used by FreeSurfer for the revision runs
(<work>/inputs, see scripts/new_participants). Writes <out>/bids and <out>/participants.txt (one label per line).

Usage: python make_mriqc_bids.py [--out outputs/mriqc] [--inputs $SCRATCH/fsvv_newpart/inputs]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fsvv.OUT / "mriqc")
    ap.add_argument("--inputs", default=Path(os.environ["SCRATCH"]) / "fsvv_newpart" / "inputs")
    args = ap.parse_args()
    bids = Path(args.out) / "bids"

    download = pd.read_csv(fsvv.OUT / "download_data.csv", index_col=0)
    missing = set(pd.read_csv(fsvv.OUT / "not_processed.csv")["PATNO"])
    first = download[~download["PATNO"].isin(missing)].drop_duplicates("PATNO")[["PATNO", "EVENT_ID"]]
    todo, absent = [], []
    for patno, event in first.itertuples(index=False):
        if fsvv.mriqc_file(patno, event).exists():
            continue
        found = sorted((Path(args.inputs) / f"sub-{patno}" / f"ses-{event}" / "anat").glob("*.nii*"))
        if len(found) != 1:
            absent.append(f"sub-{patno}_ses-{event} ({len(found)} images)")
            continue
        anat = bids / f"sub-{patno}" / f"ses-{event}" / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        link = anat / f"sub-{patno}_T1w{''.join(found[0].suffixes)}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(found[0].resolve())
        todo.append(str(patno))
    if absent:
        sys.exit(f"{len(absent)} first visits without a single input image: {absent[:5]}")
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "data", "BIDSVersion": "1.0.2"}))
    (Path(args.out) / "participants.txt").write_text("\n".join(todo) + "\n")
    print(f"{len(first)} first visits; {len(todo)} need MRIQC -> {bids}")


if __name__ == "__main__":
    main()
