"""BIDS input of MRIQC for the QPN participants processed with FreeSurfer (lists/qpn_jobs.csv).

Same layout as the paper's QPN MRIQC input (sub-<label>/ses-01/anat/sub-<label>_T1w.nii[.gz] + JSON sidecar):
symlinks to the T1w image FreeSurfer used and its sidecar. Participants that already have an MRIQC output are
left out of participants.txt. Run then scripts/mriqc/run_mriqc.sbatch with MRIQC_DIR=<out>.

Usage: python make_mriqc_bids.py [--qpn data/qpn] [--out <qpn>/mriqc]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def link(path, target):
    if path.is_symlink():
        path.unlink()
    path.symlink_to(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpn", type=Path, default=ROOT / "data" / "qpn")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.qpn / "mriqc"
    bids = out / "bids"

    todo = []
    with open(args.qpn / "lists" / "qpn_jobs.csv") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        sub = row["subject"]
        t1w = Path(row["t1w"])
        ext = "".join(t1w.suffixes[-2:]) if t1w.name.endswith(".nii.gz") else t1w.suffix
        anat = bids / sub / "ses-01" / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        link(anat / f"{sub}_T1w{ext}", t1w.resolve())
        sidecar = t1w.with_name(t1w.name[: -len(ext)] + ".json")
        if sidecar.exists():
            link(anat / f"{sub}_T1w.json", sidecar.resolve())
        if not (out / "output" / sub / "ses-01" / "anat" / f"{sub}_T1w.json").exists():
            todo.append(sub.removeprefix("sub-"))
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "QPN", "BIDSVersion": "1.0.2"}))
    (out / "participants.txt").write_text("".join(f"{s}\n" for s in todo))
    print(f"{len(rows)} participants; {len(todo)} need MRIQC -> {bids}")


if __name__ == "__main__":
    main()
