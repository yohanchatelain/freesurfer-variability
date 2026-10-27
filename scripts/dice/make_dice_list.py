"""List the segmentation Dice computations missing for the cross-sectional cohort (input of run_dice.sbatch).

Every image of `download_data.csv` (01_cohort) whose participant is processed in all versions (`not_processed.csv`)
needs, for each version pair of the paper and each segmentation, the mri_seg_overlap JSON read by
fsvv.read_seg_dice. Existing JSONs (fsvv.DICE_ROOTS: the paper's in data/stats/dice, the new ones in
outputs/dice) are skipped. One tab-separated line per missing JSON: segmentation 1, segmentation 2, output JSON.

Usage: python make_dice_list.py [--out outputs/dice/todo.tsv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

SEGMENTATIONS = {"aparc2009": "mri/aparc.a2009s+aseg.mgz", "aparc": "mri/aparc+aseg.mgz", "aseg": "mri/aseg.mgz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fsvv.OUT / "dice" / "todo.tsv")
    args = ap.parse_args()

    images = pd.read_csv(fsvv.OUT / "download_data.csv")
    missing = set(pd.read_csv(fsvv.OUT / "not_processed.csv")["PATNO"])
    images = sorted(set(images.loc[~images["PATNO"].isin(missing), "PATNO_id"]))
    rows, done = [], 0
    for subj in images:
        for pair in fsvv.PAIRS:
            for name, seg in SEGMENTATIONS.items():
                if fsvv.dice_file(pair, subj, name).exists():
                    done += 1
                    continue
                a, b = (fsvv.subject_dir(v, subj).resolve() / seg for v in pair)
                rows.append((a, b, fsvv.DICE_ROOTS[0] / f"dice_{fsvv.tag(pair)}_{subj}.{name}.json"))
    absent = [str(p) for r in rows for p in r[:2] if not p.exists()]
    if absent:
        sys.exit(f"{len(absent)} segmentations missing, e.g. {absent[:3]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", header=False, index=False)
    print(f"{len(images)} images: {done} Dice JSONs exist, {len(rows)} to compute -> {args.out}")


if __name__ == "__main__":
    main()
