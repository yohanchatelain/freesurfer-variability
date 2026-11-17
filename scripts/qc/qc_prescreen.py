"""Automated QC pre-screen of the images added for the revision, calibrated on the paper's visually QC'd images.

Every image analysed in 02 (outputs/aseg_table4.csv) gets these measures, per version or version pair:
- surface holes (aseg.stats `SurfaceHoles`, the Euler-number based count; FS6+), worst version;
- aseg volume outliers: number of aseg structures more than 5 robust SD (median/MAD of the reference images) from
  the reference, plus |robust z| of TIV and of the mean cortical thickness, worst version;
- version disagreement: median over regions of the relative difference between versions, worst pair;
- segmentation agreement: mean aseg Dice between versions, worst pair (low = one version failed);
- MRIQC (first visit only): snr_total, cnr (low = bad), cjv (high = bad).
Reference = images of the paper's participants (they passed the paper's visual QC). An image is flagged on a measure
when it is beyond the 99.5th percentile of the reference (0.5th for "low = bad" measures), so about 0.5% of the
reference images are flagged per measure by construction; the reference flag rate is printed as a check.

Writes outputs/qc/prescreen.csv (one row per image: measures, flags, reasons) and outputs/qc/flagged.txt (new
images to look at). Participant-level: outputs/ only. Prints group counts only.
Usage (compute node): python scripts/qc/qc_prescreen.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

QC = fsvv.OUT / "qc"
Q = 0.995  # reference quantile beyond which an image is flagged


def surface_holes(image, version):
    try:
        for line in open(fsvv.subject_dir(version, image) / "stats" / "aseg.stats"):
            if line.startswith("# Measure SurfaceHoles"):
                return float(line.split(",")[3])
    except FileNotFoundError:
        pass
    return np.nan


def robust_z(x, ref):
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan) if isinstance(mad, pd.Series) else (mad or np.nan)  # constant reference: no z
    return (x - med) / mad


def main():
    table = pd.read_csv(fsvv.OUT / "aseg_table4.csv").set_index("PATNO_id")
    patno = table.index.str.extract(r"sub-(\d+)_ses", expand=False).astype(int)
    is_ref = pd.Series(patno.isin(fsvv.paper_participants()), index=table.index)
    feats = pd.DataFrame(index=table.index)

    feats["holes"] = pd.concat({v: pd.Series({i: surface_holes(i, v) for i in table.index})
                                for v in fsvv.VERSIONS}, axis=1).max(axis=1)

    out_rois, tiv_z, ct_z = [], [], []
    for v in fsvv.VERSIONS:
        cols = [f"{r}_{v}" for r in fsvv.ASEG_ROIS if f"{r}_{v}" in table]
        z = robust_z(table[cols], table.loc[is_ref, cols])
        out_rois.append((z.abs() > 5).sum(axis=1))
        tiv_z.append(robust_z(table[f"TIV_{v}"], table.loc[is_ref, f"TIV_{v}"]).abs())
        ct = table.filter(regex=rf"_ct_{v}$").mean(axis=1)
        ct_z.append(robust_z(ct, ct[is_ref]).abs())
    feats["aseg_outliers"] = pd.concat(out_rois, axis=1).max(axis=1)
    feats["tiv_absz"] = pd.concat(tiv_z, axis=1).max(axis=1)
    feats["ct_absz"] = pd.concat(ct_z, axis=1).max(axis=1)

    disagree, dice = [], []
    for pair in fsvv.PAIRS:
        pct = table.filter(regex=rf"^{re.escape(fsvv.prefix(pair))}_pct_").replace([np.inf, -np.inf], np.nan)
        disagree.append(pct.median(axis=1))
        d = table.filter(regex=rf"^dice_FS{fsvv.tag(pair)}_")
        d = d[[c for c in d if c.split("_", 2)[2] in fsvv.ASEG_ROIS]]
        dice.append(d.mean(axis=1))
        print(f"{fsvv.label(pair)}: {pct.shape[1]} difference columns, {d.shape[1]} aseg Dice columns")
    feats["disagreement"] = pd.concat(disagree, axis=1).max(axis=1)
    feats["dice_min"] = pd.concat(dice, axis=1).min(axis=1)

    download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
    for p, e in download.drop_duplicates("PATNO")[["PATNO", "EVENT_ID"]].itertuples(index=False):
        i, f = f"sub-{p}_ses-{e}", fsvv.mriqc_file(p, e)
        if i in feats.index and f.exists():
            iqm = json.load(open(f))
            for m in ("snr_total", "cnr", "cjv"):
                feats.loc[i, m] = iqm[m]

    high = ["holes", "aseg_outliers", "tiv_absz", "ct_absz", "disagreement", "cjv"]
    low = ["dice_min", "snr_total", "cnr"]
    flags = pd.DataFrame(index=feats.index)
    for m in high + low:
        ref = feats.loc[is_ref, m].dropna()
        if ref.empty:
            continue
        thr = ref.quantile(Q if m in high else 1 - Q)
        flags[m] = feats[m] > thr if m in high else feats[m] < thr
    feats["n_flags"] = flags.sum(axis=1)
    feats["reasons"] = flags.apply(lambda r: " ".join(r.index[r]), axis=1)
    feats["new"] = ~is_ref
    feats.to_csv(QC / "prescreen.csv")
    flagged = feats.index[feats["new"] & (feats["n_flags"] > 0)]
    (QC / "flagged.txt").write_text("".join(f"{i}\n" for i in sorted(flagged)))

    # group counts only
    summary = pd.DataFrame({"reference (paper, passed QC)": flags[is_ref].mean() * 100,
                            "new images": flags[~is_ref].mean() * 100}).round(1)
    print(f"images: reference {int(is_ref.sum())}, new {int((~is_ref).sum())}")
    print("% flagged per measure:\n", summary.to_string())
    print("new images by number of flags:", feats.loc[~is_ref, "n_flags"].value_counts().sort_index().to_dict())
    print("reference images by number of flags:", feats.loc[is_ref, "n_flags"].value_counts().sort_index().to_dict())
    print(f"flagged new images: {len(flagged)} -> {QC / 'flagged.txt'}")


if __name__ == "__main__":
    main()
