"""Inputs of the vertex-wise GLMs (cortical thickness, fsaverage, fwhm10), same models as the paper.

For the longitudinal / clinical cohort (`stage_two_clinical.csv` of 01_cohort, HC and PD-non-MCI), writes in <out>:
  fsgd/   the 4 FSGD files built as in 00_pipeline (group / corr x baseline / long)
  con/    the 4 contrast matrices of the paper
  subjects/FSx/   SUBJECTS_DIR of each version: symlinks to the baseline cross-sectional run and the base
                  template of every participant (paper runs in data/FSx, new participants in
                  data/new_participants/light/FSx) + fsaverage of the container
Participants with a run the GLMs read (long runs, cross run of the first visit) missing in any version
(`not_processed.csv` of 01_cohort) are left out, so every
version fits the same participants.
--covariates qc adds fsvv.QC_COVARIATES (MRIQC snr_total, scanner manufacturer dummies) to every model (reviewers);
the contrasts get zeros for the new regressors (FSGD default DODS: one slope per class and variable).

Usage: python make_glm_inputs.py [--stage-two CSV] [--not-processed CSV] [--out DIR] [--covariates {none,qc}]
       (defaults: outputs/stage_two_clinical.csv, outputs/not_processed.csv, outputs/glm or outputs/glm_qc)
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

# fsaverage as seen inside each container (as recon-all links it)
FSAVERAGE = {"FS5": "/usr/local/freesurfer/subjects/fsaverage", "FS6": "/usr/local/freesurfer/subjects/fsaverage",
             "FS7": "/usr/local/freesurfer/subjects/fsaverage", "FS8": "/usr/local/freesurfer/8.2.0-1/subjects/fsaverage"}
# contrast matrices of the paper (no trailing newline, as the originals)
CONTRASTS = {"con_group_HC_PDnonMCI_base.mtx": "1 -1 0 0 0 0", "con_group_HC_PDnonMCI_long.mtx": "1 -1 0 0 0 0 0 0",
             "con_corr_UPDRS_base.mtx": "0 1 0 0", "con_corr_UPDRS_long.mtx": "0 1 0 0 0"}
# files read by mris_preproc --cache-in for each subject
CACHE = {"first_visit": "surf/{hemi}.thickness.fwhm10.fsaverage.mgh",
         "PATNO_base": "surf/{hemi}.long.thickness-rate.fwhm10.fsaverage.mgh"}


def write_fsgd(path, header, rows):
    """Header lines + tab-separated rows, as in the paper's notebooks (pandas default number format)."""
    with open(path, "w") as f:
        f.write(header)
    rows.assign(Input="Input")[["Input", *rows.columns]].to_csv(path, mode="a", sep="\t", index=False, header=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-two", default=fsvv.OUT / "stage_two_clinical.csv")
    ap.add_argument("--not-processed", default=fsvv.OUT / "not_processed.csv")
    ap.add_argument("--out")
    ap.add_argument("--covariates", choices=["none", "qc"], default="none")
    args = ap.parse_args()
    out = Path(args.out or fsvv.OUT / ("glm" if args.covariates == "none" else "glm_qc"))
    extra = fsvv.QC_COVARIATES if args.covariates == "qc" else []

    t = pd.read_csv(args.stage_two)
    t = t[t["dx_group"].isin(["HC", "PD-non-MCI"])].copy()
    t["PATNO_base"] = "sub-" + t["PATNO"].astype(str) + "_base"
    # The paper's FSGD files use the time between the two MRI scans in average months (np.timedelta64(1, "M"),
    # 2629746 s) / 12, for both groups (same operations, so the values match to the last digit); 01_cohort's
    # durationT2_T1_y (calendar months for HC, days / 365.25 for PD) is kept for Table 1.
    delta = pd.to_datetime(t["Study Date_NX"]) - pd.to_datetime(t["Study Date"])
    t["durationT2_T1_y"] = delta.dt.total_seconds() / 2629746 / 12
    # only the runs the GLMs read matter: the long runs (base template) and the cross run of the first visit
    np_ = pd.read_csv(args.not_processed) if Path(args.not_processed).exists() else pd.DataFrame(
        columns=["PATNO", "PATNO_id", "stream"])
    used = (np_["stream"] == "long") | np_["PATNO_id"].isin(t["first_visit"])
    missing = set(np_.loc[used, "PATNO"])
    n0 = len(t)
    t = t[~t["PATNO"].isin(missing)]
    if extra:
        download = pd.read_csv(fsvv.OUT / "download_data.csv", index_col=0)
        t = t.join(fsvv.qc_covariates(download), on="PATNO").dropna(subset=extra)
    print(f"{len(t)} participants ({n0 - len(t)} left out: run missing in a version"
          f"{' or no QC covariate' if extra else ''}):", t["dx_group"].value_counts().to_dict())

    hc_pd = pd.concat([t[t["dx_group"] == "HC"], t[t["dx_group"] == "PD-non-MCI"]])
    pdn = t[t["dx_group"] == "PD-non-MCI"].assign(dx_group="PDnonMCI")
    (out / "fsgd").mkdir(parents=True, exist_ok=True)
    group = "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\n"
    corr = "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName NP3TOT\nClass PDnonMCI\n"
    # headers exactly as the paper's without extra covariates ("age sex " keeps its trailing space)
    base_vars = "age sex " + " ".join(extra) + "\n"
    long_vars = "age sex durationT2_T1_y" + "".join(" " + c for c in extra) + "\n"
    write_fsgd(out / "fsgd/fsgd_cort_group_HC_PDnonMCI_baseline.fsgd", group + "Variables " + base_vars,
               hc_pd[["first_visit", "dx_group", "AGE_AT_VISIT", "SEX", *extra]])
    write_fsgd(out / "fsgd/fsgd_cort_group_HC_PDnonMCI_long.fsgd", group + "Variables " + long_vars,
               hc_pd[["PATNO_base", "dx_group", "AGE_AT_VISIT", "SEX", "durationT2_T1_y", *extra]])
    write_fsgd(out / "fsgd/fsgd_corr_PDnonMCI_group.fsgd", corr + "Variables NP3TOT " + base_vars,
               pdn[["first_visit", "dx_group", "NP3TOT", "AGE_AT_VISIT", "SEX", *extra]])
    write_fsgd(out / "fsgd/fsgd_corr_PDnonMCI_group_long.fsgd", corr + "Variables NP3TOT " + long_vars,
               pdn[["PATNO_base", "dx_group", "NP3TOT_change", "AGE_AT_VISIT", "SEX", "durationT2_T1_y", *extra]])

    (out / "con").mkdir(exist_ok=True)
    for name, c in CONTRASTS.items():
        n_classes = 2 if "group" in name else 1  # one zero per class for each extra variable
        (out / "con" / name).write_text(c + " 0" * (n_classes * len(extra)))

    absent = []
    for v in fsvv.VERSIONS:
        sd = out / "subjects" / v
        sd.mkdir(parents=True, exist_ok=True)
        for old in sd.iterdir():  # links of a previous run (participants may have changed)
            if old.is_symlink():
                old.unlink()
        links = {"fsaverage": FSAVERAGE[v]}
        for col, cache in CACHE.items():
            for s in t[col]:
                d = fsvv.subject_dir(v, s).resolve()
                links[s] = str(d)
                absent += [f"{v} {s} {cache.format(hemi=h)}" for h in fsvv.HEMIS
                           if not (d / cache.format(hemi=h)).exists()]
        for name, target in links.items():
            os.symlink(target, sd / name)
    if absent:
        sys.exit(f"{len(absent)} thickness files missing, e.g. {absent[:3]}")
    print(f"FSGD, contrasts and SUBJECTS_DIR (FS5-8) written in {out}")


if __name__ == "__main__":
    main()
