# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.11 (fsvv)
#     language: python
#     name: fsvv
# ---

# %% [markdown]
# # 12 · Repeat runs and cross-cluster variability vs between-version differences (Reviewer 1)
#
# 20 baseline images of the paper's cohort (10 PD, 10 HC; `scripts/crosscluster/make_list.py`) were re-run twice
# (rep1, rep2) on the current cluster with the paper's cross-sectional pipeline, each version on the exact input
# it received in the paper's run (`mri/orig/001.mgz`). Comparisons, per version and measure (aseg volumes, Destrieux
# volume, surface area and thickness):
#
# - **rep1 vs rep2**: same cluster, same input, same container — run-to-run repeatability;
# - **rep vs paper**: the paper's run of the same input (another cluster for FS5–7) — cross-cluster variability;
# - **between versions**: the same images in the paper's runs (version pairs of the paper).
#
# Measure of difference: absolute relative difference |A − B| / B (%), per image and measure. For each measure, the
# median over images; then the distribution of these medians over measures.
#
# FS6 did not finish one image in either replica within 43 h: `mris_fix_topology` was still correcting a very large
# left-hemisphere defect (66 h for this step in the paper's run, 73 h in total). The image is left out of the FS6
# comparisons.
# Labels FS8 does not produce or defines differently (`fsvv.incomparable_labels`) are left out of the FS8 version pairs.
#
# Outputs (aggregate only): `outputs/crosscluster_*.csv`.

# %%
import numpy as np
import pandas as pd

import fsvv

CC = fsvv.DATA / "crosscluster"
subjects = pd.read_csv(CC / "rep1" / "lists" / "qpn_jobs.csv", index_col="task")["subject"].tolist()
runs = {"rep1": fsvv.extract_measures(subjects, root=CC / "rep1" / "stats"),
        "rep2": fsvv.extract_measures(subjects, root=CC / "rep2" / "stats"),
        "paper": fsvv.extract_measures(subjects, root=fsvv.PAPER_ROOT)}

KINDS = {"aseg": "Subcortical volume", "vol": "Cortical volume", "surf": "Surface area", "ct": "Cortical thickness"}
keys = [(key, "aseg" if key in fsvv.ASEG_ROIS else kind, stem) for key, kind, stem in fsvv.measure_keys()]

# Completed runs per version (an image counts if its aseg.stats was read)
done = pd.DataFrame({r: {v: int(t[f"TIV_{v}"].notna().sum()) for v in fsvv.VERSIONS} for r, t in runs.items()})
done.to_csv(fsvv.out("crosscluster_completed.csv"))
done


# %%
def rel_diff(a, b, version_a, version_b=None):
    """|A - B| / |B| (%) per image (rows) and measure (columns key); images missing in either run dropped."""
    version_b = version_b or version_a
    cols = {}
    for key, _, stem in keys:
        x, y = a[f"{stem}_{version_a}"], b[f"{stem}_{version_b}"]
        cols[key] = 100 * (x - y).abs() / y.abs()
    d = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)
    ok = a[f"TIV_{version_a}"].notna() & b[f"TIV_{version_b}"].notna()
    return d[ok]


INCOMPARABLE = fsvv.incomparable_labels(runs["paper"])
print("left out of the FS8 pairs:", sorted(INCOMPARABLE))
comparisons = {}  # (comparison, version or pair) -> per-image x per-measure relative differences
for v in fsvv.VERSIONS:
    comparisons[("rep1 vs rep2", v)] = rel_diff(runs["rep1"], runs["rep2"], v)
    comparisons[("rep1 vs paper", v)] = rel_diff(runs["rep1"], runs["paper"], v)
    comparisons[("rep2 vs paper", v)] = rel_diff(runs["rep2"], runs["paper"], v)
for a, b in fsvv.PAIRS:
    d = rel_diff(runs["paper"], runs["paper"], a, b)
    if "FS8" in (a, b):  # labels FS8 does not produce or defines differently: not a measurement difference
        d = d.drop(columns=sorted(INCOMPARABLE))
    comparisons[("between versions", fsvv.label((a, b)))] = d

# %% [markdown]
# ## Bitwise repeatability
#
# Fraction of images whose measures are all identical between the two runs, and fraction of identical values.

# %%
rows = []
for (comp, v), d in comparisons.items():
    if comp == "between versions":
        continue
    rows.append(dict(comparison=comp, version=v, images=len(d), identical_images=int((d.fillna(0) == 0).all(axis=1).sum()),
                     identical_values_pct=100 * float((d == 0).sum().sum() / d.notna().sum().sum())))
identical = pd.DataFrame(rows)
identical.to_csv(fsvv.out("crosscluster_identical.csv"), index=False)
identical.round(1)

# %% [markdown]
# ## Size of the differences
#
# For each measure: median over images of |A − B| / B. Table: median and 90th percentile of these medians over the
# measures of each kind.

# %%
per_measure = []
for (comp, v), d in comparisons.items():
    med = d.median()
    for key, kind, _ in keys:
        if key not in med:
            continue
        per_measure.append(dict(comparison=comp, version=v, key=key, kind=kind, median_rel_diff=med[key]))
per_measure = pd.DataFrame(per_measure)
per_measure.to_csv(fsvv.out("crosscluster_measures.csv"), index=False)

summary = (per_measure.groupby(["comparison", "version", "kind"], sort=False)["median_rel_diff"]
           .agg(median="median", p90=lambda x: x.quantile(.9)).unstack("kind"))
summary = summary[[(s, k) for k in KINDS for s in ("median", "p90")]]
summary.to_csv(fsvv.out("crosscluster_summary.csv"))
summary.round(3)

# %% [markdown]
# ## Non-identical repeat runs vs between-version differences
#
# Per comparison and kind: number of measures that differ in at least one image, and the largest and median
# non-zero |A − B| / B (%) over images and measures. The between-version rows give the scale of the paper's
# differences on the same images.

# %%
rows = []
for (comp, v), d in comparisons.items():
    for kind in KINDS:
        k = [key for key, kd, _ in keys if kd == kind and key in d]
        x = d[k].values.ravel()
        x = x[~np.isnan(x)]
        nz = x[x > 0]
        rows.append(dict(comparison=comp, version=v, kind=kind, values=len(x),
                         measures_differing=int((d[k] > 0).any().sum()), measures=len(k),
                         images_differing=int((d[k] > 0).any(axis=1).sum()),
                         nonzero_pct=100 * len(nz) / len(x) if len(x) else np.nan,
                         median_nonzero=np.median(nz) if len(nz) else 0.0, max=x.max() if len(x) else np.nan,
                         median_all=np.median(x) if len(x) else np.nan))
nonzero = pd.DataFrame(rows)
nonzero.to_csv(fsvv.out("crosscluster_differences.csv"), index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
nonzero.round(4)
