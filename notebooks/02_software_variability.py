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
# # 02 · Software variability between FreeSurfer versions (PPMI, cross-sectional)
#
# 1. Structural measures (aseg volumes; Destrieux volume, surface area, thickness) for FS5/6/7/8
# 2. Paired t-tests between versions, share of regions that differ, Dice of the sets of significant regions
# 3. PD vs HC differences in software variability; MRIQC metrics and their correlation with variability
# 4. Segmentation overlap between versions (`mri_seg_overlap` Dice)
# 5. Figures (static, saved to `outputs/figures`)
#
# Bonferroni: a test is significant when p < 0.05 / n_tests (n = 187 volumes, 148 surfaces, 148 thicknesses).

# %%
# %matplotlib inline
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import fsvv
from fsvv import PAIRS, VERSIONS, label, prefix, tag

plt.rcParams.update({"figure.dpi": 80, "savefig.dpi": 200, "font.size": 8})
pd.set_option("display.max_rows", 60)

# %% [markdown]
# ## Subjects
# Both images of every participant of the cross-sectional cohort (`download_data.csv` of `01_cohort`) processed
# in every version (`not_processed.csv` lists the others), minus the images excluded after visual QC.

# %%
# Images excluded after QC; the list holds participant IDs, so it stays in data/ (not versioned)
EXCLUDED = {s.strip() for s in open(fsvv.inp("excluded_images.txt")) if s.strip()}

download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
download["PATNO_id"] = "sub-" + download["PATNO"].astype(str) + "_ses-" + download["EVENT_ID"]
# Participants with a run missing in any version (failed or never processed), from 01_cohort
not_processed = set(pd.read_csv(fsvv.inp("not_processed.csv"))["PATNO"])
subjects = sorted(set(download.loc[~download["PATNO"].isin(not_processed), "PATNO_id"]) - EXCLUDED)
dx_group = download.drop_duplicates("PATNO_id").set_index("PATNO_id")["dx_group"]
print(len(subjects), "images;", dx_group.reindex(subjects).value_counts(dropna=False).to_dict())

# %% [markdown]
# ## Structural measures and between-version differences

# %%
measures = fsvv.extract_measures(subjects)
diffs = fsvv.version_differences(measures)
table = pd.concat([measures, diffs], axis=1)
table.insert(0, "dx_group", dx_group.reindex(table.index))
print(table.shape, "| missing values per version:",
      {v: int(measures.filter(regex=f"_{v}$").isna().sum().sum()) for v in VERSIONS})

# Check: same values as the table used for the paper (aseg_table3.csv).
ref = pd.read_csv(fsvv.DATA / "tables" / "aseg_table3.csv").set_index("PATNO_id").reindex(table.index)
common = [c for c in measures.columns if c in ref.columns]
delta = (measures[common] - ref[common]).abs().max().max()
print(f"{len(common)}/{measures.shape[1]} measure columns in aseg_table3.csv; max |difference| = {delta:.3g}")

# %% [markdown]
# ## Paired t-tests between versions
# FS8 doesn't produce some aseg labels (vessel, choroid plexus, optic chiasm, 5th ventricle: volume 0 in every image).
# They are kept in the tests, but their relative difference is 1 for every image, so they are significant in every FS8
# pair by construction (absence of the label, not a measurement difference). `absent_in_FS8` marks them in
# `version_paired_ttests.csv`. The Volume column of the share of regions below includes them (5-6 of the 187 volume
# tests in each FS8 pair, about 3 percentage points).

# %%
tests = fsvv.paired_version_tests(table)
absent_fs8 = fsvv.absent_labels(table, "FS8")
tests["absent_in_FS8"] = tests["key"].isin(absent_fs8) & tests["pair"].str.startswith("8")
tests.to_csv(fsvv.out("version_paired_ttests.csv"), index=False)
sets = fsvv.significant_sets(tests)
print("labels FS8 doesn't produce:", absent_fs8)
print("significant FS8-pair volume tests due to these labels:",
      {pair: len(sets[(pair, "vol")] & set(absent_fs8)) for pair in ["87", "86", "85"]})
prop = fsvv.proportion_significant(sets)
(prop * 100).round(1)  # % of regions with a significant difference

# %% [markdown]
# ## Agreement of the sets of significant regions (Sørensen–Dice)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
    m = fsvv.dice_between_sets(sets, kind)
    sns.heatmap(m, annot=True, fmt=".2f", vmin=0, vmax=1, square=True, cmap="viridis", ax=ax,
                cbar=ax is axes[-1], annot_kws={"size": 7})
    ax.set_title(f"Sørensen–Dice of significant regions: {fsvv.MEASURE_NAMES[kind]}")
fig.tight_layout()
fsvv.savefig(fig, "dice_significant_sets")

# %% [markdown]
# ## PD vs HC differences in software variability (relative differences, two-sample t-tests)
# One image per participant: the first visit (baseline image of the cohort, the one MRIQC was run on). The submitted
# paper used both visits, so each participant counted twice; that version is kept for comparison in the last section
# of this part.

# %%
is_pd = table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])
is_hc = table["dx_group"] == "HC"
first_table = fsvv.first_visits(table, download).copy()
is_pd1, is_hc1 = is_pd.loc[first_table.index], is_hc.loc[first_table.index]
print(len(first_table), "first visits:", int(is_pd1.sum()), "PD,", int(is_hc1.sum()), "HC")
group_tests = fsvv.group_variability_tests(first_table, is_pd1, is_hc1)
group_tests.to_csv(fsvv.out("pd_vs_hc_variability_ttests.csv"), index=False)
group_tests["significant"] = group_tests["p"] < fsvv.ALPHA / group_tests["kind"].map(fsvv.N_TESTS)
group_tests.groupby(["pair", "kind"]).agg(min_p=("p", "min"), n_significant=("significant", "sum")).unstack("kind")

# %% [markdown]
# ## Software variability separately in PD and HC

# %%
by_group = {g: fsvv.paired_version_tests(table.loc[mask]) for g, mask in [("PD", is_pd), ("HC", is_hc)]}
pd.concat({g: fsvv.proportion_significant(fsvv.significant_sets(t)) for g, t in by_group.items()},
          axis=1).mul(100).round(1)  # %

# %% [markdown]
# ## MRIQC
# MRIQC 22.0.1 was run on the first visit of each subject (paper: `data/source_files/clinical/mriqc`; new participants: `outputs/mriqc`, same container and options). Metrics are attached to that image only
# (the original analysis attached them to both visits, so each subject counted twice in the PD vs HC t-tests and the
# correlations): every test below has one row per participant.

# %%
qc = download[~download["PATNO"].isin(not_processed)].copy()
missing = []
for patno, event in qc.drop_duplicates("PATNO")[["PATNO", "EVENT_ID"]].itertuples(index=False):
    f = fsvv.mriqc_file(patno, event)  # paper outputs, then the revision runs (scripts/mriqc)
    if not f.exists():
        missing.append(f"sub-{patno}_ses-{event}")
        continue
    iqm = json.load(open(f))
    for m in fsvv.MRIQC_METRICS:
        qc.loc[qc["PATNO_id"] == f"sub-{patno}_ses-{event}", m] = iqm[m]
print(f"MRIQC metrics for {qc['cjv'].notna().sum()} first visits; no MRIQC output for {len(missing)} first visits:",
      missing[:10])

from scipy.stats import ttest_ind
pd_qc, hc_qc = qc[qc["dx_group"].isin(["PD-MCI", "PD-non-MCI"])], qc[qc["dx_group"] == "HC"]
pd.DataFrame([(m, pd_qc[m].notna().sum(), hc_qc[m].notna().sum(), *ttest_ind(pd_qc[m], hc_qc[m], nan_policy="omit"))
              for m in fsvv.MRIQC_METRICS], columns=["metric", "n_PD", "n_HC", "t", "p"]).set_index("metric")

# %% [markdown]
# ### Correlation between MRIQC metrics and software variability
# First visits only (the images MRIQC was run on), one per participant.

# %%
qc_table = table.join(qc.drop_duplicates("PATNO_id").set_index("PATNO_id")[fsvv.MRIQC_METRICS])
qc_corr = fsvv.qc_correlations(qc_table)
qc_corr.to_csv(fsvv.out("mriqc_variability_correlations.csv"), index=False)
qc_corr["significant"] = qc_corr["p"] < fsvv.ALPHA / (qc_corr["kind"].map(fsvv.N_TESTS) * len(fsvv.MRIQC_METRICS))
qc_corr.groupby(["metric", "kind"]).agg(max_abs_r=("r", lambda r: r.abs().max()),
                                         n_significant=("significant", "sum")).unstack("kind")

# %% [markdown]
# ### PD vs HC differences in software variability, adjusted for image quality and scanner
# Reviewer 3 asked for the group analyses to control for site and for the SNR metrics that differ between groups.
# `fsvv.group_variability_models` fits `|Δ%| ~ group + snr_total + manufacturer` (GE and Philips dummies; Siemens =
# reference; manufacturer stands in for site) on the first visit of each participant, the image MRIQC was run on.
# The table compares: both visits (submitted paper, each participant counted twice), first visits (main analysis,
# the t-tests above), and first visits with covariates.

# %%
patno = first_table.index.str.extract(r"sub-(\d+)_", expand=False).astype(int)
first_table[fsvv.QC_COVARIATES] = fsvv.qc_covariates(download).loc[patno, fsvv.QC_COVARIATES].values
print("first visits missing snr_total:", int(first_table["snr_total"].isna().sum()))

runs = {"both visits (paper), t-test": fsvv.group_variability_tests(table, is_pd, is_hc),
        "first visit": fsvv.group_variability_models(first_table, is_pd1, is_hc1),
        "first visit + covariates": fsvv.group_variability_models(first_table, is_pd1, is_hc1, covar=fsvv.QC_COVARIATES)}
# Check: without covariates the model is the pooled two-sample t-test (the main analysis above)
print("max |t(model) - t(t-test)|, first visit:", float((runs["first visit"]["t"] - group_tests["t"]).abs().max()))
runs["first visit + covariates"].to_csv(fsvv.out("pd_vs_hc_variability_qc_covariates.csv"), index=False)

sig = {name: fsvv.significant_sets(r) for name, r in runs.items()}
summary = pd.DataFrame({name: {k: len(v) for k, v in s.items()} for name, s in sig.items()})
summary.index.names = ["pair", "kind"]
ref = sig["first visit"]
summary["kept (cov ∩ first) / first"] = [f"{len(sig['first visit + covariates'][k] & ref[k])}/{len(ref[k])}" for k in summary.index]
summary.unstack("kind")

# %% [markdown]
# ## Segmentation overlap between versions (`mri_seg_overlap`)

# %%
seg_dice = fsvv.read_seg_dice(table.index)
dice_summary = pd.DataFrame({"mean": seg_dice.mean(), "sd": seg_dice.std()}).rename_axis("metric").reset_index()
dice_summary.to_csv(fsvv.out("dice_results.csv"), index=False)
ref = pd.read_csv(fsvv.DATA / "tables" / "dice_results.csv").set_index("metric")
delta = (dice_summary.set_index("metric")["mean"] - ref["mean"]).abs()
print(f"{(delta < 1e-9).sum()}/{len(delta)} labels identical to the published dice_results.csv; changed "
      "(published value was overwritten by the aparc segmentations, now aseg):")
display(pd.DataFrame({"published": ref["mean"], "corrected": dice_summary.set_index("metric")["mean"]})[delta >= 1e-9].round(3))
dice_long = fsvv.seg_dice_long(seg_dice)
dice_long.groupby("comparison")["dice"].describe()[["mean", "std", "min", "50%"]]

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 16))
for ax, hemi in zip(axes, fsvv.HEMIS):
    fsvv.plot_seg_dice(dice_long, hemi, ax=ax)
fig.tight_layout()
fsvv.savefig(fig, "segmentation_dice")

# %% [markdown]
# ## Figures: estimates per version

# %%
fig, ax = plt.subplots(figsize=(12, 4))
fsvv.plot_estimates(table, "aseg", ax=ax)
fsvv.savefig(fig, "estimates_subcortical")

fig, axes = plt.subplots(6, 1, figsize=(18, 24))
for row, (kind, hemi) in enumerate([(k, h) for k in fsvv.CORTICAL_MEASURES for h in fsvv.HEMIS]):
    fsvv.plot_estimates(table, kind, hemi, ax=axes[row])
fig.tight_layout()
fsvv.savefig(fig, "estimates_cortical")

# %% [markdown]
# ## Figures: between-version vs between-subject variability (Figures 1–3)
# Boxes: absolute difference between two versions for the same image. Black markers: SD across subjects
# for each version.

# %%
fig, ax = plt.subplots(figsize=(7, 10))
fsvv.plot_variability(table, "aseg", ax=ax, xlim=(-200, 14000))
fsvv.savefig(fig, "variability_subcortical")

# %%
for kind in fsvv.CORTICAL_MEASURES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 18), sharex=True)
    for ax, hemi in zip(axes, fsvv.HEMIS):
        fsvv.plot_variability(table, kind, hemi, ax=ax)
    fig.tight_layout()
    fsvv.savefig(fig, f"variability_{kind}")

# %% [markdown]
# ## Save the full table (measures, version differences, segmentation Dice, group)

# %%
full = pd.concat([table, seg_dice], axis=1).reset_index()
full.to_csv(fsvv.out("aseg_table4.csv"), index=False)
print(full.shape)
