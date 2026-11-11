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
# # 03 · Impact of FreeSurfer versions on clinical analyses (PPMI)
#
# Longitudinal stream, clinical cohort from `01_cohort` (`stage_two_clinical.csv`): PD-non-MCI with
# UPDRS-III OFF at both visits, and HC.
#
# * Subcortical volumes: partial correlation with UPDRS-III in PD-non-MCI (baseline volume, % change),
#   ANCOVA HC vs PD-non-MCI (baseline volume, % change). Covariates: age, sex (+ time between visits).
# * Vertex-wise cortical thickness: clusters from `mri_glmfit-sim` (cluster-wise p < .05), all versions.
#
# Tables 2–3 report raw p-values; `*` = significant after Bonferroni over the 14 subcortical
# regions (p < 0.05/14). p-values are never multiplied.

# %%
import matplotlib.pyplot as plt
import pandas as pd

import fsvv

plt.rcParams.update({"figure.dpi": 80, "font.size": 8})
pd.set_option("display.max_rows", 80)

# %% [markdown]
# ## Subcortical volumes (longitudinal stream)

# %%
stage_two = pd.read_csv(fsvv.inp("stage_two_clinical.csv"))
clin, dropped = fsvv.clinical_volumes(stage_two)
print(f"{len(clin)} subjects with all 4 versions at both visits; {len(dropped)} dropped (missing longitudinal output)")
clin["dx_group"].value_counts()

# %%
print("Dropped (first visit -> second visit):")
dropped.head(10)

# %% [markdown]
# ## Statistics

# %%
corr = fsvv.clinical_partial_correlations(clin)
anc = fsvv.clinical_ancova(clin)
corr.to_csv(fsvv.out("clinical_partial_corr_updrs.csv"), index=False)
anc.to_csv(fsvv.out("clinical_ancova_hc_pdnonmci.csv"), index=False)

# %% [markdown]
# ### Check: FS5–7 only reproduces the submitted Table 2
# With FS8, one PD-non-MCI participant has no longitudinal FS8 output and is dropped (124 PD-non-MCI).
# Requiring only FS5–7 gives the submitted sample (125 PD-non-MCI, 106 HC) and its values
# (e.g. ANCOVA, left thalamus, baseline: F = 15.696 / 15.582 / 15.627).
#
# Note: `results_*_cov.csv` in `data/tables` (original FS8 run) used a different 455-row
# `stage_two_clinical.csv` (147 HC, 308 PD-non-MCI), not the clinical cohort of the paper.

# %%
fs567 = ["FS5", "FS6", "FS7"]
clin567, _ = fsvv.clinical_volumes(stage_two, versions=fs567)
print(len(clin567), "subjects:", clin567["dx_group"].value_counts().to_dict())
fsvv.clinical_ancova(clin567, versions=fs567, rois=["Left-Thalamus"]).query("time == 'base'").round(3)

# %% [markdown]
# ### Table 2 — Partial correlation between UPDRS-III and subcortical volumes

# %%
for time, title in [("base", "Baseline volume vs UPDRS-III"), ("long", "% change in volume vs change in UPDRS-III")]:
    print(title)
    display(fsvv.clinical_table(corr, "r", time))

# %% [markdown]
# ### Table 3 — Group differences (ANCOVA, HC vs PD-non-MCI)

# %%
for time, title in [("base", "Baseline volume"), ("long", "% change in volume")]:
    print(title)
    display(fsvv.clinical_table(anc, "F", time))


# %% [markdown]
# ### Agreement between versions of the significant results (Sørensen–Dice)

# %%
def dice_versions(results, time, alpha=fsvv.ALPHA / fsvv.N_SUBCORTICAL):
    r = results[(results["time"] == time) & results["roi"].isin(fsvv.CLINICAL_TABLE_ROIS)]
    sig = {v: set(g.loc[g["p"] < alpha, "roi"]) for v, g in r.groupby("version")}
    m = pd.DataFrame(index=fsvv.VERSIONS, columns=fsvv.VERSIONS, dtype=float)
    for a in fsvv.VERSIONS:
        for b in fsvv.VERSIONS:
            n = len(sig[a]) + len(sig[b])
            m.loc[a, b] = 2 * len(sig[a] & sig[b]) / n if n else float("nan")
    return m, {v: sorted(s) for v, s in sig.items()}


for name, res in [("ANCOVA", anc), ("partial correlation", corr)]:
    for time in ["base", "long"]:
        m, sig = dice_versions(res, time)
        print(f"{name}, {time}: significant regions per version {sig}")
        display(m.round(2))

# %% [markdown]
# ## Sensitivity: image quality and scanner covariates (reviewers)
# Same models with three more covariates: MRIQC `snr_total` and the scanner manufacturer (GE, Philips; Siemens =
# reference) of the first visit (`fsvv.QC_COVARIATES`). PPMI site isn't in the study files, so the manufacturer
# stands in for it. Participants without MRIQC output are left out.

# %%
from scipy.stats import chi2_contingency

download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
clin_qc = clin.join(fsvv.qc_covariates(download), on="PATNO").dropna(subset=fsvv.QC_COVARIATES)
maker = clin_qc["scanner_GE"].map({1: "GE"}).fillna(clin_qc["scanner_Philips"].map({1: "Philips"})).fillna("Siemens")
counts = pd.crosstab(clin_qc["dx_group"], maker)
chi2, p, _, _ = chi2_contingency(counts)
print(f"{len(clin_qc)} of {len(clin)} participants with QC covariates; manufacturer vs group: chi2 = {chi2:.2f}, p = {p:.3g}")
display(counts)
print("snr_total by group:", clin_qc.groupby("dx_group")["snr_total"].agg(["mean", "std"]).round(2).to_dict("index"))

corr_qc = fsvv.clinical_partial_correlations(clin_qc, extra_covar=fsvv.QC_COVARIATES)
anc_qc = fsvv.clinical_ancova(clin_qc, extra_covar=fsvv.QC_COVARIATES)
corr_qc.to_csv(fsvv.out("clinical_partial_corr_updrs_qc_covariates.csv"), index=False)
anc_qc.to_csv(fsvv.out("clinical_ancova_hc_pdnonmci_qc_covariates.csv"), index=False)

# %%
for name, res, stat in [("ANCOVA", anc_qc, "F"), ("partial correlation", corr_qc, "r")]:
    for time in ["base", "long"]:
        print(f"{name} with QC covariates, {time}")
        display(fsvv.clinical_table(res, stat, time))

# %%
# Significant regions without / with the QC covariates
rows = []
for name, main, qc in [("ANCOVA", anc, anc_qc), ("partial correlation", corr, corr_qc)]:
    for time in ["base", "long"]:
        _, sig_main = dice_versions(main, time)
        _, sig_qc = dice_versions(qc, time)
        for v in fsvv.VERSIONS:
            a, b = set(sig_main[v]), set(sig_qc[v])
            rows.append((name, time, v, ", ".join(sorted(a)), ", ".join(sorted(b)),
                         2 * len(a & b) / (len(a) + len(b)) if a or b else float("nan")))
pd.DataFrame(rows, columns=["analysis", "time", "version", "age/sex model", "+ QC covariates", "Dice"])

# %% [markdown]
# ## Vertex-wise cortical thickness (whole brain)
# `group`: HC vs PD-non-MCI; `corr`: correlation with UPDRS-III in PD-non-MCI; `base`: baseline thickness;
# `long`: rate of change (two-stage model). Rows = significant clusters (none = no row).

# %%
clusters = fsvv.read_cluster_summaries()
clusters.to_csv(fsvv.out("vertexwise_clusters.csv"), index=False)
counts = clusters.groupby(["analysis", "time", "hemi", "version"]).size().unstack("version").reindex(
    columns=fsvv.VERSIONS).fillna(0).astype(int)
counts

# %%
clusters[["version", "analysis", "time", "hemi", "ClusterNo", "Max", "Size(mm^2)", "CWP", "NVtxs", "Annot"]]
