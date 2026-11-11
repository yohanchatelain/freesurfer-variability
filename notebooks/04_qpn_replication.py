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
# # 04 · Replication sample (QPN)
#
# The paper's replication analyses (Methods, "Replication analyses were run with the QPN cohort") on the Quebec
# Parkinson Network (QPN) release `qpn-nc-r01`: one session per participant, a single site (Siemens Prisma Fit 3T).
# Processed for the revision in FS5, FS6, FS7 and FS8 with the paper's cross-sectional pipeline (`scripts/qpn/`,
# `recon-all -all -qcache`).
#
# **Main analyses = the paper's, on QPN PD** (the paper's QPN cohort was PD only: 160 PD from an older snapshot;
# this release has 201 PD with a usable T1w):
# 1. paired t-tests between versions (Table S6);
# 2. software variability, QPN PD vs PPMI PD (Methods: "The group differences between the PPMI PD group and the QPN
#    cohort in software variability were also tested");
# 3. MRIQC: image quality QPN PD vs PPMI PD, and correlation of the MRIQC metrics with software variability in QPN PD
#    (Tables S29–S38).
#
# PPMI side: first visit of each participant (one image each, as in `02`), PD = PD-non-MCI and PD-MCI.
#
# **Additional analyses** (end of the notebook; not in the submitted paper): the QPN release also has 69 HC, so the
# version tests on all 270 participants and QPN PD vs QPN HC; and QPN PD vs PPMI HC, which is what the submitted
# code ran for the variability test (its text says PPMI PD).
#
# Inputs (participant-level, gitignored): `data/qpn/QPN_subjects.csv` (PATNO, sex, age, dx_group),
# `data/qpn/stats/FSx/<subject>/stats/`, `data/qpn/mriqc/output` (MRIQC 22.0.1, paper's container,
# `scripts/qpn/make_mriqc_bids.py`); PPMI: `aseg_table4.csv`, `version_paired_ttests.csv` and `download_data.csv` of
# `01`/`02`. Outputs: aggregate CSVs `outputs/qpn_*.csv`.

# %%
import json

import pandas as pd
from scipy.stats import ttest_ind

import fsvv

QPN_DIR = fsvv.DATA / "qpn"
MRIQC_DIR = QPN_DIR / "mriqc" / "output"
if not (QPN_DIR / "QPN_subjects.csv").exists():
    raise FileNotFoundError(f"QPN inputs missing: link them as {QPN_DIR} (see TODO.md)")

QPN_VERSIONS = [v for v in fsvv.VERSIONS if (QPN_DIR / "stats" / v).is_dir()]
QPN_PAIRS = [p for p in fsvv.PAIRS if set(p) <= set(QPN_VERSIONS)]
print("QPN versions:", QPN_VERSIONS)

qpn = pd.read_csv(QPN_DIR / "QPN_subjects.csv")[["PATNO", "sex", "age", "dx_group"]].set_index("PATNO")
demo = qpn.groupby("dx_group").agg(n=("age", "size"), women=("sex", lambda s: (s == "F").sum()),
                                   age_mean=("age", "mean"), age_sd=("age", "std"))
demo.to_csv(fsvv.out("qpn_demographics.csv"))
demo.round(1)

# %% [markdown]
# ## Measures and version differences (all QPN participants; the main analyses use the PD rows)

# %%
measures = fsvv.extract_measures(qpn.index, QPN_VERSIONS, root=QPN_DIR / "stats")
qpn_table = pd.concat([measures, fsvv.version_differences(measures, QPN_PAIRS)], axis=1)
qpn_table.insert(0, "dx_group", qpn["dx_group"].reindex(qpn_table.index))
print(qpn_table.shape, "| missing values per version:",
      {v: int(measures.filter(regex=f"_{v}$").isna().sum().sum()) for v in QPN_VERSIONS})
is_pd, is_hc = qpn_table["dx_group"] == "PD", qpn_table["dx_group"] == "HC"
qpn_pd = qpn_table.loc[is_pd]
print(len(qpn_pd), "QPN PD,", int(is_hc.sum()), "QPN HC")

# PPMI first visits (one image per participant)
download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
download["PATNO_id"] = "sub-" + download["PATNO"].astype(str) + "_ses-" + download["EVENT_ID"]
ppmi = fsvv.first_visits(pd.read_csv(fsvv.inp("aseg_table4.csv")).set_index("PATNO_id"), download)
ppmi_groups = {"PPMI-PD": ppmi["dx_group"].isin(["PD-MCI", "PD-non-MCI"]), "PPMI-HC": ppmi["dx_group"] == "HC"}
print(len(ppmi), "PPMI first visits:", {g: int(m.sum()) for g, m in ppmi_groups.items()})


def version_tests(table):
    """Paired version t-tests, with the labels FS8 doesn't produce flagged (significant by construction, as in 02)."""
    t = fsvv.paired_version_tests(table, QPN_PAIRS)
    t["absent_in_FS8"] = t["key"].isin(absent_fs8) & t["pair"].str.startswith("8")
    return t


def qpn_vs_ppmi(ppmi_group):
    """Two-sample t-tests of the relative version differences, QPN PD vs a PPMI group (first visits)."""
    cols = qpn_table.columns.intersection(ppmi.columns)
    both = pd.concat([qpn_pd[cols], ppmi.loc[ppmi_groups[ppmi_group], cols]])
    in_qpn = pd.Series(both.index.isin(qpn_pd.index), index=both.index)
    return fsvv.group_variability_tests(both, in_qpn, ~in_qpn, QPN_PAIRS).assign(comparison=f"QPN-PD vs {ppmi_group}")


def n_significant(tests, by=("pair", "kind")):
    sig = tests["p"] < fsvv.ALPHA / tests["kind"].map(fsvv.N_TESTS)
    return tests.assign(significant=sig).groupby(list(by)).agg(min_p=("p", "min"), n_significant=("significant", "sum"))


absent_fs8 = fsvv.absent_labels(qpn_table, "FS8") if "FS8" in QPN_VERSIONS else []
print("labels FS8 doesn't produce:", absent_fs8)

# %% [markdown]
# ## 1. Paired t-tests between versions, QPN PD (Table S6)

# %%
tests_pd = version_tests(qpn_pd)
tests_pd.to_csv(fsvv.out("qpn_version_paired_ttests.csv"), index=False)
sets_pd = fsvv.significant_sets(tests_pd)
prop_pd = fsvv.proportion_significant(sets_pd, QPN_PAIRS).mul(100)
prop_pd.to_csv(fsvv.out("qpn_proportion_significant.csv"))
prop_pd.round(1)  # % of regions with a significant difference

# %% [markdown]
# Share of significant regions by lobe (the Results paragraph of the replication sample).

# %%
by_lobe = fsvv.proportion_by_lobe(tests_pd, QPN_PAIRS)
by_lobe.to_csv(fsvv.out("qpn_proportion_by_lobe.csv"))
by_lobe.round(0)

# %% [markdown]
# Agreement of the sets of significant regions between version pairs (Sørensen–Dice).

# %%
for kind in fsvv.CORTICAL_MEASURES:
    print(fsvv.MEASURE_NAMES[kind])
    display(fsvv.dice_between_sets(sets_pd, kind, QPN_PAIRS).round(2))

# %% [markdown]
# Agreement between the samples: Sørensen–Dice of the sets of significant regions of the same version pair in QPN PD
# and in PPMI (`version_paired_ttests.csv` of `02`, the paper's main sample).

# %%
ppmi_tests = pd.read_csv(fsvv.inp("version_paired_ttests.csv"), dtype={"pair": str})  # tags like "76", not ints
ppmi_sets = fsvv.significant_sets(ppmi_tests)
rows = []
for pair in QPN_PAIRS:
    for kind in fsvv.CORTICAL_MEASURES:
        a, b = sets_pd[(fsvv.tag(pair), kind)], ppmi_sets[(fsvv.tag(pair), kind)]
        rows.append((fsvv.label(pair), fsvv.MEASURE_NAMES[kind], len(a), len(b),
                     2 * len(a & b) / (len(a) + len(b)) if a or b else float("nan")))
ppmi_vs_qpn = pd.DataFrame(rows, columns=["pair", "measure", "n_sig_QPN", "n_sig_PPMI", "dice"])
ppmi_vs_qpn.to_csv(fsvv.out("qpn_ppmi_significant_sets_dice.csv"), index=False)
ppmi_vs_qpn.pivot(index="pair", columns="measure", values="dice").round(2)

# %% [markdown]
# ## 2. Software variability: QPN PD vs PPMI PD (two-sample t-tests of the relative differences)

# %%
pd_vs_ppmi_pd = qpn_vs_ppmi("PPMI-PD")
pd_vs_ppmi_pd.to_csv(fsvv.out("qpn_pd_vs_ppmi_pd_variability_ttests.csv"), index=False)
n_significant(pd_vs_ppmi_pd).unstack("kind")

# %% [markdown]
# ## 3. MRIQC
# ### Image quality: QPN PD vs PPMI PD

# %%
iqm = {}
for s in qpn.index:
    f = MRIQC_DIR / s / "ses-01" / "anat" / f"{s}_T1w.json"
    if f.exists():
        d = json.load(open(f))
        iqm[s] = {m: d[m] for m in fsvv.MRIQC_METRICS}
qc = pd.DataFrame.from_dict(iqm, orient="index").reindex(qpn.index)
print(f"MRIQC metrics for {qc['cjv'].notna().sum()}/{len(qc)} QPN participants")
qpn_qc = qpn_table.join(qc)

ppmi_qc = {}
for patno, event, dx in download.drop_duplicates("PATNO")[["PATNO", "EVENT_ID", "dx_group"]].itertuples(index=False):
    f = fsvv.mriqc_file(patno, event)
    if f"sub-{patno}_ses-{event}" in ppmi.index and f.exists():
        d = json.load(open(f))
        ppmi_qc[f"sub-{patno}_ses-{event}"] = {"dx_group": dx, **{m: d[m] for m in fsvv.MRIQC_METRICS}}
ppmi_qc = pd.DataFrame.from_dict(ppmi_qc, orient="index")
qc_groups = {"QPN-PD": qc[is_pd], "QPN-HC": qc[is_hc],
             "PPMI-PD": ppmi_qc[ppmi_qc["dx_group"].isin(["PD-MCI", "PD-non-MCI"])],
             "PPMI-HC": ppmi_qc[ppmi_qc["dx_group"] == "HC"]}


def qc_ttests(a, b):
    rows = []
    for m in fsvv.MRIQC_METRICS:
        t, p = ttest_ind(qc_groups[a][m], qc_groups[b][m], nan_policy="omit")
        rows.append((f"{a} vs {b}", m, qc_groups[a][m].notna().sum(), qc_groups[b][m].notna().sum(),
                     qc_groups[a][m].mean(), qc_groups[b][m].mean(), float(t), float(p)))
    return pd.DataFrame(rows, columns=["comparison", "metric", "n1", "n2", "mean1", "mean2", "t", "p"])


qc_pd = qc_ttests("QPN-PD", "PPMI-PD")
qc_pd.to_csv(fsvv.out("qpn_pd_vs_ppmi_pd_mriqc_ttests.csv"), index=False)
qc_pd.set_index("metric").round(3)

# %% [markdown]
# ### Correlation between MRIQC metrics and software variability, QPN PD (Tables S29–S38)
# Bonferroni over regions and metrics, as in `02`.

# %%
qc_corr = fsvv.qc_correlations(qpn_qc.loc[is_pd], pairs=QPN_PAIRS, replace_inf=True)
qc_corr.to_csv(fsvv.out("qpn_mriqc_variability_correlations.csv"), index=False)
qc_corr["significant"] = qc_corr["p"] < fsvv.ALPHA / (qc_corr["kind"].map(fsvv.N_TESTS) * len(fsvv.MRIQC_METRICS))
qc_corr.groupby(["metric", "kind"]).agg(max_abs_r=("r", lambda r: r.abs().max()),
                                         n_significant=("significant", "sum")).unstack("kind")

# %% [markdown]
# ## Additional analyses (not in the submitted paper)
# ### QPN PD vs PPMI HC: what the submitted code ran for the variability test (its text says PPMI PD)

# %%
pd_vs_ppmi_hc = qpn_vs_ppmi("PPMI-HC")
pd_vs_ppmi_hc.to_csv(fsvv.out("qpn_pd_vs_ppmi_hc_variability_ttests.csv"), index=False)
n_significant(pd_vs_ppmi_hc).unstack("kind")

# %% [markdown]
# ### With the QPN controls: version tests on all 270 participants and in HC, and QPN PD vs QPN HC
# Same scanner for both groups, so this PD vs HC test has no site effect (the replication of `02`'s PPMI PD vs HC).

# %%
tests_extra = {"all": version_tests(qpn_table), "HC": version_tests(qpn_table.loc[is_hc])}
tests_extra["all"].to_csv(fsvv.out("qpn_all_version_paired_ttests.csv"), index=False)
prop_extra = pd.concat({"PD": prop_pd, **{n: fsvv.proportion_significant(fsvv.significant_sets(t), QPN_PAIRS).mul(100)
                                          for n, t in tests_extra.items()}}, axis=1)
prop_extra.to_csv(fsvv.out("qpn_proportion_significant_all_pd_hc.csv"))
prop_extra.round(1)

# %%
pd_vs_hc = fsvv.group_variability_tests(qpn_table, is_pd, is_hc, QPN_PAIRS)
pd_vs_hc.to_csv(fsvv.out("qpn_pd_vs_hc_variability_ttests.csv"), index=False)
n_significant(pd_vs_hc).unstack("kind")

# %%
qc_hc = qc_ttests("QPN-PD", "QPN-HC")
qc_hc.to_csv(fsvv.out("qpn_pd_vs_hc_mriqc_ttests.csv"), index=False)
qc_hc.set_index("metric").round(3)
