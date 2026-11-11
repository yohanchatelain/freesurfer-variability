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
# # 07 · Table 1 and supplementary tables
#
# From the outputs of `01_cohort`, `02_software_variability` and `scripts/glm` (run them first). Everything goes to
# `outputs/tables` (participant-level inputs, aggregate outputs only). Table S6 (QPN) needs `04_qpn_replication`.
#
# | Paper | Output | Content |
# |---|---|---|
# | Table 1 | `table1_descriptive.csv`, `table1_tests.csv` | samples actually analysed; PD vs HC tests quoted in the text |
# | Table S1 | `tableS1_version_ttests_all.csv` | paired t-tests between versions, every image (Figures 1–2) |
# | Tables S2, S3 | `tableS2_version_ttests_PD.csv`, `tableS3_version_ttests_HC.csv` | same, PD (PD-non-MCI + PD-MCI) and HC images |
# | Table 2 | `table2_stage1_by_lobe.csv` | % of regions with a significant version difference, per measure, subcortical and lobe |
# | Table S6 | `tableS6_version_ttests_QPN.csv` | paired t-tests between versions, QPN PD (from `04`) |
# | Table S4 | `tableS4_pd_vs_hc_variability.csv` | PD vs HC, relative version differences, first visits |
# | Table S5 | `tableS5_pd_vs_hc_variability_qc_covariates.csv` | same, adjusted for MRIQC SNR and scanner manufacturer |
# | Tables S9–S18 | `tableS9-S18_mriqc_corr_all.csv` | MRIQC metrics vs relative version differences, one block per metric |
# | Tables S19–S28 | `tableS19-S28_mriqc_corr_PD.csv` | same, PD images only |
# | Tables S29–S38 | `tableS29-S38_mriqc_corr_QPN_PD.csv` | same, QPN PD (from `04`) |
#
# Numbering of the revision (`paper/latex/supplementary.tex`): S6 = QPN, S7–S8 = Tables 3–6 with QC covariates
# (`scripts/paper/make_latex_tables.py`); 10 MRIQC metrics (`fsvv.MRIQC_METRICS`) instead of the submitted 9. Cells: statistic and raw p; `*` = significant after Bonferroni
# (alpha divided by the number of tests: 187 volume, 148 area, 148 thickness; × the 10 metrics for MRIQC).

# %%
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

import fsvv

pd.set_option("display.max_rows", 60, "display.max_columns", 30)
TABLES = fsvv.OUT / "tables"
TABLES.mkdir(exist_ok=True)

download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
download["PATNO_id"] = "sub-" + download["PATNO"].astype(str) + "_ses-" + download["EVENT_ID"]
table = pd.read_csv(fsvv.OUT / "aseg_table4.csv", index_col="PATNO_id")  # 02: images analysed, measures, differences
print(len(table), "images;", table["dx_group"].value_counts().to_dict())


# %% [markdown]
# ## Table 1 — descriptive statistics
#
# Cross-sectional: participants with at least one image in `02` (demographics at the first visit of the cohort).
# Longitudinal: participants of the vertex-wise GLMs and subcortical analyses (`outputs/glm` FSGD, 275).

# %%
def summary(groups, clinical=False):
    ms = lambda d, c: f"{d[c].mean():.1f} ± {d[c].std():.1f}"
    rows = {
        "n": {g: f"{len(d)}" for g, d in groups.items()},
        "age (y)": {g: ms(d, "AGE_AT_VISIT") for g, d in groups.items()},
        "age range": {g: f"{d['AGE_AT_VISIT'].min():.1f} - {d['AGE_AT_VISIT'].max():.1f}" for g, d in groups.items()},
        "sex (male, %)": {g: f"{(d['SEX'] == 1).sum()}, {(d['SEX'] == 1).mean() * 100:.1f}%" for g, d in groups.items()},
        "education (y)": {g: ms(d, "EDUCYRS") for g, d in groups.items()},
    }
    if clinical:
        rows["duration T2-T1 (y)"] = {g: ms(d, "duration_y") for g, d in groups.items()}
        rows["UPDRS-III OFF baseline"] = {g: ms(d, "NP3TOT") if g != "HC" else "" for g, d in groups.items()}
        rows["UPDRS-III OFF follow-up"] = {g: ms(d, "NP3TOT_NX") if g != "HC" else "" for g, d in groups.items()}
    return pd.DataFrame(rows).T


def tests(pd_group, hc_group, sample, extra=()):
    rows = []
    for c in ["AGE_AT_VISIT", "EDUCYRS", *extra]:
        t, p = ttest_ind(pd_group[c], hc_group[c], nan_policy="omit")
        rows.append((sample, c, "t", t, p))
    obs = [[(g["SEX"] == s).sum() for s in (0, 1)] for g in (pd_group, hc_group)]
    chi2, p, _, _ = chi2_contingency(obs)
    rows.append((sample, "SEX", "chi2", chi2, p))
    return pd.DataFrame(rows, columns=["sample", "variable", "test", "stat", "p"])


cross = download[download["PATNO_id"].isin(table.index)].drop_duplicates("PATNO")
cross_groups = {g: cross[cross["dx_group"] == g] for g in ["HC", "PD-non-MCI", "PD-MCI"]}

fsgd = fsvv.OUT / "glm" / "fsgd" / "fsgd_cort_group_HC_PDnonMCI_long.fsgd"
glm_patno = {int(l.split("\t")[1][4:-5]) for l in open(fsgd) if l.startswith("Input")}  # sub-<PATNO>_base
stage_two = pd.read_csv(fsvv.OUT / "stage_two_clinical.csv")
long_ = stage_two[stage_two["PATNO"].isin(glm_patno)].copy()
# visit interval as in the GLMs (make_glm_inputs.py): average months / 12
long_["duration_y"] = (pd.to_datetime(long_["Study Date_NX"]) - pd.to_datetime(long_["Study Date"])).dt.total_seconds() / 2629746 / 12
long_groups = {g: long_[long_["dx_group"] == g] for g in ["HC", "PD-non-MCI"]}
assert len(long_) == len(glm_patno), (len(long_), len(glm_patno))

table1 = pd.concat({"Cross-sectional": summary(cross_groups), "Longitudinal": summary(long_groups, clinical=True)},
                   axis=1).fillna("")
table1.to_csv(TABLES / "table1_descriptive.csv")
table1_tests = pd.concat([
    tests(pd.concat([cross_groups["PD-non-MCI"], cross_groups["PD-MCI"]]), cross_groups["HC"], "cross-sectional, PD vs HC"),
    tests(long_groups["PD-non-MCI"], long_groups["HC"], "longitudinal, PD-non-MCI vs HC", extra=["duration_y"]),
], ignore_index=True)
table1_tests.to_csv(TABLES / "table1_tests.csv", index=False)
table1

# %%
table1_tests.round(3)

# %% [markdown]
# ## Table 2 — share of regions with a significant difference between versions
#
# Bonferroni as in Figures 1–2; subcortical = the 39 aseg structures (volume only); lobes = `fsvv.DESTRIEUX_LOBES`
# (both hemispheres; the 11 insular and boundary regions per hemisphere count only in "all").

# %%
table2 = fsvv.proportion_by_lobe(pd.read_csv(fsvv.OUT / "version_paired_ttests.csv", dtype={"pair": str}))
table2.round(1).to_csv(TABLES / "table2_stage1_by_lobe.csv")
table2.round(1)

# %% [markdown]
# ## Supplementary tables — paired t-tests between versions (S1–S3)

# %%
KIND_ORDER = {"vol": 0, "surf": 1, "ct": 2}


def region_index(keys, kind):
    """(measure, hemisphere, region) of a test key: aseg keys are volumes without hemisphere."""
    out = []
    for k in keys:
        if kind == "vol" and k in fsvv.ASEG_ROIS:
            out.append(("Volume (subcortical)", "", k))
        else:
            hemi, rest = k.split("_", 1)
            out.append((fsvv.MEASURE_NAMES[kind], hemi, rest[: -len(kind) - 1]))
    return pd.MultiIndex.from_tuples(out, names=["measure", "hemi", "region"])


def wide(tests, stat, n_extra=1):
    """One row per (measure, region), (pair, stat / p) columns; p raw with '*' after Bonferroni."""
    blocks = []
    for kind, g in sorted(tests.groupby("kind"), key=lambda x: KIND_ORDER[x[0]]):
        n_tests = fsvv.N_TESTS[kind] * n_extra
        keys = list(dict.fromkeys(g["key"]))
        d = g.set_index(["key", "pair"])
        cols = {}
        for pair in fsvv.PAIRS:
            s = d.xs(fsvv.tag(pair), level="pair").reindex(keys)
            cols[(fsvv.label(pair), stat)] = s[stat].round(3).values
            cols[(fsvv.label(pair), "p")] = [fsvv.fmt_p(p, n_tests) if pd.notna(p) else "n/a" for p in s["p"]]
        blocks.append(pd.DataFrame(cols, index=region_index(keys, kind)))
    return pd.concat(blocks)


def n_significant(tests, n_extra=1):
    sig = tests["p"] < fsvv.ALPHA / (tests["kind"].map(fsvv.N_TESTS) * n_extra)
    return tests.assign(sig=sig).groupby(["kind", "pair"])["sig"].sum().unstack("pair")[
        [fsvv.tag(p) for p in fsvv.PAIRS]]


all_tests = pd.read_csv(fsvv.OUT / "version_paired_ttests.csv", dtype={"pair": str})
is_pd = table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])
is_hc = table["dx_group"] == "HC"
group_tests = {"PD": fsvv.paired_version_tests(table.loc[is_pd]), "HC": fsvv.paired_version_tests(table.loc[is_hc])}
for name, t in [("tableS1_version_ttests_all", all_tests), ("tableS2_version_ttests_PD", group_tests["PD"]),
                ("tableS3_version_ttests_HC", group_tests["HC"])]:
    wide(t, "t").to_csv(TABLES / f"{name}.csv")
print(f"images: all {len(table)}, PD {is_pd.sum()}, HC {is_hc.sum()}")
pd.concat({"all": n_significant(all_tests), "PD": n_significant(group_tests["PD"]),
           "HC": n_significant(group_tests["HC"])})  # significant regions per pair

# %% [markdown]
# ## Table S6 — paired t-tests between versions, QPN replication sample (PD)
#
# From `04_qpn_replication` (`qpn_version_paired_ttests.csv`: QPN PD, one image per participant, as in the paper).

# %%
qpn_tests = pd.read_csv(fsvv.OUT / "qpn_version_paired_ttests.csv", dtype={"pair": str})
wide(qpn_tests, "t").to_csv(TABLES / "tableS6_version_ttests_QPN.csv")
n_significant(qpn_tests)  # significant regions per pair

# %% [markdown]
# ## Tables S4–S5 — PD vs HC differences in software variability
#
# Two-sample t-tests of the relative version differences, one image per participant (first visit, decision
# 2026-10-26), from `02`. `_qc_covariates`: OLS with MRIQC snr_total and scanner manufacturer.

# %%
s4 = pd.read_csv(fsvv.OUT / "pd_vs_hc_variability_ttests.csv", dtype={"pair": str})
s4_qc = pd.read_csv(fsvv.OUT / "pd_vs_hc_variability_qc_covariates.csv", dtype={"pair": str})
wide(s4, "t").to_csv(TABLES / "tableS4_pd_vs_hc_variability.csv")
wide(s4_qc, "t").to_csv(TABLES / "tableS5_pd_vs_hc_variability_qc_covariates.csv")
pd.concat({"t-test": n_significant(s4), "with QC covariates": n_significant(s4_qc)})

# %% [markdown]
# ## MRIQC metrics vs software variability (all images, then PD images)
#
# Pearson correlation between the MRIQC metrics and the relative version differences of the first-visit images (the
# images MRIQC was run on), as in `02`.

# %%
# as in 02: the metrics of the first visit (the image MRIQC was run on) go to that image only
rows = {}
for patno, event, pid in download.drop_duplicates("PATNO")[["PATNO", "EVENT_ID", "PATNO_id"]].itertuples(index=False):
    f = fsvv.mriqc_file(patno, event)
    if f.exists():
        iqm = json.load(open(f))
        rows[pid] = [iqm[m] for m in fsvv.MRIQC_METRICS]
qc_table = table.join(pd.DataFrame.from_dict(rows, orient="index", columns=fsvv.MRIQC_METRICS))
print("images with MRIQC metrics:", qc_table["cjv"].notna().sum(), "of", len(qc_table))

qc_all = pd.read_csv(fsvv.OUT / "mriqc_variability_correlations.csv", dtype={"pair": str})
qc_pd = fsvv.qc_correlations(qc_table.loc[is_pd])
qc_pd.to_csv(fsvv.out("mriqc_variability_correlations_PD.csv"), index=False)
n_metrics = len(fsvv.MRIQC_METRICS)
qc_qpn = pd.read_csv(fsvv.OUT / "qpn_mriqc_variability_correlations.csv", dtype={"pair": str})  # 04, QPN PD
for name, res in [("tableS9-S18_mriqc_corr_all", qc_all), ("tableS19-S28_mriqc_corr_PD", qc_pd),
                  ("tableS29-S38_mriqc_corr_QPN_PD", qc_qpn)]:
    pd.concat({m: wide(g.drop(columns="metric"), "r", n_metrics) for m, g in res.groupby("metric", sort=False)},
              names=["metric"]).to_csv(TABLES / f"{name}.csv")
pd.concat({s: r.assign(sig=r["p"] < fsvv.ALPHA / (r["kind"].map(fsvv.N_TESTS) * n_metrics)).groupby(
    ["metric", "kind"])["sig"].sum().unstack("kind") for s, r in [("all", qc_all), ("PD", qc_pd)]}, axis=1)

# %%
sorted(p.name for p in TABLES.glob("table*.csv"))
