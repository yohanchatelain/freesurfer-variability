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
# # 05 · Desikan-Killiany atlas (R3 major 4)
#
# The analyses of `02` (version differences) and a region-based version of `03` (clinical) with the DK atlas
# (`?h.aparc.stats`, 34 regions per hemisphere, already computed by every version, so nothing is re-run).
#
# 1. Version differences per hemisphere (68 regions) and bilateral (34 regions: mean of lh and rh for volume and
#    thickness, sum for surface area), grouped by lobe
# 2. Comparison with Destrieux (share of regions that differ)
# 3. PD vs HC differences in software variability
# 4. Clinical analyses on bilateral DK thickness, volume and area (longitudinal stream): ANCOVA HC vs PD-non-MCI and
#    partial correlation with UPDRS-III, baseline and % change, per version
#
# Bonferroni by dividing alpha: version tests p < 0.05 / n (lh/rh: 39 + 68 volumes, 68 surfaces, 68 thicknesses;
# bilateral: 39 + 34, 34, 34); clinical tests p < 0.05 / 34 per measure.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import fsvv
from fsvv import PAIRS, VERSIONS, label, prefix, tag

plt.rcParams.update({"figure.dpi": 80, "savefig.dpi": 200, "font.size": 8})
pd.set_option("display.max_rows", 80)
fsvv.set_atlas("dk")

# %% [markdown]
# ## Subjects
# Same images as `02` (cross-sectional cohort of `01_cohort`, processed in every version, minus the visual-QC exclusions).

# %%
EXCLUDED = {s.strip() for s in open(fsvv.inp("excluded_images.txt")) if s.strip()}
download = pd.read_csv(fsvv.inp("download_data.csv"), index_col=0)
download["PATNO_id"] = "sub-" + download["PATNO"].astype(str) + "_ses-" + download["EVENT_ID"]
not_processed = set(pd.read_csv(fsvv.inp("not_processed.csv"))["PATNO"])
subjects = sorted(set(download.loc[~download["PATNO"].isin(not_processed), "PATNO_id"]) - EXCLUDED)
dx_group = download.drop_duplicates("PATNO_id").set_index("PATNO_id")["dx_group"]
print(len(subjects), "images;", dx_group.reindex(subjects).value_counts(dropna=False).to_dict())

# %% [markdown]
# ## DK measures and between-version differences

# %%
measures = fsvv.extract_measures(subjects)
measures = pd.concat([measures, fsvv.bilateral_measures(measures)], axis=1)
print(measures.shape, "| missing values per version:",
      {v: int(measures.filter(regex=f"_{v}$").isna().sum().sum()) for v in VERSIONS})

tables = {}
for hemis in [("lh", "rh"), ("bi",)]:
    fsvv.set_atlas("dk", hemis)
    t = pd.concat([measures, fsvv.version_differences(measures)], axis=1)
    t.insert(0, "dx_group", dx_group.reindex(t.index))
    tables[hemis] = t
table, table_bi = tables[("lh", "rh")], tables[("bi",)]


# %% [markdown]
# ## Paired t-tests between versions
# As in `02`, the aseg labels FS8 doesn't produce (vessel, choroid plexus, optic chiasm, 5th ventricle) are kept; they
# are significant in every FS8 pair by construction and marked `absent_in_FS8` in the CSV files.

# %%
def lobe_of(key):
    for roi in sorted(fsvv.DK_ROIS, key=len, reverse=True):
        if f"_{roi}_" in f"_{key}_":
            return fsvv.DK_LOBE[roi]
    return "subcortical"


absent_fs8 = fsvv.absent_labels(table, "FS8")
print("labels FS8 doesn't produce:", absent_fs8)
results = {}
for hemis, name in [(("lh", "rh"), "lh/rh"), (("bi",), "bilateral")]:
    fsvv.set_atlas("dk", hemis)
    tests = fsvv.paired_version_tests(tables[hemis])
    tests["lobe"] = tests["key"].map(lobe_of)
    tests["absent_in_FS8"] = tests["key"].isin(absent_fs8) & tests["pair"].str.startswith("8")
    tests["significant"] = tests["p"] < fsvv.ALPHA / tests["kind"].map(fsvv.N_TESTS)
    sets = fsvv.significant_sets(tests)
    results[name] = (tests, sets)
    tests.to_csv(fsvv.out(f"dk_version_paired_ttests_{'bilateral' if hemis == ('bi',) else 'hemi'}.csv"), index=False)
    print(f"DK {name}: % of regions with a significant difference (n tests: {fsvv.N_TESTS})")
    display((fsvv.proportion_significant(sets) * 100).round(1))

# %% [markdown]
# ### By lobe (bilateral)
# % of the regions of each lobe with a significant version difference (cortical measures only).

# %%
tests_bi = results["bilateral"][0]
by_lobe = (tests_bi[tests_bi["lobe"] != "subcortical"]
           .assign(pair=lambda d: d["pair"].map({tag(p): label(p) for p in PAIRS}),
                   measure=lambda d: d["kind"].map(fsvv.MEASURE_NAMES))
           .groupby(["measure", "pair", "lobe"])["significant"].mean().mul(100).round(0)
           .unstack("lobe")[list(fsvv.DK_LOBES)])
by_lobe.to_csv(fsvv.out("dk_significant_by_lobe.csv"))
by_lobe

# %%
# Effect size of the version differences per region (bilateral, mean relative difference in %), sorted by lobe
order = [r for lobe in fsvv.DK_LOBES for r in fsvv.DK_LOBES[lobe]]
fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
    m = pd.DataFrame({label(p): [table_bi[f"{prefix(p)}_pct_bi_{r}_{kind}"].mean() * 100 for r in order]
                      for p in PAIRS}, index=[f"{r} ({fsvv.DK_LOBE[r][:4]})" for r in order])
    sns.heatmap(m, annot=True, fmt=".1f", cmap="magma_r", ax=ax, cbar=False, annot_kws={"size": 6})
    ax.set_title(f"{fsvv.MEASURE_NAMES[kind]}: mean |relative difference| (%)")
fig.tight_layout()
fsvv.savefig(fig, "dk_relative_difference_bilateral")

# %% [markdown]
# ## Comparison with Destrieux
# Share of cortical regions with a significant version difference, per atlas (lh/rh tests; Destrieux from `02`,
# `outputs/version_paired_ttests.csv`).

# %%
fsvv.set_atlas("destrieux")
destrieux = pd.read_csv(fsvv.inp("version_paired_ttests.csv"), dtype={"pair": str})
prop_destrieux = fsvv.proportion_significant(fsvv.significant_sets(destrieux))
fsvv.set_atlas("dk")
prop_dk = fsvv.proportion_significant(results["lh/rh"][1])
pd.concat({"Destrieux": prop_destrieux, "DK": prop_dk}, axis=1).mul(100).round(1)

# %% [markdown]
# ## Agreement of the sets of significant regions between comparisons (Sørensen–Dice, DK lh/rh)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
    m = fsvv.dice_between_sets(results["lh/rh"][1], kind)
    sns.heatmap(m, annot=True, fmt=".2f", vmin=0, vmax=1, square=True, cmap="viridis", ax=ax,
                cbar=ax is axes[-1], annot_kws={"size": 7})
    ax.set_title(f"DK, {fsvv.MEASURE_NAMES[kind]}")
fig.tight_layout()
fsvv.savefig(fig, "dk_dice_significant_sets")

# %% [markdown]
# ## PD vs HC differences in software variability (two-sample t-tests of the relative differences)
# One image per participant (first visit), as in `02`.

# %%
is_pd = table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])
is_hc = table["dx_group"] == "HC"
group_tests = {}
for hemis, name, t in [(("lh", "rh"), "lh/rh", table), (("bi",), "bilateral", table_bi)]:
    fsvv.set_atlas("dk", hemis)
    t1 = fsvv.first_visits(t, download)
    g = fsvv.group_variability_tests(t1, is_pd.loc[t1.index], is_hc.loc[t1.index])
    g["significant"] = g["p"] < fsvv.ALPHA / g["kind"].map(fsvv.N_TESTS)
    group_tests[name] = g
    g.to_csv(fsvv.out(f"dk_pd_vs_hc_variability_ttests_{'bilateral' if hemis == ('bi',) else 'hemi'}.csv"), index=False)
    print(f"{name}: {len(t1)} first visits")
    display(g.groupby(["pair", "kind"]).agg(min_p=("p", "min"), n_significant=("significant", "sum")).unstack("kind"))

# %% [markdown]
# ## Figures: between-version vs between-subject variability (bilateral DK)
# Boxes: absolute difference between two versions for the same image; black markers: SD across subjects per version.

# %%
fsvv.set_atlas("dk", ("bi",))
fig, axes = plt.subplots(1, 3, figsize=(20, 9))
for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
    fsvv.plot_variability(table_bi, kind, "bi", ax=ax)
fig.tight_layout()
fsvv.savefig(fig, "dk_variability_bilateral")

# %% [markdown]
# ## Clinical analyses on DK regions (longitudinal stream, bilateral)
# Same cohort and models as the subcortical analyses of `03`: ANCOVA HC vs PD-non-MCI and partial correlation with
# UPDRS-III in PD-non-MCI; covariates age, sex (+ time between visits for the change). Surface area and volume also
# adjust for head size: the eTIV of the same version (first visit, `TIV_{v}_first`); thickness doesn't (it scales little
# with head size). Multiple comparisons over the 34 bilateral regions of each analysis × measure × version: Bonferroni
# (main, p < 0.05/34) and Benjamini–Hochberg FDR (q < 0.05) for comparison.

# %%
fsvv.set_atlas("dk", ("bi",))
stage_two = pd.read_csv(fsvv.inp("stage_two_clinical.csv"))
stems = [f"{r}_bi_{k}" for k in fsvv.CORTICAL_MEASURES for r in fsvv.DK_ROIS]
clin, dropped = fsvv.clinical_volumes(stage_two, rois=stems, aparc=True)
print(f"{len(clin)} participants with all 4 versions at both visits; {len(dropped)} dropped")
print(clin["dx_group"].value_counts().to_dict())
# eTIV of each version for area and volume; thickness without
size = [s for s in stems if not s.endswith("_ct")]
thick = [s for s in stems if s.endswith("_ct")]
corr = pd.concat([fsvv.clinical_partial_correlations(clin, rois=size, extra_covar=["TIV_{v}_first"]),
                  fsvv.clinical_partial_correlations(clin, rois=thick)], ignore_index=True)
anc = pd.concat([fsvv.clinical_ancova(clin, rois=size, extra_covar=["TIV_{v}_first"]),
                 fsvv.clinical_ancova(clin, rois=thick)], ignore_index=True)
for res in (corr, anc):
    res["kind"] = res["roi"].str.rsplit("_", n=1).str[1]
    res["region"] = res["roi"].str.split("_bi_").str[0]
corr.to_csv(fsvv.out("dk_clinical_partial_corr_updrs.csv"), index=False)
anc.to_csv(fsvv.out("dk_clinical_ancova_hc_pdnonmci.csv"), index=False)

# %%
from scipy.stats import false_discovery_control

n_dk = len(fsvv.DK_ROIS)
summary = []
for name, res in [("ANCOVA HC vs PD-non-MCI", anc), ("partial correlation with UPDRS-III", corr)]:
    for (time, kind, v), g in res.groupby(["time", "kind", "version"]):
        sig = sorted(g.loc[g["p"] < fsvv.ALPHA / n_dk, "region"])
        fdr = sorted(g.loc[false_discovery_control(g["p"]) < fsvv.ALPHA, "region"])
        summary.append((name, time, fsvv.MEASURE_NAMES[kind], v, int((g["p"] < fsvv.ALPHA).sum()), len(sig),
                        ", ".join(sig), len(fdr), ", ".join(r for r in fdr if r not in sig)))
summary = pd.DataFrame(summary, columns=["analysis", "time", "measure", "version", "n p<.05 (uncorrected)",
                                         "n Bonferroni", "Bonferroni-significant regions", "n FDR",
                                         "FDR-only regions"])
summary.to_csv(fsvv.out("dk_clinical_summary.csv"), index=False)
summary

# %%
# Thickness tables (paper format: raw p, '*' after Bonferroni over 34 regions)
for res, stat, name in [(anc, "F", "ANCOVA"), (corr, "r", "partial correlation")]:
    for time in ["base", "long"]:
        print(f"{name}, thickness, {time}")
        display(fsvv.clinical_table(res, stat, time, rois=[f"{r}_bi_ct" for r in fsvv.DK_ROIS], n_tests=n_dk))

# %%
# Agreement between versions of the Bonferroni-significant regions (Sørensen–Dice), per analysis and measure
rows = []
for name, res in [("ANCOVA", anc), ("partial correlation", corr)]:
    for (time, kind), g in res.groupby(["time", "kind"]):
        sig = {v: set(gv.loc[gv["p"] < fsvv.ALPHA / n_dk, "region"]) for v, gv in g.groupby("version")}
        for a, b in [(7, 6), (7, 5), (6, 5), (8, 7), (8, 6), (8, 5)]:
            sa, sb = sig[f"FS{a}"], sig[f"FS{b}"]
            rows.append((name, time, fsvv.MEASURE_NAMES[kind], f"FS {a} vs {b}",
                         2 * len(sa & sb) / (len(sa) + len(sb)) if sa or sb else np.nan))
pd.DataFrame(rows, columns=["analysis", "time", "measure", "pair", "dice"]).pivot_table(
    index=["analysis", "time", "measure"], columns="pair", values="dice").round(2)
