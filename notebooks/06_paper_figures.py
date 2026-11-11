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
# # 06 · Paper figures and tables (6 version pairs, 4 versions)
#
# Paper-ready figures and tables of the revision, from the outputs of `02_software_variability`, `03_clinical`
# (and `outputs/glm`, `scripts/glm`) and `04_qpn_replication` (QPN panel of Figure 1). Run those first.
#
# | Paper | Output (`outputs/figures`, `outputs/tables`) | Source |
# |---|---|---|
# | Figure 1 — version differences, subcortical volumes | `fig1_pvalues_subcortical` | `version_paired_ttests.csv` (02) |
# | Figure 2 — version differences, Destrieux volume / area / thickness | `fig2_pvalues_cortical` | `version_paired_ttests.csv` (02) |
# | Figures 3–6 — between-version vs between-subject variability | `variability_subcortical`, `variability_{vol,surf,ct}` | made by 02 |
# | Table 3 — HC vs PD-non-MCI, subcortical volumes (ANCOVA) | `table3_ancova_subcortical.csv` | `clinical_ancova_hc_pdnonmci.csv` (03) |
# | Table 4 — UPDRS-III vs subcortical volumes (partial correlation) | `table4_corr_subcortical.csv` | `clinical_partial_corr_updrs.csv` (03) |
# | Table 5 — vertex-wise UPDRS-III vs thickness | `table5_vertexwise_corr.csv` | `outputs/glm` |
# | Table 6 — vertex-wise HC vs PD-non-MCI thickness | `table6_vertexwise_group.csv` | `outputs/glm` |
# | Figures 7–10 — vertex-wise maps | `fig7_corr_base`, `fig8_corr_long`, `fig9_group_base`, `fig10_group_long` | `outputs/glm` |
#
# Each `fig*` has a CSV of the values it shows in `outputs/tables`. p-values are raw; significance uses
# Bonferroni by dividing alpha (version tests: 0.05/187 volume, 0.05/148 area and thickness; Tables 3–4: 0.05/14).

# %%
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import plotting

import fsvv

plt.rcParams.update({"figure.dpi": 80, "font.size": 8})
pd.set_option("display.max_rows", 100)
TABLES = fsvv.OUT / "tables"
TABLES.mkdir(exist_ok=True)
PAIR_LABELS = [fsvv.label(p) for p in fsvv.PAIRS]

# %% [markdown]
# ## Figures 1–2 — differences between versions (paired t-tests)
#
# Colours as in the paper (Gronenschild et al., 2012), after Bonferroni correction: p < .05 green, < .001 yellow,
# < .0001 orange, < .00001 red (thresholds divided by the number of tests). × = label FS8 doesn't produce
# (volume 0; significant by construction, see `fsvv.absent_labels`).

# %%
tests = pd.read_csv(fsvv.OUT / "version_paired_ttests.csv", dtype={"pair": str})
P_LEVELS = [0.05, 0.001, 0.0001, 0.00001]
P_COLORS = ["white", "#4daf4a", "#ffff33", "#ff7f00", "#e41a1c"]
P_LABELS = ["n.s.", "p < .05", "p < .001", "p < .0001", "p < .00001"]


def p_level(p, n_tests):
    """0 = not significant, 1-4 = Bonferroni-corrected level reached (P_LEVELS / n_tests)."""
    return sum(p < np.array(P_LEVELS) / n_tests)


def p_grid(keys, kind, tests=tests):
    """(levels, raw p, absent-in-FS8) tables: rows = keys, columns = the 6 pairs."""
    t = tests[tests["kind"] == kind].set_index(["key", "pair"])
    p = pd.DataFrame({fsvv.label(q): [t.loc[(k, fsvv.tag(q)), "p"] for k in keys] for q in fsvv.PAIRS}, index=keys)
    absent = pd.DataFrame({fsvv.label(q): [bool(t.loc[(k, fsvv.tag(q)), "absent_in_FS8"]) for k in keys]
                           for q in fsvv.PAIRS}, index=keys)
    levels = p.map(lambda x: p_level(x, fsvv.N_TESTS[kind]))
    return levels, p, absent


def draw_grid(ax, levels, absent, labels=None, title=None):
    ax.imshow(levels.values, cmap=ListedColormap(P_COLORS), vmin=-0.5, vmax=4.5, aspect="auto",
              interpolation="nearest")
    for (i, j) in zip(*np.nonzero(absent.values)):
        ax.text(j, i, "×", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(levels.shape[1]))
    ax.set_xticklabels([c.replace("FS ", "") for c in levels.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(levels.shape[0]))
    ax.set_yticklabels(labels if labels is not None else [], fontsize=6)
    ax.set_xticks(np.arange(-0.5, levels.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, levels.shape[0]), minor=True)
    ax.grid(which="minor", color="0.8", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, fontsize=8)


def p_legend(fig, **kw):
    fig.legend([Patch(facecolor=c, edgecolor="0.6") for c in P_COLORS], P_LABELS, ncol=5, fontsize=7,
               frameon=False, **kw)


# %% [markdown]
# Figure 1: PPMI (all images, `02`) and the QPN replication sample (PD, `04_qpn_replication`) side by side.

# %%
rois = fsvv.ASEG_PLOT_ORDER
qpn_tests = pd.read_csv(fsvv.OUT / "qpn_version_paired_ttests.csv", dtype={"pair": str})
n_qpn_pd = int(pd.read_csv(fsvv.OUT / "qpn_demographics.csv", index_col="dx_group").loc["PD", "n"])
n_ppmi = len(pd.read_csv(fsvv.OUT / "aseg_table4.csv", usecols=["PATNO_id"]))  # images analysed in 02
samples = {"PPMI": (tests, f"PPMI ({n_ppmi} images)"), "QPN": (qpn_tests, f"QPN, PD ({n_qpn_pd} images)")}
fig, axes = plt.subplots(1, 2, figsize=(5.2, 8), sharey=True)
frames, counts = {}, {}
for ax, (name, (t, title)) in zip(axes, samples.items()):
    levels, p, absent = p_grid(rois, "vol", t)
    draw_grid(ax, levels, absent, rois if ax is axes[0] else None, title)
    frames[name], counts[name] = p, (levels > 0).sum()
fig.suptitle("Subcortical volume", fontsize=9)
p_legend(fig, loc="lower center", bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
fsvv.savefig(fig, "fig1_pvalues_subcortical")
pd.concat(frames, names=["sample", "region"]).to_csv(TABLES / "fig1_pvalues_subcortical.csv")
pd.DataFrame(counts).T  # significant regions per pair

# %%
panels = [(kind, hemi) for kind in fsvv.CORTICAL_MEASURES for hemi in fsvv.HEMIS]
fig, axes = plt.subplots(1, len(panels), figsize=(14, 13))
frames = {}
for ax, (kind, hemi) in zip(axes, panels):
    keys = [f"{hemi}_{r}_{kind}" for r in fsvv.CORTICAL_ROIS]
    levels, p, absent = p_grid(keys, kind)
    draw_grid(ax, levels, absent, fsvv.CORTICAL_ROIS if ax is axes[0] else None,
              f"{fsvv.MEASURE_NAMES[kind]} ({hemi})")
    frames[(kind, hemi)] = p.set_axis(fsvv.CORTICAL_ROIS)
p_legend(fig, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout()
fsvv.savefig(fig, "fig2_pvalues_cortical")
pd.concat(frames, names=["measure", "hemi", "region"]).to_csv(TABLES / "fig2_pvalues_cortical.csv")
pd.DataFrame({f"{fsvv.MEASURE_NAMES[k]} {h}": (p_grid([f"{h}_{r}_{k}" for r in fsvv.CORTICAL_ROIS], k)[0] > 0).mean()
              * 100 for k, h in panels}).round(1)  # % of regions with a significant difference

# %% [markdown]
# ## Figures 3–6 — between-version vs between-subject variability
#
# Made by `02_software_variability` with the 6 pairs and the 4 versions: `variability_subcortical` (Figure 3),
# `variability_vol`, `variability_surf`, `variability_ct` (Figures 4–6; lh and rh side by side, the paper shows lh
# and puts rh in the supplement).

# %%
for name in ["variability_subcortical", "variability_vol", "variability_surf", "variability_ct"]:
    f = fsvv.FIGURES / f"{name}.pdf"
    print(f"{name}: {'ok' if f.exists() else 'MISSING (run 02)'}")

# %% [markdown]
# ## Tables 3–4 — subcortical volumes (longitudinal stream)
#
# Raw p; `*` = significant after Bonferroni (p < 0.05/14). Baseline: volume at the first visit; longitudinal:
# % change in volume (and change in UPDRS-III). n/a = label not estimated (FS8).

# %%
anc = pd.read_csv(fsvv.OUT / "clinical_ancova_hc_pdnonmci.csv")
corr = pd.read_csv(fsvv.OUT / "clinical_partial_corr_updrs.csv")


def paper_table(results, stat):
    return pd.concat({"Baseline": fsvv.clinical_table(results, stat, "base"),
                      "Longitudinal": fsvv.clinical_table(results, stat, "long")}, names=["time"])


table3 = paper_table(anc, "F")
table4 = paper_table(corr, "r")
table3.to_csv(TABLES / "table3_ancova_subcortical.csv")
table4.to_csv(TABLES / "table4_corr_subcortical.csv")
table3

# %%
table4

# %% [markdown]
# ## Tables 5–6 — vertex-wise cortical thickness (significant clusters)
#
# Clusters of `mri_glmfit-sim` (cluster-forming threshold p < .05, two-sided; cluster-wise p < .05, corrected for both
# hemispheres). Max = peak −log10(p), signed; CWP = cluster-wise p. Versions without a row have no significant
# cluster.

# %%
clusters = pd.read_csv(fsvv.OUT / "vertexwise_clusters.csv")


def cluster_table(analysis):
    c = clusters[clusters["analysis"] == analysis].copy()
    c["time"] = c["time"].map({"base": "Baseline", "long": "Longitudinal"})
    c["Region"] = c["hemi"].str[0].str.upper() + " " + c["Annot"]
    c = c.sort_values(["time", "version", "hemi", "ClusterNo"])
    cols = {"Region": "Region", "Size(mm^2)": "Size (mm2)", "MNIX": "MNI X", "MNIY": "MNI Y", "MNIZ": "MNI Z",
            "Max": "Max", "CWP": "CWP"}
    return c.set_index(["time", "version"])[list(cols)].rename(columns=cols)


table5 = cluster_table("corr")
table6 = cluster_table("group")
table5.to_csv(TABLES / "table5_vertexwise_corr.csv")
table6.to_csv(TABLES / "table6_vertexwise_group.csv")
table5

# %%
table6

# %% [markdown]
# ## Figures 7–10 — vertex-wise maps
#
# Cluster-corrected significance (`cache.th13.abs.sig.masked.mgh`, signed −log10 p) on the fsaverage inflated
# surface (FS7's fsaverage, `outputs/fsaverage`). One row per version; lateral and medial views of each hemisphere.
# Same colour scale in every panel.

# %%
FSAVERAGE = fsvv.OUT / "fsaverage"
VIEWS = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]
VMAX = 5


def plot_glm(analysis, time, name, title):
    fig, axes = plt.subplots(len(fsvv.VERSIONS), len(VIEWS), figsize=(10, 1.9 * len(fsvv.VERSIONS) + 1),
                             subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.12, wspace=-0.05, hspace=-0.1)
    n_vertices = {}
    for row, v in zip(axes, fsvv.VERSIONS):
        for ax, (hemi, view) in zip(row, VIEWS):
            sig = nib.load(fsvv.glm_contrast_dir(v, analysis, time, hemi) / "cache.th13.abs.sig.masked.mgh")
            sig = sig.get_fdata().ravel()
            n_vertices[(v, hemi)] = int((sig != 0).sum())
            plotting.plot_surf_stat_map(str(FSAVERAGE / f"{hemi}.inflated"), sig, hemi="left" if hemi == "lh" else "right",
                                        view=view, bg_map=str(FSAVERAGE / f"{hemi}.sulc"), axes=ax, figure=fig,
                                        colorbar=False, threshold=1.3, vmax=VMAX, cmap="cold_hot")
            ax.set_rasterized(True)  # the vector meshes would make a ~140 MB PDF
        row[0].text2D(-0.05, 0.5, v, transform=row[0].transAxes, rotation=90, va="center", fontsize=10)
    for ax, (hemi, view) in zip(axes[0], VIEWS):
        ax.set_title(f"{hemi} {view}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="cold_hot", norm=plt.Normalize(-VMAX, VMAX))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.35, 0.06, 0.3, 0.02]), orientation="horizontal")
    cb.set_label("signed −log10(p), cluster-corrected", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fsvv.savefig(fig, name, dpi=200)
    plt.close(fig)
    return pd.Series(n_vertices, name=name)


figs = [("corr", "base", "fig7_corr_base", "Figure 7. Correlation between UPDRS-III and cortical thickness (baseline)"),
        ("corr", "long", "fig8_corr_long", "Figure 8. Correlation between the change in UPDRS-III and the rate of change in thickness"),
        ("group", "base", "fig9_group_base", "Figure 9. HC vs PD-non-MCI, cortical thickness (baseline)"),
        ("group", "long", "fig10_group_long", "Figure 10. HC vs PD-non-MCI, rate of change in cortical thickness")]
n_sig = pd.concat([plot_glm(*f) for f in figs], axis=1)
n_sig.to_csv(TABLES / "fig7-10_significant_vertices.csv")
n_sig  # vertices in a significant cluster
