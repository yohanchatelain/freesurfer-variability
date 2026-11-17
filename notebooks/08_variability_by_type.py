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
# # 08 · Software variability by structure type, relative error and asymmetry (Reviewer 2)
#
# From the table of `02_software_variability` (`aseg_table4.csv`: 830 images, measures of the four versions).
#
# 1. Subcortical volumes split by type (large structures, subcortical gray matter, CSF spaces), left and right merged
#    (bilateral volume = left + right): relative difference between versions, |A − B| / B in %, per pair, next to the
#    between-subject coefficient of variation (SD / mean, %) of each version — the relative counterpart of the
#    between-subject SD markers of Figures 3–6.
# 2. Cortical regions (Destrieux, bilateral: mean thickness and volume, summed area), relative differences summarized
#    per lobe: for each image, the median over the lobe's regions.
# 3. Asymmetry: for every structure with a left and a right label and every version pair, paired t-test (across
#    images) of the left vs right relative difference; Bonferroni per measure (alpha / number of structures × pairs).
#
# Labels FS8 doesn't produce (vessel, choroid plexus, optic chiasm, 5th ventricle) are left out of the FS8 pairs.
# Outputs: `outputs/figures/variability_by_type*`, aggregate CSVs `outputs/variability_by_type*.csv`,
# `outputs/asymmetry_ttests.csv`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import fsvv

plt.rcParams.update({"figure.dpi": 80, "font.size": 8})
table = pd.read_csv(fsvv.inp("aseg_table4.csv")).set_index("PATNO_id")
print(len(table), "images")
absent = set(fsvv.absent_labels(table, "FS8"))
print("labels FS8 doesn't produce:", sorted(absent))

# %% [markdown]
# ## Subcortical structures by type (bilateral)

# %%
TYPES = {
    "Large structures": ["Cerebellum-White-Matter", "Cerebellum-Cortex", "Brain-Stem", "CC_Posterior", "CC_Mid_Posterior",
                         "CC_Central", "CC_Mid_Anterior", "CC_Anterior"],
    "Subcortical gray matter": ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
                                "Accumbens-area", "VentralDC"],
    "CSF spaces": ["Lateral-Ventricle", "Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "5th-Ventricle", "CSF"],
}
OTHER = ["vessel", "choroid-plexus", "Optic-Chiasm"]  # left out of the figure (absent in FS8, tiny labels)


def bilateral_volume(struct, v):
    cols = [f"{s}-{struct}_{v}" for s in ("Left", "Right")]
    return table[cols].sum(axis=1, min_count=2) if cols[0] in table else table[f"{struct}_{v}"]


def produced_by_all(struct):
    names = {f"{s}-{struct}" for s in ("Left", "Right")} | {struct}
    return not (names & absent)


vol = {(st, v): bilateral_volume(st, v) for ty in TYPES.values() for st in ty for v in fsvv.VERSIONS}
# Labels FS8 (SynthSeg) produces with a different definition (e.g. CSF: SynthSeg labels the extra-cerebral CSF),
# from fsvv.incomparable_labels (median volume more than 10 times smaller or larger than FS7's)
inc = fsvv.incomparable_labels(table)
redefined = {st for ty in TYPES.values() for st in ty
             if (st in inc or f"Left-{st}" in inc) and produced_by_all(st)}
print("labels with a different definition in FS8 (left out of the FS8 pairs):",
      {st: (round(vol[(st, "FS8")].median(), 1), round(vol[(st, "FS7")].median(), 1)) for st in sorted(redefined)})
absent |= {f"{s}-{st}" for st in redefined for s in ("Left", "Right")} | redefined
rows = []
for ty, structs in TYPES.items():
    for st in structs:
        for pair in fsvv.PAIRS:
            if "FS8" in pair and not produced_by_all(st):
                continue
            a, b = vol[(st, pair[0])], vol[(st, pair[1])]
            rel = ((a - b).abs() / b * 100).replace([np.inf, -np.inf], np.nan)
            rows += [{"type": ty, "structure": st, "pair": fsvv.label(pair), "image": i, "rel_diff": x}
                     for i, x in rel.items()]
rel_sub = pd.DataFrame(rows)
cv_sub = pd.DataFrame([{"type": ty, "structure": st, "version": v,
                        "cv": vol[(st, v)].std() / vol[(st, v)].mean() * 100}
                       for ty, structs in TYPES.items() for st in structs for v in fsvv.VERSIONS
                       if v != "FS8" or produced_by_all(st)])
summary_sub = (rel_sub.groupby(["type", "structure", "pair"])["rel_diff"].median().unstack("pair")
               .join(cv_sub.pivot_table(index=["type", "structure"], columns="version", values="cv").add_prefix("CV ")))
summary_sub.round(2).to_csv(fsvv.out("variability_by_type_subcortical.csv"))
summary_sub.round(1)  # median relative difference (%) per pair; between-subject CV (%) per version

# %%
pairs = [fsvv.label(p) for p in fsvv.PAIRS]
fig, axes = plt.subplots(1, len(TYPES), figsize=(15, 6), gridspec_kw={"width_ratios": [len(s) for s in TYPES.values()]})
for ax, (ty, structs) in zip(axes, TYPES.items()):
    width = 0.8 / len(pairs)
    for j, pair in enumerate(pairs):
        data = [rel_sub.loc[(rel_sub["structure"] == st) & (rel_sub["pair"] == pair), "rel_diff"].dropna().values
                for st in structs]
        pos = np.arange(len(structs)) - 0.4 + width * (j + 0.5)
        keep = [k for k, d in enumerate(data) if len(d)]
        bp = ax.boxplot([data[k] for k in keep], positions=pos[keep], widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "linewidth": 0.8})
        for box in bp["boxes"]:
            box.set(facecolor=fsvv.PAIR_COLORS[pair], linewidth=0.5)
    for v in fsvv.VERSIONS:
        cv = cv_sub[(cv_sub["type"] == ty) & (cv_sub["version"] == v)].set_index("structure")["cv"]
        x = [k for k, st in enumerate(structs) if st in cv]
        ax.scatter(x, [cv[structs[k]] for k in x], marker=fsvv.VERSION_MARKERS[v], color="black", s=16, zorder=5,
                   label=f"between-subject CV {v}")
    ax.set_xticks(range(len(structs)))
    ax.set_xticklabels([s.replace("_", " ") for s in structs], rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_title(ty, fontsize=10)
    ax.grid(axis="y", which="major", color="0.85", linewidth=0.5)
axes[0].set_ylabel("relative difference between versions, |A − B| / B (%)")
handles = [plt.Rectangle((0, 0), 1, 1, color=fsvv.PAIR_COLORS[p]) for p in pairs]
h2, l2 = axes[0].get_legend_handles_labels()
fig.legend(handles + h2, pairs + l2, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.tight_layout()
fsvv.savefig(fig, "variability_by_type_subcortical")

# %% [markdown]
# ## Cortical regions by lobe (Destrieux, bilateral)

# %%
measures = table[[c for c in table if c.endswith(tuple(f"_{v}" for v in fsvv.VERSIONS))]]
bi = fsvv.bilateral_measures(measures)
lobe_of = {r: fsvv.DESTRIEUX_LOBE.get(r, "other") for r in fsvv.CORTICAL_ROIS}
lobes = [*fsvv.DESTRIEUX_LOBES, "other"]
rows = []
for kind in fsvv.CORTICAL_MEASURES:
    for pair in fsvv.PAIRS:
        a = bi[[f"{r}_bi_{kind}_{pair[0]}" for r in fsvv.CORTICAL_ROIS]].values
        b = bi[[f"{r}_bi_{kind}_{pair[1]}" for r in fsvv.CORTICAL_ROIS]].values
        rel = pd.DataFrame(np.abs(a - b) / b * 100, index=bi.index, columns=fsvv.CORTICAL_ROIS)
        for lobe in lobes:
            regs = [r for r in fsvv.CORTICAL_ROIS if lobe_of[r] == lobe]
            med = rel[regs].median(axis=1)
            rows += [{"measure": fsvv.MEASURE_NAMES[kind], "pair": fsvv.label(pair), "lobe": lobe, "image": i,
                      "rel_diff": x} for i, x in med.items()]
rel_ctx = pd.DataFrame(rows)
summary_ctx = rel_ctx.groupby(["measure", "lobe", "pair"])["rel_diff"].median().unstack("pair")[pairs]
summary_ctx.round(2).to_csv(fsvv.out("variability_by_type_cortical.csv"))
summary_ctx.round(1)  # median over images of the per-image median relative difference (%) of the lobe's regions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
    d = rel_ctx[rel_ctx["measure"] == fsvv.MEASURE_NAMES[kind]]
    width = 0.8 / len(pairs)
    for j, pair in enumerate(pairs):
        data = [d.loc[(d["lobe"] == l) & (d["pair"] == pair), "rel_diff"].dropna().values for l in lobes]
        bp = ax.boxplot(data, positions=np.arange(len(lobes)) - 0.4 + width * (j + 0.5), widths=width * 0.9,
                        patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 0.8})
        for box in bp["boxes"]:
            box.set(facecolor=fsvv.PAIR_COLORS[pair], linewidth=0.5)
    ax.set_xticks(range(len(lobes)))
    ax.set_xticklabels(lobes)
    ax.set_title(f"{fsvv.MEASURE_NAMES[kind]} (bilateral)", fontsize=10)
    ax.grid(axis="y", color="0.85", linewidth=0.5)
axes[0].set_ylabel("relative difference between versions (%),\nmedian over the lobe's regions")
fig.legend(handles, pairs, ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.tight_layout()
fsvv.savefig(fig, "variability_by_type_cortical")

# %% [markdown]
# ## Asymmetry of the version differences (left vs right relative difference)

# %%
lateral = [s.removeprefix("Left-") for s in fsvv.ASEG_ROIS if s.startswith("Left-")]
rows = []
for pair in fsvv.PAIRS:
    a, b = pair
    for st in lateral:
        if "FS8" in pair and not produced_by_all(st):
            continue
        rel = {s: ((table[f"{s}-{st}_{a}"] - table[f"{s}-{st}_{b}"]).abs() / table[f"{s}-{st}_{b}"] * 100)
               for s in ("Left", "Right")}
        d = pd.concat(rel, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        t, p = stats.ttest_rel(d["Left"], d["Right"])
        rows.append(("Volume (subcortical)", st, fsvv.label(pair), len(d), d["Left"].mean(), d["Right"].mean(), t, p))
    for kind in fsvv.CORTICAL_MEASURES:
        for r in fsvv.CORTICAL_ROIS:
            rel = {h: ((table[f"{r}_{h}_{kind}_{a}"] - table[f"{r}_{h}_{kind}_{b}"]).abs()
                       / table[f"{r}_{h}_{kind}_{b}"] * 100) for h in fsvv.HEMIS}
            d = pd.concat(rel, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            t, p = stats.ttest_rel(d["lh"], d["rh"])
            rows.append((fsvv.MEASURE_NAMES[kind], r, fsvv.label(pair), len(d), d["lh"].mean(), d["rh"].mean(), t, p))
asym = pd.DataFrame(rows, columns=["measure", "region", "pair", "n", "mean_rel_left", "mean_rel_right", "t", "p"])
n_tests = asym.groupby("measure")["p"].transform("size")
asym["significant"] = asym["p"] < fsvv.ALPHA / n_tests
asym.to_csv(fsvv.out("asymmetry_ttests.csv"), index=False)
asym_summary = asym.groupby(["measure", "pair"]).agg(
    tests=("p", "size"), significant=("significant", "sum"),
    left_larger=("t", lambda t: int(((t > 0) & asym.loc[t.index, "significant"]).sum())),
    right_larger=("t", lambda t: int(((t < 0) & asym.loc[t.index, "significant"]).sum())))
asym_summary.to_csv(fsvv.out("asymmetry_summary.csv"))
asym_summary
