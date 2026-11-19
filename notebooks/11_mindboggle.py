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
# # 11 · Accuracy against manual labels: Mindboggle-101 (Exp 2, Reviewer 1)
#
# The four versions (FS5.3, FS6.0.1, FS7.3.2, FS8.2.0) were run with `recon-all -all` on the 101 Mindboggle-101 T1w
# images (`scripts/exp2`). `scripts/exp2/eval.sbatch` resampled the DKT (`aparc.DKTatlas[40]+aseg`) and DK
# (`aparc+aseg`) cortical labels of each version to the native space of the T1w, and `eval_metrics.py` computed the
# Dice coefficient and volume of each of the 62 DKT31 regions against the manual labels
# (`$FSVV_MB_WORK/eval/dice_FSx.csv`; DK regions absent from DKT31 are not scored).
#
# 1. Dice per version: subject mean over the 62 regions, by parcellation.
# 2. Paired Wilcoxon signed-rank tests between versions on the subject means (6 pairs × 2 parcellations, Bonferroni).
# 3. Per region: paired tests per region and pair (Bonferroni over 62 regions × 6 pairs per parcellation), and the
#    number of regions where the newer version is significantly better or worse.
# 4. Per Mindboggle sub-dataset (scanner/site).
# 5. Volume error vs the manual labels (relative, %).
#
# Caveat: the FreeSurfer DKT classifier atlas was built from 40 of the Mindboggle-101 brains, so the DKT parcellation
# is evaluated partly on its own training data and its Dice is optimistic. The DK parcellation (atlas built from other
# brains) is not affected, but it follows a different protocol from the manual DKT31 labels, which lowers its Dice.
#
# Outputs (aggregate only): `outputs/mindboggle_*.csv`, `outputs/figures/mindboggle_dice.*`.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import fsvv

plt.rcParams.update({"figure.dpi": 80, "font.size": 8})
EVAL = Path(os.environ.get("FSVV_MB_WORK", Path(os.environ.get("SCRATCH", "")) / "fsvv/mindboggle")) / "eval"
PARCS = {"dkt": "DKT", "dk": "DK"}
d = pd.concat([pd.read_csv(EVAL / f"dice_{v}.csv") for v in fsvv.VERSIONS], ignore_index=True)

# FreeSurfer colour-table codes of the DK/DKT cortical regions: 1000 + index (lh), 2000 + index (rh)
DK_CODES = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus", "entorhinal",
            "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
            "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
            "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
            "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
            "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
            "temporalpole", "transversetemporal", "insula"]
d["hemi"] = np.where(d["label"] < 2000, "lh", "rh")
d["region"] = d["label"].map(lambda l: DK_CODES[l % 1000 - 1])
d["lobe"] = d["region"].map(fsvv.DK_LOBE)
d["dataset"] = d["subject"].str.rsplit("-", n=1).str[0]
d["vol_err"] = 100 * (d["vol_auto"] - d["vol_manual"]) / d["vol_manual"]

n = d.groupby(["version", "parcellation"]).agg(subjects=("subject", "nunique"), regions=("label", "nunique"),
                                                missing_dice=("dice", lambda x: x.isna().sum()))
print(n.to_string())
assert (n["subjects"] == 101).all() and (n["regions"] == 62).all()

# %% [markdown]
# ## 1 · Dice per version (subject mean over the 62 regions)

# %%
subj = d.groupby(["parcellation", "version", "subject"])["dice"].mean().rename("dice").reset_index()
summary = (subj.groupby(["parcellation", "version"])["dice"]
           .agg(mean="mean", sd="std", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
                min="min")
           .join(d.groupby(["parcellation", "version"])["dice"].agg(region_mean="mean", region_median="median"))
           .round(4))
summary.to_csv(fsvv.out("mindboggle_summary.csv"))
summary

# %% [markdown]
# ## 2 · Paired tests between versions (subject means)

# %%
wide = subj.pivot_table(index=["parcellation", "subject"], columns="version", values="dice")
N_PAIR_TESTS = len(fsvv.PAIRS) * len(PARCS)
rows = []
for parc in PARCS:
    w = wide.loc[parc]
    for a, b in fsvv.PAIRS:
        diff = w[a] - w[b]
        res = stats.wilcoxon(w[a], w[b])
        rows.append(dict(parcellation=parc, pair=fsvv.label((a, b)), mean_diff=diff.mean(), median_diff=diff.median(),
                         newer_better=int((diff > 0).sum()), newer_worse=int((diff < 0).sum()), W=res.statistic,
                         p=res.pvalue, p_fmt=fsvv.fmt_p(res.pvalue, N_PAIR_TESTS)))
pairs = pd.DataFrame(rows)
pairs.to_csv(fsvv.out("mindboggle_pairs.csv"), index=False)
print(f"Bonferroni: alpha / {N_PAIR_TESTS} = {fsvv.ALPHA / N_PAIR_TESTS:.4f}")
pairs.round(4)

# %% [markdown]
# ## 3 · Per region

# %%
N_REGION_TESTS = d["label"].nunique() * len(fsvv.PAIRS)
reg_rows = []
for (parc, label), g in d.groupby(["parcellation", "label"]):
    w = g.pivot(index="subject", columns="version", values="dice")
    r = dict(parcellation=parc, label=label, hemi=g["hemi"].iat[0], region=g["region"].iat[0], lobe=g["lobe"].iat[0])
    r.update({f"dice_{v}": w[v].mean() for v in fsvv.VERSIONS})
    for a, b in fsvv.PAIRS:
        ok = w[[a, b]].dropna()
        p = stats.wilcoxon(ok[a], ok[b]).pvalue if (ok[a] != ok[b]).any() else 1.0
        r[f"diff_{fsvv.tag((a, b))}"] = (ok[a] - ok[b]).mean()
        r[f"p_{fsvv.tag((a, b))}"] = p
    reg_rows.append(r)
regions = pd.DataFrame(reg_rows)
regions.to_csv(fsvv.out("mindboggle_regions.csv"), index=False)

thr = fsvv.ALPHA / N_REGION_TESTS
counts = []
for parc in PARCS:
    rp = regions[regions["parcellation"] == parc]
    for pair in fsvv.PAIRS:
        t = fsvv.tag(pair)
        sig = rp[f"p_{t}"] < thr
        counts.append(dict(parcellation=parc, pair=fsvv.label(pair), significant=int(sig.sum()),
                           newer_better=int((sig & (rp[f"diff_{t}"] > 0)).sum()),
                           newer_worse=int((sig & (rp[f"diff_{t}"] < 0)).sum())))
counts = pd.DataFrame(counts)
counts.to_csv(fsvv.out("mindboggle_region_counts.csv"), index=False)
print(f"Bonferroni per parcellation: alpha / {N_REGION_TESTS} = {thr:.2e}")
counts

# %%
# Regions with the largest drop from FS5 to FS8 (mean Dice difference), DKT parcellation
cols = ["hemi", "region", "lobe"] + [f"dice_{v}" for v in fsvv.VERSIONS] + ["diff_85", "p_85"]
regions[regions["parcellation"] == "dkt"].sort_values("diff_85")[cols].head(10).round(4)

# %%
lobes = (d.groupby(["parcellation", "lobe", "version"])["dice"].mean().unstack("version")[fsvv.VERSIONS].round(3))
lobes.to_csv(fsvv.out("mindboggle_lobes.csv"))
lobes

# %% [markdown]
# ## 4 · Per Mindboggle sub-dataset

# %%
by_ds = (subj.assign(dataset=subj["subject"].str.rsplit("-", n=1).str[0])
         .groupby(["parcellation", "dataset", "version"])["dice"].mean().unstack("version")[fsvv.VERSIONS])
by_ds["n"] = subj[subj["version"] == "FS5"].assign(dataset=lambda x: x["subject"].str.rsplit("-", n=1).str[0]) \
    .groupby(["parcellation", "dataset"]).size()
by_ds = by_ds.round(3)
by_ds.to_csv(fsvv.out("mindboggle_datasets.csv"))
by_ds

# %% [markdown]
# ## 5 · Volume error vs the manual labels

# %%
vol = (d.groupby(["parcellation", "version"])["vol_err"]
       .agg(median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
            abs_median=lambda x: x.abs().median())
       .round(2))
vol.to_csv(fsvv.out("mindboggle_volume_error.csv"))
vol

# %% [markdown]
# ## Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
for ax, parc in zip(axes[:2], PARCS):
    data = [wide.loc[parc][v].values for v in fsvv.VERSIONS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops=dict(markersize=2))
    for patch, v in zip(bp["boxes"], fsvv.VERSIONS):
        patch.set_facecolor(fsvv.VERSION_COLORS[v])
        patch.set_alpha(0.7)
    for i, v in enumerate(fsvv.VERSIONS):
        ax.scatter(np.random.default_rng(0).normal(i + 1, 0.06, len(data[i])), data[i], s=2, color="k", alpha=0.3)
    ax.set_xticks(range(1, 5), [v.replace("FS", "FS ") for v in fsvv.VERSIONS])
    ax.set_title(f"{PARCS[parc]} parcellation")
    ax.set_ylabel("Dice vs manual DKT31 (subject mean)")
ax = axes[2]
lk = lobes.loc["dkt"]
x = np.arange(len(lk))
for i, v in enumerate(fsvv.VERSIONS):
    ax.bar(x + (i - 1.5) * 0.2, lk[v], 0.2, color=fsvv.VERSION_COLORS[v], label=v.replace("FS", "FS "))
ax.set_xticks(x, lk.index, rotation=30, ha="right")
ax.set_ylim(0.6, 0.95)
ax.set_title("DKT, mean Dice per lobe")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fsvv.savefig(fig, "mindboggle_dice")
