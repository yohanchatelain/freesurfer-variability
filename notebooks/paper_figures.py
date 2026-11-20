"""Paper figures with every visual setting as a parameter (Figures 1-12 and S1-S6), shared by the tuning notebooks
(09_figure_tuning, 10_supp_figure_tuning; UI in figure_tuning.py), scripts/paper/figure_variants.py and, once the settings are chosen, 02/06.

Each draw_* function returns a matplotlib figure; data are loaded once by load_data(). The chosen settings are kept in
scripts/paper/figure_params.json (no participant data): save_to_paper() writes the figure under its paper name in
outputs/figures and records its settings there.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import fsvv

PARAMS_FILE = fsvv.ROOT / "scripts" / "paper" / "figure_params.json"
PAPER_NAMES = {"fig1": "fig1_pvalues_subcortical", "fig2": "fig2_pvalues_cortical", "fig3": "variability_subcortical",
               "fig3a": "variability_subcortical_part1", "fig3b": "variability_subcortical_part2", "fig4": "variability_vol",
               "fig5": "variability_surf", "fig6": "variability_ct", "fig7": "fig_subcortical_group",
               "fig8": "fig_subcortical_corr", "fig9": "fig7_corr_base", "fig10": "fig8_corr_long",
               "fig11": "fig9_group_base", "fig12": "fig10_group_long",
               "figS1": "estimates_subcortical", "figS2": "estimates_cortical", "figS4": "segmentation_dice",
               "figS5": "variability_by_type_subcortical", "figS6": "variability_by_type_cortical"}
# Tabs of the tuning notebooks (09: main paper, 10: supplement), in the order of the PDFs. Figure S3 (example
# hippocampi) is a picture, not drawn here.
MAIN_FIGURES = ["fig1", "fig2", "fig3a", "fig3b", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10",
                "fig11", "fig12"]
SUPP_FIGURES = ["figS1", "figS2", "figS4", "figS5", "figS6"]
TITLES = {"fig1": "Fig 1 p-values subcortical", "fig2": "Fig 2 p-values cortical", "fig3a": "Fig 3a subcortical",
          "fig3b": "Fig 3b subcortical", "fig3": "Fig 3 subcortical (full)", "fig4": "Fig 4 volume",
          "fig5": "Fig 5 area", "fig6": "Fig 6 thickness", "fig7": "Fig 7 group (subcortical)",
          "fig8": "Fig 8 UPDRS (subcortical)", "fig9": "Fig 9 UPDRS baseline (cortex)",
          "fig10": "Fig 10 UPDRS change (cortex)", "fig11": "Fig 11 group baseline (cortex)",
          "fig12": "Fig 12 group change (cortex)", "figS1": "S1 estimates subcortical", "figS2": "S2 estimates cortical",
          "figS4": "S4 segmentation Dice", "figS5": "S5 variability by type", "figS6": "S6 variability by lobe"}
KIND_OF = {"fig4": "vol", "fig5": "surf", "fig6": "ct"}
DEFAULTS = {
    "fig2": dict(width=16, height=22, font=10, title_font=12, legend_font=10, grid_lw=0.3),
    "fig3": dict(layout="enigma", metric="ratio", width=12, height=16, font=10, title_font=13, legend_font=10,
                 fliers=False, xscale="linear", xlim="whiskers", xmax=None, marker_size=30, box_width=0.8,
                 legend_cols=5, ratio_max=2.0, ratio_cmap="RdBu_r", lower_height=0.3, flier_size=3.0),
    # Figure 3 split in two (see _aseg_parts), each part with its own settings
    "fig3a": dict(width=17, height=10, font=20, title_font=24, legend_font=20, fliers=False, flier_size=3.0,
                  xscale="linear", xlim="whiskers", xmax=None, marker_size=100, box_width=0.85, legend_cols=5),
    "fig3b": dict(width=17, height=34, font=20, title_font=24, legend_font=20, fliers=False, flier_size=3.0,
                  xscale="linear", xlim="whiskers", xmax=None, marker_size=100, box_width=0.85, legend_cols=5),
    "fig4": dict(layout="ratio_heatmap", hemis="lh+rh", width=12, height=18, font=9, title_font=12, legend_font=9,
                 fliers=False, xscale="linear", xlim="whiskers", xmax=None, marker_size=30, box_width=0.8,
                 legend_cols=5, ratio_max=2.0, ratio_cmap="RdBu_r", lobe_split=True, preview_part=1,
                 flier_size=3.0),
    "fig7": dict(style="glass_subcortical", width=9, row_height=2.0, font=12, colorbar_font=10, vmax=5.0, threshold=1.3,
                 cmap="cold_hot", cut_y="10,-5,-20", ventricles=True,
                 superior_anterior="up", bonferroni=True, view1="L lateral", view2="R medial", view3="superior"),
}
DEFAULTS.update({
    "fig1": dict(width=5.2, height=8, font=7, title_font=8, suptitle_font=9, legend_font=7, grid_lw=0.3, cross_size=6,
                 legend_cols=5),
    "figS1": dict(width=12, height=4, font=7, title_font=10, legend_font=8, box_width=0.8, fliers=True,
                  flier_size=1.5, yscale="linear", legend_cols=4),
    "figS2": dict(width=18, height=24, font=7, title_font=10, legend_font=8, box_width=0.8, fliers=True,
                  flier_size=1.5, yscale="linear", legend_cols=4),
    "figS4": dict(width=14, height=16, font=7, title_font=10, legend_font=8, box_width=0.8, fliers=True,
                  flier_size=1.5, xmin=0.0, legend_cols=6),
    "figS5": dict(width=15, height=6, font=8, title_font=10, legend_font=7, box_width=0.8, fliers=False,
                  flier_size=1.5, marker_size=16, yscale="log", legend_cols=5),
    "figS6": dict(width=15, height=4.5, font=8, title_font=10, legend_font=7, box_width=0.8, fliers=False,
                  flier_size=1.5, yscale="linear", legend_cols=6),
})
for _f in ("fig9", "fig10", "fig11", "fig12"):
    DEFAULTS[_f] = dict(width=10, row_height=1.9, font=10, title_font=10, colorbar_font=8, vmax=5.0, threshold=1.3,
                        cmap="cold_hot", title=True, wspace=0.0, hspace=0.0)
DEFAULTS["fig5"] = dict(DEFAULTS["fig4"])
DEFAULTS["fig6"] = dict(DEFAULTS["fig4"])
DEFAULTS["fig8"] = dict(DEFAULTS["fig7"])
# Position of the figure legend (see _place_legend), for every figure with a legend
LEGEND_POSITIONS = ["below", "above", "right", "left", "inside upper right", "inside upper left",
                    "inside lower right", "inside lower left"]
for _p in DEFAULTS.values():
    if "legend_font" in _p:
        _p.update(legend_pos="below", legend_dx=0.0, legend_dy=0.0)

P_LEVELS = [0.05, 0.001, 0.0001, 0.00001]
P_COLORS = ["white", "#4daf4a", "#ffff33", "#ff7f00", "#e41a1c"]
P_LABELS = ["n.s.", "p < .05", "p < .001", "p < .0001", "p < .00001"]
SUBCORTICAL_IDS = {"Thalamus": (10, 49), "Caudate": (11, 50), "Putamen": (12, 51), "Pallidum": (13, 52),
                   "Hippocampus": (17, 53), "Amygdala": (18, 54), "Accumbens-area": (26, 58)}
_data = {}


def load_data():
    """Inputs of the figures (participant-level table read on the cluster only; never written elsewhere)."""
    if not _data:
        _data["table"] = pd.read_csv(fsvv.inp("aseg_table4.csv")).set_index("PATNO_id")
        _data["tests"] = pd.read_csv(fsvv.OUT / "version_paired_ttests.csv", dtype={"pair": str})
        _data["anc"] = pd.read_csv(fsvv.OUT / "clinical_ancova_hc_pdnonmci.csv")
        _data["corr"] = pd.read_csv(fsvv.OUT / "clinical_partial_corr_updrs.csv")
    return _data


SEED_FROM = {"fig8": "fig7", "fig10": "fig9", "fig11": "fig9", "fig12": "fig9"}  # a figure not saved yet starts from the saved settings of its companion


def load_params():
    """Defaults updated by the saved settings (keys no longer in DEFAULTS are dropped)."""
    saved = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    saved = {**{k: saved[src] for k, src in SEED_FROM.items() if k not in saved and src in saved}, **saved}
    return {k: {**v, **{kk: vv for kk, vv in saved.get(k, {}).items() if kk in v}} for k, v in DEFAULTS.items()}


def save_to_paper(fig_id, fig, params):
    """Write the figure under its paper name (PNG + PDF in outputs/figures) and record its settings.

    Figures 4-6 with lobe_split: both parts are drawn and written (<name>_part1, <name>_part2); `fig` is ignored.
    """
    if fig_id in KIND_OF and params.get("lobe_split"):
        for part in (1, 2):
            f = draw(fig_id, **{**params, "preview_part": part})
            fsvv.savefig(f, f"{PAPER_NAMES[fig_id]}_part{part}")
            plt.close(f)
    else:
        fsvv.savefig(fig, PAPER_NAMES[fig_id])
    saved = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    saved[fig_id] = params
    PARAMS_FILE.write_text(json.dumps(saved, indent=1, sort_keys=True) + "\n")
    return fsvv.FIGURES / f"{PAPER_NAMES[fig_id]}.pdf"


# --------------------------------------------------------------------------- Figure 2: p-value grid
def draw_fig2(width=16, height=22, font=10, title_font=12, legend_font=10, grid_lw=0.3):
    tests = load_data()["tests"]
    panels = [(k, h) for k in fsvv.CORTICAL_MEASURES for h in fsvv.HEMIS]
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height))
    for ax, (kind, hemi) in zip(axes, panels):
        t = tests[tests["kind"] == kind].set_index(["key", "pair"])
        keys = [f"{hemi}_{r}_{kind}" for r in fsvv.CORTICAL_ROIS]
        lev = np.array([[sum(t.loc[(k, fsvv.tag(q)), "p"] < np.array(P_LEVELS) / fsvv.N_TESTS[kind])
                         for q in fsvv.PAIRS] for k in keys])
        ax.imshow(lev, cmap=ListedColormap(P_COLORS), vmin=-0.5, vmax=4.5, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(fsvv.PAIRS)))
        ax.set_xticklabels([fsvv.label(q).replace("FS ", "") for q in fsvv.PAIRS], rotation=90, fontsize=font)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(fsvv.CORTICAL_ROIS if ax is axes[0] else [], fontsize=font)
        ax.set_xticks(np.arange(-0.5, len(fsvv.PAIRS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
        ax.grid(which="minor", color="0.8", linewidth=grid_lw)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{fsvv.MEASURE_NAMES[kind]} ({hemi})", fontsize=title_font)
    fig.legend([Patch(facecolor=c, edgecolor="0.6") for c in P_COLORS], P_LABELS, ncol=5, fontsize=legend_font,
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


# --------------------------------------------------------------------------- Figures 3-6: variability
def _legend_below(fig, ax, font, cols):
    handles, labels = ax.get_legend_handles_labels()
    boxes = [plt.Rectangle((0, 0), 1, 1, color=fsvv.PAIR_COLORS[fsvv.label(p)]) for p in fsvv.PAIRS]
    fig.legend(boxes + handles, [fsvv.label(p) for p in fsvv.PAIRS] + labels, ncol=cols, fontsize=font,
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0),
               title="boxes: between-version |difference|; markers: between-subject SD", title_fontsize=font)


def _style_axes(ax, title_font, xmax, xscale):
    ax.title.set_fontsize(title_font)
    if xmax:
        lo = ax.get_xlim()[0] if xscale == "log" else -0.02 * xmax
        ax.set_xlim(lo, xmax)


def _aseg_parts():
    """Figure 3 split: part 1 = lateral ventricles and cerebellum white matter and cortex (left and right), part 2 =
    the other structures. Their differences and SDs are much larger, so on a shared linear axis they squash the others."""
    large = [r for r in fsvv.ASEG_PLOT_ORDER if "Cerebellum" in r or r.endswith("-Lateral-Ventricle")]
    return {1: (large, "Subcortical volumes (a)"),
            2: ([r for r in fsvv.ASEG_PLOT_ORDER if r not in large], "Subcortical volumes (b)")}


def draw_variability(fig_id, width, height, font=10, title_font=13, legend_font=10, fliers=False, xscale="linear",
                     xlim="whiskers", xmax=None, marker_size=30, box_width=0.8, legend_cols=5, part=None,
                     flier_size=3.0):
    """Figure 3 (fig_id "fig3": subcortical) or 4-6 ("fig4"/"fig5"/"fig6": cortical lh and rh).

    fliers=True draws the outliers (size flier_size, colour of the pair); with xlim="whiskers" those beyond the whiskers
    and SD markers are clipped, xlim="auto" shows them all.
    Figure 3 with part=1 or 2 draws only that part of _aseg_parts() (figures "fig3a" and "fig3b").
    """
    table = load_data()["table"]
    kw = dict(fontsize=font, showfliers=fliers, xscale=xscale, xlim=xlim if xlim != "auto" else None,
              marker_size=marker_size, legend=False, box_width=box_width, flier_size=flier_size)
    if fig_id == "fig3":
        regions, title = _aseg_parts()[part] if part else (fsvv.ASEG_PLOT_ORDER, None)
        n_rows = int(np.ceil((len(fsvv.PAIRS) + len(fsvv.VERSIONS)) / legend_cols)) + 1  # + legend title
        fig, ax = plt.subplots(figsize=(width, height))
        fsvv.plot_variability(table, "aseg", ax=ax, regions=regions, title=title, **kw)
        _style_axes(ax, title_font, xmax, xscale)
        _legend_below(fig, ax, legend_font, legend_cols)
        fig.tight_layout(rect=(0, n_rows * legend_font * 1.8 / 72 / height, 1, 1))
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True)
    for ax, hemi in zip(axes, fsvv.HEMIS):
        fsvv.plot_variability(table, KIND_OF[fig_id], hemi, ax=ax, **kw)
    lims = [a.get_xlim() for a in axes]
    axes[0].set_xlim(min(l[0] for l in lims), max(l[1] for l in lims))
    for ax in axes:
        _style_axes(ax, title_font, xmax, xscale)
    _legend_below(fig, axes[0], legend_font, legend_cols)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


# --------------------------------------------------------------------------- Figures 4-6: cortical layouts
def _cortical_values(kind, hemis):
    """{side: (diffs {pair label: DataFrame images x regions of |A - B|}, sds DataFrame regions x versions)}.

    hemis "lh+rh": lh and rh; "bilateral": mean of lh and rh for thickness and volume, sum for surface area.
    """
    table = load_data()["table"]
    rois = fsvv.CORTICAL_ROIS
    if hemis == "bilateral":
        est = fsvv.bilateral_measures(table)  # all three measures (it loops over them)
        sides = {"bilateral": lambda v: est[[f"{r}_bi_{kind}_{v}" for r in rois]].set_axis(rois, axis=1)}
    else:
        sides = {h: (lambda v, h=h: table[[f"{r}_{h}_{kind}_{v}" for r in rois]].set_axis(rois, axis=1))
                 for h in fsvv.HEMIS}
    out = {}
    for side, get in sides.items():
        diffs = {fsvv.label(p): (get(p[0]) - get(p[1])).abs() for p in fsvv.PAIRS}
        sds = pd.DataFrame({v: get(v).std() for v in fsvv.VERSIONS})
        out[side] = (diffs, sds)
    return out


def _lobe_order():
    lobe_of = {r: fsvv.DESTRIEUX_LOBE.get(r, "other") for r in fsvv.CORTICAL_ROIS}
    lobes = [*fsvv.DESTRIEUX_LOBES, "other"]
    order = [r for l in lobes for r in fsvv.CORTICAL_ROIS if lobe_of[r] == l]
    return order, lobe_of, lobes


LOBE_PARTS = {1: ["limbic", "frontal", "temporal"], 2: ["occipital", "parietal", "other"]}


def draw_cortical(fig_id, layout="ratio_heatmap", hemis="lh+rh", width=12, height=18, font=9, title_font=12,
                  legend_font=9, fliers=False, xscale="linear", xlim="whiskers", xmax=None, marker_size=30,
                  box_width=0.8, legend_cols=5, ratio_max=2.0, ratio_cmap="RdBu_r", lobe_split=False,
                  preview_part=1, flier_size=3.0):
    """Figures 4-6 (volume, area, thickness): between-version |difference| vs between-subject SD.

    layout: "ratio_heatmap" (median |diff| / mean SD of the two versions, regions x pairs, grouped by lobe),
    "dots" (median and IQR per pair, SD markers), "boxes_by_lobe" (boxes, one panel per lobe), "boxes" (original).
    """
    kind = KIND_OF[fig_id]
    if layout == "boxes":
        return draw_variability(fig_id, width, height, font, title_font, legend_font, fliers, xscale, xlim, xmax,
                                marker_size, box_width, legend_cols, flier_size=flier_size)
    vals = _cortical_values(kind, hemis)
    order, lobe_of, lobes = _lobe_order()
    if lobe_split:  # keep only the lobes of the requested part
        lobes = [l for l in lobes if l in LOBE_PARTS[int(preview_part)]]
        order = [r for r in order if lobe_of[r] in lobes]
    unit = fsvv.MEASURE_UNITS[kind]
    name = fsvv.MEASURE_NAMES[kind]
    pairs = [fsvv.label(p) for p in fsvv.PAIRS]
    bounds = np.cumsum([sum(lobe_of[r] == l for r in order) for l in lobes])[:-1] - 0.5

    if layout == "ratio_heatmap":
        fig, axes = plt.subplots(1, len(vals), figsize=(width, height), sharey=True, squeeze=False)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=ratio_max)
        for ax, (side, (diffs, sds)) in zip(axes[0], vals.items()):
            ratio = pd.DataFrame({fsvv.label(p): diffs[fsvv.label(p)].median()
                                  / sds[[p[0], p[1]]].mean(axis=1) for p in fsvv.PAIRS}).loc[order]
            im = ax.imshow(ratio.values, aspect="auto", cmap=ratio_cmap, norm=norm, interpolation="nearest")
            ax.set_xticks(range(len(pairs)))
            ax.set_xticklabels([p.replace("FS ", "") for p in pairs], rotation=90, fontsize=font)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=font)
            for b in bounds:
                ax.axhline(b, color="black", linewidth=0.8)
            ax.set_title(side, fontsize=title_font)
        last = axes[0][-1]
        tr = matplotlib.transforms.blended_transform_factory(last.transAxes, last.transData)
        for b, nxt, l in zip(np.r_[-0.5, bounds], np.r_[bounds, len(order) - 0.5], lobes):  # lobe names on the right
            last.text(1.02, (b + nxt) / 2, l, transform=tr, rotation=-90, va="center", fontsize=font)
        cb = fig.colorbar(im, ax=axes[0].tolist(), orientation="horizontal", fraction=0.025, pad=0.06)
        cb.set_label(f"{name}: median between-version |difference| / between-subject SD "
                     "(> 1: versions differ more than subjects)", fontsize=legend_font)
        cb.ax.tick_params(labelsize=font)
        return fig

    if layout == "dots":
        fig, axes = plt.subplots(1, len(vals), figsize=(width, height), sharey=True, sharex=True, squeeze=False)
        y = np.arange(len(order))
        offs = np.linspace(-0.3, 0.3, len(pairs))
        for ax, (side, (diffs, sds)) in zip(axes[0], vals.items()):
            for off, p in zip(offs, pairs):
                d = diffs[p][order]
                q1, med, q3 = d.quantile(0.25), d.median(), d.quantile(0.75)
                ax.hlines(y + off, q1, q3, color=fsvv.PAIR_COLORS[p], linewidth=1)
                ax.scatter(med, y + off, color=fsvv.PAIR_COLORS[p], s=marker_size * 0.5, zorder=3, label=p)
            for v in fsvv.VERSIONS:
                ax.scatter(sds.loc[order, v], y, marker=fsvv.VERSION_MARKERS[v], color="black",
                           s=marker_size * 0.6, linewidths=0.8, zorder=4, label=f"SD {v}")
            for b in bounds:
                ax.axhline(b, color="0.5", linewidth=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(order, fontsize=font)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_xscale(xscale)
            if xmax:
                ax.set_xlim(right=xmax)
            ax.tick_params(axis="x", labelsize=font)
            ax.grid(axis="x", linestyle=":", linewidth=0.5)
            ax.set_xlabel(f"{name} ({unit})", fontsize=font + 1)
            ax.set_title(side, fontsize=title_font)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, ncol=legend_cols, fontsize=legend_font, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 0), title="dots and lines: median and IQR of the between-version |difference|; "
                   "black: between-subject SD", title_fontsize=legend_font)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    # boxes_by_lobe: one row of panels per lobe (heights proportional to the number of regions), sides as columns
    counts = [sum(lobe_of[r] == l for r in order) for l in lobes]
    fig, axes = plt.subplots(len(lobes), len(vals), figsize=(width, height), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": counts})
    for col, (side, (diffs, sds)) in enumerate(vals.items()):
        for row, l in enumerate(lobes):
            ax = axes[row][col]
            regs = [r for r in order if lobe_of[r] == l]
            data = {p: {r: diffs[p][r].values for r in regs} for p in pairs}
            fsvv._grouped_boxes(ax, data, regs, pairs, fsvv.PAIR_COLORS, horizontal=True, width=box_width,
                                fontsize=font, showfliers=fliers, flier_size=flier_size)
            for v in fsvv.VERSIONS:
                ax.scatter(sds.loc[regs, v], np.arange(len(regs)), marker=fsvv.VERSION_MARKERS[v], color="black",
                           s=marker_size, linewidths=0.8, zorder=5, label=f"SD {v}")
            ax.invert_yaxis()
            ax.set_xscale(xscale)
            if col == 0:
                ax.set_ylabel(l, fontsize=font + 1)
            if row == 0:
                ax.set_title(side, fontsize=title_font)
            if col > 0:
                ax.set_yticklabels([])
    if xmax:
        axes[0][0].set_xlim(right=xmax)
    elif xlim == "whiskers":
        allv = [np.asarray(diffs[p][r].dropna()) for diffs, _ in vals.values() for p in pairs for r in order]
        hi = max(min(v.max(), np.percentile(v, 75) + 1.5 * (np.percentile(v, 75) - np.percentile(v, 25)))
                 for v in allv if len(v))
        hi = max(hi, max(float(sds.max().max()) for _, sds in vals.values()))
        axes[0][0].set_xlim(-0.02 * hi if xscale == "linear" else None, hi * 1.05)
    for ax in axes[-1]:
        ax.set_xlabel(f"{name} ({unit})", fontsize=font + 1)
        ax.tick_params(axis="x", labelsize=font)
    _legend_below(fig, axes[0][0], legend_font, legend_cols)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.3)
    return fig


# --------------------------------------------------------------------------- 3D subcortical renderer
ENIGMA_STRUCTURES = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens-area",
                     "Lateral-Ventricle"]
VIEWS_3D = [("Left", "lateral", 180), ("Left", "medial", 0), ("Right", "medial", 180), ("Right", "lateral", 0)]


def _meshes():
    """{structure name: (vertices, faces)} of fsaverage's subcortical meshes (make_subcortical_meshes.sh)."""
    if "meshes" not in _data:
        d = fsvv.OUT / "fsaverage" / "subcortical"
        paths = sorted(d.glob("*.surf"))
        if not paths:
            raise FileNotFoundError(f"no subcortical meshes (*.surf) in {d}: run scripts/paper/make_subcortical_meshes.sh "
                                    "on the cluster, or copy outputs/fsaverage/subcortical from there")
        _data["meshes"] = {p.stem: nib.freesurfer.read_geometry(str(p)) for p in paths}
    return _data["meshes"]


def _cmap(name):
    """Matplotlib colour map by name, including nilearn's (e.g. cold_hot)."""
    try:
        return plt.get_cmap(name)
    except ValueError:
        from nilearn.plotting import cm as ncm
        return ncm._cmap_d[name]


def _render(ax, side, azim, values, cmap, norm, threshold=None, gray="0.82", elev=0, skip=()):
    """Draw the structures of one hemisphere (side "Left"/"Right", or "" for both), coloured by values[structure]
    (gray when missing or below threshold). All triangles go in one collection so that matplotlib depth-sorts them
    together (one collection per structure is sorted as a block, and hidden structures showed through).
    skip: structure names (without side) not drawn."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    light = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])
    cm = _cmap(cmap)
    pts, tris, cols = [], [], []
    for name, (v, f) in _meshes().items():
        if not name.startswith(side) or name.split("-", 1)[1] in skip:
            continue
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.55 + 0.45 * np.abs(n @ light)
        val = values.get(name, np.nan)
        if np.isfinite(val) and (threshold is None or abs(val) >= threshold):
            base = np.array(cm(norm(val))[:3])
        else:
            base = np.array(matplotlib.colors.to_rgb(gray))
        tris.append(tri)
        cols.append(np.clip(base * shade[:, None], 0, 1))
        pts.append(v)
    ax.add_collection3d(Poly3DCollection(np.vstack(tris), facecolors=np.vstack(cols), linewidths=0))
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _ratio_by_structure(table, pair, metric):
    """Per lateral structure: median between-version |diff| / mean between-subject SD of the two versions, or the
    median relative difference |A - B| / B in %."""
    a, b = pair
    out = {}
    skip = fsvv.incomparable_labels(table) if "FS8" in pair else set()
    for roi in fsvv.ASEG_ROIS:
        A, B = table[f"{roi}_{a}"], table[f"{roi}_{b}"]
        if roi in skip:  # FS8 does not produce this label, or defines it differently
            continue
        if metric == "ratio":
            out[roi] = (A - B).abs().median() / ((A.std() + B.std()) / 2)
        else:
            out[roi] = ((A - B).abs() / B).replace([np.inf, -np.inf], np.nan).median() * 100
    return out


def draw_subcortical_variability(width=12, height=16, font=10, title_font=13, legend_font=10, metric="ratio",
                                 ratio_max=2.0, ratio_cmap="RdBu_r", lower_height=0.3, marker_size=30, **_):
    """Figure 3, ENIGMA-style: per version pair, the 16 structures coloured by the between-version / between-subject
    ratio (or the relative difference, %) in four views; below, the same values for every aseg structure."""
    table = load_data()["table"]
    vals = {fsvv.label(p): _ratio_by_structure(table, p, metric) for p in fsvv.PAIRS}
    if metric == "ratio":
        norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=ratio_max)
        clabel = "median between-version |difference| / between-subject SD (> 1: versions differ more than subjects)"
    else:
        vmax = np.nanpercentile([x for v in vals.values() for x in v.values()], 95)
        norm = matplotlib.colors.Normalize(0, vmax)
        clabel = "median relative difference between versions, |A − B| / B (%)"
    pairs = list(vals)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 1, height_ratios=[1 - lower_height, lower_height], hspace=0.12)
    top = gs[0].subgridspec(len(pairs), len(VIEWS_3D) + 1, width_ratios=[0.5] + [1] * len(VIEWS_3D),
                            wspace=0.0, hspace=0.0)
    for i, p in enumerate(pairs):
        lab = fig.add_subplot(top[i, 0]); lab.set_axis_off()
        lab.text(0.9, 0.5, p, ha="right", va="center", fontsize=font + 1)
        for j, (side, view, azim) in enumerate(VIEWS_3D):
            ax = fig.add_subplot(top[i, j + 1], projection="3d")
            _render(ax, side, azim, {k: v for k, v in vals[p].items()}, ratio_cmap, norm)
            if i == 0:
                ax.set_title(f"{side} {view}", fontsize=font, pad=0)
    sm = plt.cm.ScalarMappable(cmap=_cmap(ratio_cmap), norm=norm)
    cax = fig.add_axes([0.3, 1 - (1 - lower_height) * 0.98 + 0.005, 0.4, 0.008])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(clabel, fontsize=legend_font)
    cb.ax.tick_params(labelsize=font)
    low = fig.add_subplot(gs[1])
    order = [r for r in fsvv.ASEG_PLOT_ORDER]
    x = np.arange(len(order))
    for k, p in enumerate(pairs):
        y = [vals[p].get(r, np.nan) for r in order]
        low.scatter(x + (k - 2.5) * 0.11, y, color=fsvv.PAIR_COLORS[p], s=marker_size * 0.6, label=p, zorder=3)
    if metric == "ratio":
        low.axhline(1, color="black", linewidth=0.8, linestyle="--")
    low.set_xticks(x)
    low.set_xticklabels(order, rotation=90, fontsize=font - 1)
    low.tick_params(axis="y", labelsize=font)
    low.set_ylabel("ratio" if metric == "ratio" else "%", fontsize=font)
    low.set_xlim(-0.6, len(order) - 0.4)
    low.grid(axis="y", linestyle=":", linewidth=0.5)
    low.legend(ncol=6, fontsize=legend_font, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    return fig


# --------------------------------------------------------------------------- Figures 7-8: subcortical maps
def _aseg_2mm():
    """fsaverage's aseg resampled once to 2 mm (nearest neighbour): ~8 times fewer voxels for the 8 panels."""
    if "aseg_2mm" not in _data:
        from nilearn import image
        aseg = nib.load(fsvv.OUT / "fsaverage" / "mri" / "aseg.mgz")
        aseg = nib.Nifti1Image(np.asarray(aseg.dataobj).astype(np.int16), aseg.affine)
        _data["aseg_2mm"] = image.resample_img(aseg, target_affine=np.diag([2.0, 2.0, 2.0]),
                                               interpolation="nearest", force_resample=True, copy_header=True)
    return _data["aseg_2mm"]


# MNI box around the deep gray matter (mm), and the axes of each glass-brain view
SUBCORTICAL_BOX = {"x": (-40, 40), "y": (-50, 35), "z": (-35, 30)}
VIEW_AXES = {"l": ("y", "z"), "r": ("y", "z"), "x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}


def _crop_subcortical(display):
    """Zoom every view of a nilearn glass-brain display on SUBCORTICAL_BOX, keeping each view's orientation."""
    for direction, gax in display.axes.items():
        h, v = VIEW_AXES[direction]
        ax = gax.ax
        for get, set_, coord in ((ax.get_xlim, ax.set_xlim, h), (ax.get_ylim, ax.set_ylim, v)):
            lo, hi = SUBCORTICAL_BOX[coord]
            cur = get()
            set_((hi, lo) if cur[0] > cur[1] else (lo, hi))


def _sign(r, signed):
    """Sign of each result: r for correlations; for the ANCOVA, the direction column of 03 (+1: HC > PD-non-MCI)."""
    if signed:
        return np.sign(r["r"]).values
    return r["direction"].fillna(1).values if "direction" in r else np.ones(len(r))


_SIDES = {"fig7": ("PD-non-MCI > HC", "HC > PD-non-MCI"), "fig8": ("negative r", "positive r")}
_P_LABEL = {"fig7": "p, ANCOVA HC vs PD-non-MCI", "fig8": "p, partial correlation with UPDRS-III"}


def _p_label(x):
    """Tick label of a −log10(p) position, as a p-value (0.05, 0.01, 0.001, 0.0036, 1e-05)."""
    return f"{10 ** -x:.2g}".replace("e-0", "e-")


# −log10 of the Bonferroni threshold of Tables 2-3 (0.05/14): with bonferroni=True, Figures 7-8 colour only the
# structures significant after correction (the threshold slider is then ignored)
BONFERRONI_THRESHOLD = -np.log10(0.05 / fsvv.N_SUBCORTICAL)


def _p_colorbar(fig, rect, cmap, vmax, threshold, signed, label, font, gray="0.82", n_bonf=fsvv.N_SUBCORTICAL,
                sides=("negative r", "positive r")):
    """Colour bar on the −log10(p) scale with p-value tick labels; the band |x| < threshold (p ≥ .05) is gray.

    signed: both halves, `sides` naming the left and right; otherwise the positive half only (unsigned F test).
    n_bonf: number of tests of the Bonferroni tick (None: no tick, e.g. cluster-corrected maps).
    """
    base = _cmap(cmap)
    lo = -vmax if signed else 0.0
    xs = np.linspace(lo, vmax, 512)
    colors = base((xs + vmax) / (2 * vmax))  # same colours as the maps (norm −vmax..vmax)
    colors[np.abs(xs) < threshold] = matplotlib.colors.to_rgba(gray)
    cm = ListedColormap(colors)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo, vmax))
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    bonf = round(-np.log10(0.05 / n_bonf), 3) if n_bonf else None
    # ticks: the threshold, vmax and the integer -log10(p) between them (not closer than 0.6 to another tick)
    pos = [threshold] + [k for k in range(int(np.ceil(threshold)), int(vmax) + 1)
                         if k - threshold >= 0.6 and vmax - k >= 0.6] + [vmax]
    ticks = sorted(set([-x for x in pos] + pos)) if signed else pos
    cb.set_ticks(ticks)
    cb.set_ticklabels([_p_label(abs(x)) for x in ticks], fontsize=font)
    side = f"  (left: {sides[0]}, right: {sides[1]})" if signed else ""
    at_bonf = bool(bonf) and np.isclose(threshold, bonf, atol=1e-3)
    if at_bonf:  # the gray band is the Bonferroni threshold itself
        gray_txt = f"gray: not significant after Bonferroni correction (p ≥ 0.05/{n_bonf} = {0.05 / n_bonf:.2g})"
    else:
        gray_txt = f"gray: p ≥ {_p_label(threshold)}"
        if bonf and threshold < bonf <= vmax:  # Bonferroni threshold: black line and * above the bar (no tick label,
            for x in ([-bonf, bonf] if signed else [bonf]):  # it would collide with the neighbouring ticks)
                cb.ax.axvline(x, color="black", linewidth=1.5)
                cb.ax.text(x, 1.05, "*", transform=cb.ax.get_xaxis_transform(), ha="center", va="bottom",
                           fontsize=font + 2)
            gray_txt += f"; * black line: Bonferroni threshold (p = 0.05/{n_bonf} = {0.05 / n_bonf:.2g})"
    raw = ", uncorrected" if n_bonf else ""  # cluster-wise maps (n_bonf=None) are already corrected
    cb.set_label(f"{label}{side}{raw}; {gray_txt}", fontsize=font)
    return cb


def _pair_layout(view, sides=("Left", "Right"), skip=(), gap=8.0):
    """Subcortical meshes placed for one view, side by side when both hemispheres are drawn (left on the left):
    [(structure, vertices, faces)], in a frame where the camera looks along +x (lateral/medial; screen right = -y,
    up = z) or down -z (superior). Each hemisphere is centred on the origin (in height: z = 0).

    view "lateral": each hemisphere turned so that its lateral face points at the camera (left hemisphere: anterior
    to the left); "medial": its medial face (left hemisphere: anterior to the right); "superior": seen from above.
    """
    key = ("layout", view, tuple(sides), tuple(skip), gap)
    if key in _data:
        return _data[key]
    turn = np.diag([-1.0, -1.0, 1.0])  # 180 degrees about z: brings the other face of a hemisphere to the camera
    axis, sgn = (0, 1.0) if view == "superior" else (1, -1.0)  # axis along which hemispheres sit side by side
    out = []
    for side in sides:
        meshes = {n: vf for n, vf in _meshes().items() if n.startswith(side) and n.split("-", 1)[1] not in skip}
        allv = np.vstack([v for v, _ in meshes.values()])
        centre = (allv.min(0) + allv.max(0)) / 2
        centre[2] = 0.0  # keep both hemispheres at the same height
        rot = turn if view != "superior" and (view == "lateral") == (side == "Right") else np.eye(3)
        placed = {n: ((v - centre) @ rot.T, f) for n, (v, f) in meshes.items()}
        if len(sides) > 1:
            h = sgn * np.vstack([w for w, _ in placed.values()])[:, axis]
            shift = -(gap / 2) - h.max() if side == "Left" else (gap / 2) - h.min()
            for w, _ in placed.values():
                w[:, axis] += sgn * shift
        out += [(n, w, f) for n, (w, f) in placed.items()]
    _data[key] = out
    return out


def _screen(view, anterior):
    """(camera direction from the focal point, up vector, screen-horizontal and vertical coordinate functions)."""
    if view != "superior":
        return np.array([-1.0, 0, 0]), (0, 0, 1), (lambda p: -p[:, 1]), (lambda p: p[:, 2])
    if anterior == "left":  # seen from above, anterior to the left (as the left lateral view), right hemisphere up
        return np.array([0, 0, 1.0]), (1, 0, 0), (lambda p: -p[:, 1]), (lambda p: p[:, 0])
    return np.array([0, 0, 1.0]), (0, 1, 0), (lambda p: p[:, 0]), (lambda p: p[:, 1])  # anterior up, left on left


def _panel_extent(layout, view, anterior):
    """Screen width and height (mm) of a laid-out view, and its centre in 3D."""
    allv = np.vstack([w for _, w, _ in layout])
    _, _, hor, ver = _screen(view, anterior)
    centre = (allv.min(0) + allv.max(0)) / 2
    return np.ptp(hor(allv)), np.ptp(ver(allv)), centre


def _render_vtk(ax, layout, view, values, cmap, norm, threshold=None, gray="0.82", anterior="up", height=None,
                px_per_mm=10):
    """Laid-out meshes rendered with VTK (PyVista off screen: depth buffer, smooth shading, parallel projection) and
    shown as an image in the 2D axes `ax`, in mm, centred, with the axes height `height` mm (default: the image's),
    so that panels of a figure drawn with the same height share one scale."""
    import pyvista as pv
    w, h, centre = _panel_extent(layout, view, anterior)
    direction, up, _, _ = _screen(view, anterior)
    pl = pv.Plotter(off_screen=True, window_size=(max(int(w * px_per_mm), 1), max(int(h * px_per_mm), 1)))
    for name, verts, f in layout:
        mesh = pv.PolyData(verts, np.hstack([np.full((len(f), 1), 3), f]).ravel())
        pl.add_mesh(mesh, color=_structure_color(name, values, cmap, norm, threshold, gray), smooth_shading=True,
                    ambient=0.25, diffuse=0.75, specular=0.1)
    pl.enable_parallel_projection()
    pl.camera_position = [tuple(centre + 500 * direction), tuple(centre), up]
    pl.camera.parallel_scale = h / 2
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    ax.imshow(img, extent=(-w / 2, w / 2, -h / 2, h / 2), interpolation="lanczos")
    H = height or h
    ax.set_xlim(-w / 2, w / 2)
    ax.set_ylim(-H / 2, H / 2)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _structure_color(name, values, cmap, norm, threshold, gray):
    val = values.get(name, np.nan)
    if np.isfinite(val) and (threshold is None or abs(val) >= threshold):
        return np.array(_cmap(cmap)(norm(val))[:3])
    return np.array(matplotlib.colors.to_rgb(gray))


def _render_mpl(ax, layout, view, values, cmap, norm, threshold=None, gray="0.82", anterior="up"):
    """Laid-out meshes drawn with matplotlib's 3D axes (fallback when PyVista is not installed; hidden triangles
    removed by _visible, since matplotlib has no depth buffer)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    light = np.array([0.3, -0.4, 0.87]) if view == "superior" else np.array([-0.75, 0.3, 0.6])
    light = light / np.linalg.norm(light)
    tris, cols = [], []
    for name, w, f in layout:
        t = w[f]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.55 + 0.45 * np.abs(n @ light)
        base = _structure_color(name, values, cmap, norm, threshold, gray)
        tris.append(t)
        cols.append(np.clip(base * shade[:, None], 0, 1))
    allt, allc = np.vstack(tris), np.vstack(cols)
    keep = _visible(allt, view)
    ax.add_collection3d(Poly3DCollection(allt[keep], facecolors=allc[keep], linewidths=0))
    lo, hi = allt.reshape(-1, 3).min(0), allt.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_proj_type("ortho")
    if view != "superior":
        ax.view_init(elev=0, azim=180)
    else:
        ax.view_init(elev=90, azim=180 if anterior == "left" else -90)
    ax.set_axis_off()


def _visible(tris, view, pixel=0.5, eps=1.0):
    """Triangles not hidden by another surface, for the camera of _pair_layout (matplotlib has no depth buffer: it
    paints triangles back to front by their centroid, and hidden structures showed through where surfaces touch).

    Each triangle is sampled at its vertices, edge midpoints and centroid, projected on a `pixel` mm grid, and kept if
    one of its samples is within `eps` mm of the nearest surface in that pixel."""
    if view == "superior":  # camera above: screen (x, y), depth z (larger = nearer)
        uv, depth = tris[..., :2], tris[..., 2]
    else:  # camera on -x: screen (-y, z), depth -x
        uv, depth = np.stack([-tris[..., 1], tris[..., 2]], -1), -tris[..., 0]
    c = tris.mean(1, keepdims=True)
    mids = (tris + np.roll(tris, -1, axis=1)) / 2
    samples = np.concatenate([tris, mids, c], 1)  # (n, 7, 3)
    if view == "superior":
        uv, depth = samples[..., :2], samples[..., 2]
    else:
        uv, depth = np.stack([-samples[..., 1], samples[..., 2]], -1), -samples[..., 0]
    ij = np.floor((uv - uv.reshape(-1, 2).min(0)) / pixel).astype(int)
    shape = ij.reshape(-1, 2).max(0) + 1
    flat = np.ravel_multi_index((ij[..., 0], ij[..., 1]), shape)
    front = np.full(int(np.prod(shape)), -np.inf)
    np.maximum.at(front, flat.ravel(), depth.ravel())
    return (depth >= front[flat] - eps).any(1)


# panels of Figures 7-8 (surface3d), chosen per position by view1..view3: name -> (hemispheres drawn, view)
VIEWS_3D_FIG = {"L lateral": (("Left",), "lateral"), "L medial": (("Left",), "medial"),
                "R lateral": (("Right",), "lateral"), "R medial": (("Right",), "medial"),
                "superior": (("Left", "Right"), "superior")}


def draw_subcortical_3d(fig_id, width=16, row_height=1.6, font=12, colorbar_font=10, vmax=5.0, threshold=1.3,
                        cmap="cold_hot", ventricles=True, superior_anterior="up", bonferroni=False,
                        view1="L lateral", view2="R medial", view3="superior"):
    """Figures 7-8 on the 3D subcortical meshes: rows = versions, columns = baseline and rate of change, three views
    each (view1..view3, from left to right: a key of VIEWS_3D_FIG, e.g. "L lateral", "R medial", "superior" with both
    hemispheres; superior_anterior "up" or "left"), all at the
    same scale; structures coloured by (signed) -log10(p), gray below the threshold. ventricles=False hides the
    lateral ventricles (drawn gray, for context: they are not among the 14 structures of Tables 2-3), which cover
    the caudate and thalamus."""
    if bonferroni:
        threshold = BONFERRONI_THRESHOLD
    skip = () if ventricles else ("Lateral-Ventricle",)
    try:
        import pyvista  # noqa: F401
        vtk = True
    except ImportError:
        vtk = False
    d = load_data()
    res, signed = (d["anc"], False) if fig_id == "fig7" else (d["corr"], True)
    norm = matplotlib.colors.Normalize(-vmax, vmax)
    times = [("base", "Baseline"), ("long", "Rate of change")]
    panels = [(*VIEWS_3D_FIG[name], name) for name in (view1, view2, view3)]
    layouts = [_pair_layout(view, sides, skip) for sides, view, _ in panels]
    ext = [_panel_extent(lay, view, superior_anterior)[:2] for lay, (_, view, _) in zip(layouts, panels)]
    H = max(h for _, h in ext) * 1.04  # common panel height (mm): one scale for every panel
    widths = [w * 1.04 for w, _ in ext]
    fig = plt.figure(figsize=(width, row_height * len(fsvv.VERSIONS) + 1.4))
    gs = fig.add_gridspec(len(fsvv.VERSIONS), 1 + 2 * len(widths), width_ratios=[0.25 * max(widths)] + widths * 2,
                          wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.9, bottom=0.12)
    for i, v in enumerate(fsvv.VERSIONS):
        lab = fig.add_subplot(gs[i, 0]); lab.set_axis_off()
        lab.text(0.9, 0.5, v, ha="right", va="center", fontsize=font)
        for t, (time, title) in enumerate(times):
            r = res[(res["version"] == v) & (res["time"] == time)]
            sign = _sign(r, signed)
            # only the structures of Tables 2-3 (and of the Bonferroni count) are coloured; the ventricles are context
            vals = {roi: -np.log10(max(p, 1e-300)) * s for roi, p, s in zip(r["roi"], r["p"], sign)
                    if np.isfinite(p) and roi in fsvv.CLINICAL_TABLE_ROIS}
            for j, (lay, (_, view, name)) in enumerate(zip(layouts, panels)):
                cell = gs[i, 1 + t * len(widths) + j]
                if vtk:
                    ax = fig.add_subplot(cell)
                    _render_vtk(ax, lay, view, vals, cmap, norm, threshold=threshold, anterior=superior_anterior,
                                height=H)
                else:
                    ax = fig.add_subplot(cell, projection="3d")
                    _render_mpl(ax, lay, view, vals, cmap, norm, threshold=threshold, anterior=superior_anterior)
                if i == 0:
                    ax.set_title(name, fontsize=font - 2, pad=2)
        if i == 0:
            for t, (_, title) in enumerate(times):
                x0 = gs[0, 1 + t * len(widths)].get_position(fig).x0
                x1 = gs[0, (t + 1) * len(widths)].get_position(fig).x1
                fig.text((x0 + x1) / 2, 0.96, title, ha="center", fontsize=font + 1)
    _p_colorbar(fig, [0.3, 0.05, 0.4, 0.02], cmap, vmax, threshold, signed=True, label=_P_LABEL[fig_id],
                font=colorbar_font, sides=_SIDES[fig_id])
    return fig


def draw_subcortical(fig_id, style="glass_ortho", width=9, row_height=2.0, font=12, colorbar_font=10, vmax=5.0,
                     threshold=1.3, cmap="cold_hot", cut_y="10,-5,-20", ventricles=True, superior_anterior="up",
                     bonferroni=False, view1="L lateral", view2="R medial", view3="superior"):
    """Figure 7 (ANCOVA HC vs PD-non-MCI) or 8 (correlation with UPDRS-III); style glass_subcortical (split
    view cropped on the deep gray matter) / glass_ortho / glass_xz / glass_lyrz / mni_slices / fsaverage_slices."""
    if style == "surface3d":
        return draw_subcortical_3d(fig_id, width=width, row_height=row_height, font=font,
                                   colorbar_font=colorbar_font, vmax=vmax, threshold=threshold, cmap=cmap,
                                   ventricles=ventricles, superior_anterior=superior_anterior,
                                   bonferroni=bonferroni, view1=view1, view2=view2, view3=view3)
    if bonferroni:
        threshold = BONFERRONI_THRESHOLD
    from nilearn import datasets, plotting
    d = load_data()
    res, signed = (d["anc"], False) if fig_id == "fig7" else (d["corr"], True)
    aseg = _aseg_2mm()
    data = np.asarray(aseg.dataobj)
    label_id = {f"{side}-{s}": v[i] for s, v in SUBCORTICAL_IDS.items() for i, side in enumerate(("Left", "Right"))}
    cuts = tuple(float(c) for c in str(cut_y).split(","))

    def volume(r):
        vol = np.zeros(data.shape, np.float32)
        sign = _sign(r, signed)
        for roi, p, s in zip(r["roi"], r["p"], sign):
            if roi in label_id and np.isfinite(p):
                vol[data == label_id[roi]] = -np.log10(max(p, 1e-300)) * s
        empty = not (np.abs(vol) >= threshold).any()
        if empty:  # nilearn shrinks the glass brain of an empty map: plot one voxel, removed after drawing
            vol[tuple(np.argwhere(data == label_id["Left-Thalamus"])[0])] = vmax
        return nib.Nifti1Image(vol, aseg.affine), empty

    times = [("base", "Baseline"), ("long", "Rate of change")]
    fig, axes = plt.subplots(len(fsvv.VERSIONS), 2, figsize=(width, row_height * len(fsvv.VERSIONS) + 1.2))
    for row, v in zip(axes, fsvv.VERSIONS):
        for ax, (time, title) in zip(row, times):
            img, empty = volume(res[(res["version"] == v) & (res["time"] == time)])
            common = dict(axes=ax, figure=fig, threshold=threshold, vmax=vmax, cmap=cmap, colorbar=False,
                          annotate=False)
            if style == "glass_subcortical":  # split view (left sagittal, coronal, axial, right sagittal), cropped
                disp = plotting.plot_glass_brain(img, display_mode="lyrz", plot_abs=False, black_bg=False, **common)
                _crop_subcortical(disp)
            elif style.startswith("glass"):
                disp = plotting.plot_glass_brain(img, display_mode=style.split("_")[1], plot_abs=False,
                                                 black_bg=False, **common)
            if style.startswith("glass") and empty:  # remove the placeholder voxel (the outlines are patches)
                for gax in disp.axes.values():
                    for im in list(gax.ax.images):
                        im.remove()
            if not style.startswith("glass"):
                if empty:
                    img = nib.Nifti1Image(np.zeros(data.shape, np.float32), aseg.affine)
                bg = (datasets.load_mni152_template(resolution=1) if style == "mni_slices"
                      else str(fsvv.OUT / "fsaverage" / "mri" / "T1.mgz"))
                plotting.plot_stat_map(img, bg_img=bg, display_mode="y", cut_coords=cuts, draw_cross=False,
                                       black_bg=False, dim=-0.3 if style == "mni_slices" else 0, **common)
            if v == fsvv.VERSIONS[0]:
                ax.set_title(title, fontsize=font)
        row[0].text(-0.04, 0.5, v, transform=row[0].transAxes, rotation=90, va="center", ha="right", fontsize=font)
    _p_colorbar(fig, [0.3, 0.04, 0.4, 0.018], cmap, vmax, threshold, signed=True, label=_P_LABEL[fig_id],
                font=colorbar_font, sides=_SIDES[fig_id])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.1, wspace=0.03, hspace=0.05)
    return fig


# --------------------------------------------------------------------------- Figure 1: p-value grid, PPMI | QPN
def _p_grid(tests, keys, kind):
    """(levels 0-4: Bonferroni level reached, absent-in-FS8 flags), rows = keys, columns = the 6 pairs."""
    t = tests[tests["kind"] == kind].set_index(["key", "pair"])
    p = np.array([[t.loc[(k, fsvv.tag(q)), "p"] for q in fsvv.PAIRS] for k in keys])
    absent = np.array([[bool(t.loc[(k, fsvv.tag(q)), "absent_in_FS8"]) if "absent_in_FS8" in t else False
                        for q in fsvv.PAIRS] for k in keys])
    levels = (p[..., None] < np.array(P_LEVELS) / fsvv.N_TESTS[kind]).sum(-1)
    return levels, absent


def _draw_p_grid(ax, levels, absent, labels, title, font, title_font, grid_lw, cross_size):
    ax.imshow(levels, cmap=ListedColormap(P_COLORS), vmin=-0.5, vmax=4.5, aspect="auto", interpolation="nearest")
    for i, j in zip(*np.nonzero(absent)):
        ax.text(j, i, "×", ha="center", va="center", fontsize=cross_size)
    ax.set_xticks(range(len(fsvv.PAIRS)))
    ax.set_xticklabels([fsvv.label(q).replace("FS ", "") for q in fsvv.PAIRS], rotation=90, fontsize=font)
    ax.set_yticks(range(levels.shape[0]))
    if labels is not None:
        ax.set_yticklabels(labels, fontsize=font)
    else:  # hide, not set_yticklabels([]): the panels share y, which would clear the labels of the first one too
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xticks(np.arange(-0.5, len(fsvv.PAIRS)), minor=True)
    ax.set_yticks(np.arange(-0.5, levels.shape[0]), minor=True)
    ax.grid(which="minor", color="0.8", linewidth=grid_lw)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=title_font)


def draw_fig1(width=5.2, height=8, font=7, title_font=8, suptitle_font=9, legend_font=7, grid_lw=0.3, cross_size=6,
              legend_cols=5):
    """Figure 1: Bonferroni level of the paired t-tests between versions, subcortical volumes; PPMI (all images) and
    the QPN replication sample (PD) side by side; × label not produced by FS8."""
    d = load_data()
    if "qpn_tests" not in d:
        d["qpn_tests"] = pd.read_csv(fsvv.OUT / "qpn_version_paired_ttests.csv", dtype={"pair": str})
        d["n_qpn_pd"] = int(pd.read_csv(fsvv.OUT / "qpn_demographics.csv", index_col="dx_group").loc["PD", "n"])
    rois = fsvv.ASEG_PLOT_ORDER
    samples = [(d["tests"], f"PPMI ({len(d['table'])} images)"), (d["qpn_tests"], f"QPN, PD ({d['n_qpn_pd']} images)")]
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    for ax, (tests, title) in zip(axes, samples):
        levels, absent = _p_grid(tests, rois, "vol")
        _draw_p_grid(ax, levels, absent, rois if ax is axes[0] else None, title, font, title_font, grid_lw, cross_size)
    fig.suptitle("Subcortical volume", fontsize=suptitle_font)
    fig.legend([Patch(facecolor=c, edgecolor="0.6") for c in P_COLORS], P_LABELS, ncol=legend_cols,
               fontsize=legend_font, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0))
    rows = int(np.ceil(len(P_COLORS) / legend_cols))
    fig.tight_layout(rect=(0, rows * legend_font * 1.8 / 72 / height, 1, 1))
    return fig


# --------------------------------------------------------------------------- Figures 9-12: vertex-wise maps
GLM_FIGURES = {"fig9": ("corr", "base", "Correlation between UPDRS-III and cortical thickness (baseline)"),
               "fig10": ("corr", "long", "Correlation between the change in UPDRS-III and the rate of change in "
                                         "thickness"),
               "fig11": ("group", "base", "HC vs PD-non-MCI, cortical thickness (baseline)"),
               "fig12": ("group", "long", "HC vs PD-non-MCI, rate of change in cortical thickness")}
SURF_VIEWS = [("lh", "lateral"), ("lh", "medial"), ("rh", "lateral"), ("rh", "medial")]


def _surface(hemi):
    """fsaverage inflated surface (vertices, faces) and sulcal depth of one hemisphere."""
    key = ("surf", hemi)
    if key not in _data:
        d = fsvv.OUT / "fsaverage"
        v, f = nib.freesurfer.read_geometry(str(d / f"{hemi}.inflated"))
        _data[key] = (v, f, nib.freesurfer.read_morph_data(str(d / f"{hemi}.sulc")))
    return _data[key]


def _glm_sig(version, analysis, time, hemi):
    """Cluster-corrected map (cache.th13.abs.sig.masked.mgh: signed vertex −log10 p inside significant clusters)."""
    key = ("sig", version, analysis, time, hemi)
    if key not in _data:
        f = fsvv.glm_contrast_dir(version, analysis, time, hemi) / "cache.th13.abs.sig.masked.mgh"
        _data[key] = np.asarray(nib.load(str(f)).dataobj).ravel()
    return _data[key]


def _render_surface_vtk(ax, hemi, view, sig, cmap, vmax, threshold, px=700):
    """Inflated hemisphere (sulcal depth in gray) with the map overlaid where |sig| >= threshold; parallel
    projection, rendered with VTK and shown as an image in the 2D axes `ax`."""
    import pyvista as pv
    v, f, sulc = _surface(hemi)
    rgb = np.where(sulc[:, None] > 0, 0.55, 0.78) * np.ones((1, 3))  # sulci dark, gyri light
    show = np.abs(sig) >= threshold
    rgb[show] = _cmap(cmap)((np.clip(sig[show], -vmax, vmax) + vmax) / (2 * vmax))[:, :3]
    mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    mesh.point_data["rgb"] = (rgb * 255).astype(np.uint8)
    lo, hi = v.min(0), v.max(0)
    w, h = hi[1] - lo[1], hi[2] - lo[2]
    pl = pv.Plotter(off_screen=True, window_size=(px, max(int(px * h / w), 1)))
    pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.05)
    centre = (lo + hi) / 2
    side = -1 if (hemi == "lh") == (view == "lateral") else 1  # camera on -x: lh lateral and rh medial
    pl.enable_parallel_projection()
    pl.camera_position = [tuple(centre + [side * 500, 0, 0]), tuple(centre), (0, 0, 1)]
    pl.camera.parallel_scale = h / 2 * 1.02
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()


def draw_glm(fig_id, width=10, row_height=1.9, font=10, title_font=10, colorbar_font=8, vmax=5.0, threshold=1.3,
             cmap="cold_hot", title=True, wspace=0.0, hspace=0.0):
    """Figures 9-12: vertex-wise cluster-corrected maps on the fsaverage inflated surface, one row per version,
    lateral and medial views of each hemisphere; PyVista when installed, else nilearn (matplotlib)."""
    analysis, time, suptitle = GLM_FIGURES[fig_id]
    try:
        import pyvista  # noqa: F401
        vtk = True
    except ImportError:
        vtk = False
    fig = plt.figure(figsize=(width, row_height * len(fsvv.VERSIONS) + 1))
    gs = fig.add_gridspec(len(fsvv.VERSIONS), len(SURF_VIEWS), left=0.05, right=0.99,
                          top=0.9 if title else 0.95, bottom=0.12, wspace=wspace, hspace=hspace)
    for i, v in enumerate(fsvv.VERSIONS):
        for j, (hemi, view) in enumerate(SURF_VIEWS):
            sig = _glm_sig(v, analysis, time, hemi)
            if vtk:
                ax = fig.add_subplot(gs[i, j])
                _render_surface_vtk(ax, hemi, view, sig, cmap, vmax, threshold)
            else:
                from nilearn import plotting
                ax = fig.add_subplot(gs[i, j], projection="3d")
                d = fsvv.OUT / "fsaverage"
                plotting.plot_surf_stat_map(str(d / f"{hemi}.inflated"), sig, hemi="left" if hemi == "lh" else "right",
                                            view=view, bg_map=str(d / f"{hemi}.sulc"), axes=ax, figure=fig,
                                            colorbar=False, threshold=threshold, vmax=vmax, cmap=cmap)
                ax.set_rasterized(True)
            if i == 0:
                ax.set_title(f"{hemi} {view}", fontsize=font)
            if j == 0:
                ax.text(-0.04, 0.5, v, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=font)
    sides = ("PD-non-MCI > HC", "HC > PD-non-MCI") if analysis == "group" else ("negative r", "positive r")
    _p_colorbar(fig, [0.3, 0.05, 0.4, 0.02], cmap, vmax, threshold, signed=True, label="cluster-wise p",
                font=colorbar_font, n_bonf=None, sides=sides)
    if title:
        fig.suptitle(suptitle, fontsize=title_font)
    return fig


# --------------------------------------------------------------------------- Supplementary figures
def _boxes(ax, data, positions, width, color, fliers, flier_size, vert=True):
    bp = ax.boxplot(data, positions=positions, widths=width, vert=vert, patch_artist=True, showfliers=fliers,
                    manage_ticks=False, medianprops=dict(color="black", linewidth=0.8),
                    flierprops=dict(markersize=flier_size, alpha=0.4, markeredgewidth=0, markerfacecolor=color),
                    whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6), boxprops=dict(linewidth=0.6))
    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_alpha(0.85)


def _version_legend(fig, font, cols, rows_frac=0.0):
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=fsvv.VERSION_COLORS[v]) for v in fsvv.VERSIONS], fsvv.VERSIONS,
               ncol=cols, fontsize=font, frameon=False, loc="lower center", bbox_to_anchor=(0.5, rows_frac))


def _pair_legend(fig, font, cols, extra=((), ())):
    pairs = [fsvv.label(p) for p in fsvv.PAIRS]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=fsvv.PAIR_COLORS[p]) for p in pairs] + list(extra[0]),
               pairs + list(extra[1]), ncol=cols, fontsize=font, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0))


def _legend_rect(n, cols, font, height):
    return int(np.ceil(n / cols)) * font * 1.8 / 72 / height


def _estimates_panel(ax, kind, hemi, font, title_font, box_width, fliers, flier_size, yscale):
    table = load_data()["table"]
    if kind == "aseg":
        regions, col, unit, title = fsvv.ASEG_PLOT_ORDER, (lambda v, r: f"{r}_{v}"), fsvv.MEASURE_UNITS["vol"], \
            "Subcortical volumes"
    else:
        regions, col, unit = fsvv.CORTICAL_ROIS, (lambda v, r: f"{r}_{hemi}_{kind}_{v}"), fsvv.MEASURE_UNITS[kind]
        title = f"{fsvv.MEASURE_NAMES[kind]} ({hemi})"
    n = len(fsvv.VERSIONS)
    w = box_width / n
    for k, v in enumerate(fsvv.VERSIONS):
        data = [table[col(v, r)].dropna().values for r in regions]
        _boxes(ax, data, np.arange(len(regions)) + (k - (n - 1) / 2) * w, w * 0.9, fsvv.VERSION_COLORS[v], fliers,
               flier_size)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=90, fontsize=font)
    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_yscale(yscale)
    ax.tick_params(axis="y", labelsize=font)
    ax.set_ylabel(unit, fontsize=font)
    ax.set_title(title, fontsize=title_font)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)


def draw_estimates(fig_id, width=12, height=4, font=7, title_font=10, legend_font=8, box_width=0.8, fliers=True,
                   flier_size=1.5, yscale="linear", legend_cols=4):
    """Figures S1 (subcortical) and S2 (cortical, one panel per measure and hemisphere): estimates per version."""
    panels = [("aseg", None)] if fig_id == "figS1" else [(k, h) for k in fsvv.CORTICAL_MEASURES for h in fsvv.HEMIS]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, height), squeeze=False)
    for ax, (kind, hemi) in zip(axes[:, 0], panels):
        _estimates_panel(ax, kind, hemi, font, title_font, box_width, fliers, flier_size, yscale)
    _version_legend(fig, legend_font, legend_cols)
    fig.tight_layout(rect=(0, _legend_rect(4, legend_cols, legend_font, height), 1, 1))
    return fig


def _dice_long():
    if "dice_long" not in _data:
        _data["dice_long"] = fsvv.seg_dice_long(fsvv.read_seg_dice(load_data()["table"].index))
    return _data["dice_long"]


def draw_seg_dice(width=14, height=16, font=7, title_font=10, legend_font=8, box_width=0.8, fliers=True,
                  flier_size=1.5, xmin=0.0, legend_cols=6):
    """Figure S4: segmentation Dice between versions per region (boxes per version pair), one panel per hemisphere."""
    d = _dice_long()
    pairs = [fsvv.label(p) for p in fsvv.PAIRS]
    fig, axes = plt.subplots(1, len(fsvv.HEMIS), figsize=(width, height))
    if "dice_groups" not in _data:  # (hemi, comparison, roi) -> values, grouped once (filtering per box is slow)
        _data["dice_groups"] = {k: g.values for k, g in d.groupby(["hemi", "comparison", "roi"])["dice"]}
    groups = _data["dice_groups"]
    for ax, hemi in zip(axes, fsvv.HEMIS):
        regions = sorted({r for h, _, r in groups if h == hemi})
        w = box_width / len(pairs)
        for k, p in enumerate(pairs):
            data = [groups.get((hemi, p, r), np.array([])) for r in regions]
            _boxes(ax, data, np.arange(len(regions)) + (k - (len(pairs) - 1) / 2) * w, w * 0.9, fsvv.PAIR_COLORS[p],
                   fliers, flier_size, vert=False)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions, fontsize=font)
        ax.set_ylim(len(regions) - 0.4, -0.6)
        ax.set_xlim(xmin, 1)
        ax.tick_params(axis="x", labelsize=font)
        ax.set_xlabel("Sørensen–Dice coefficient", fontsize=font + 1)
        ax.set_title(f"Segmentation overlap between versions ({hemi})", fontsize=title_font)
        ax.grid(axis="x", linestyle=":", linewidth=0.5)
    _pair_legend(fig, legend_font, legend_cols)
    fig.tight_layout(rect=(0, _legend_rect(len(pairs), legend_cols, legend_font, height), 1, 1))
    return fig


TYPES = {  # Figure S5 (as in 08_variability_by_type)
    "Large structures": ["Cerebellum-White-Matter", "Cerebellum-Cortex", "Brain-Stem", "CC_Posterior",
                         "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"],
    "Subcortical gray matter": ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
                                "Accumbens-area", "VentralDC"],
    "CSF spaces": ["Lateral-Ventricle", "Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "5th-Ventricle", "CSF"],
}


def _by_type_subcortical():
    """(relative differences per image, between-subject CV) of the bilateral structures (08_variability_by_type)."""
    if "by_type_sub" in _data:
        return _data["by_type_sub"]
    table = load_data()["table"]
    absent = set(fsvv.absent_labels(table, "FS8"))

    def bilateral(st, v):
        cols = [f"{s}-{st}_{v}" for s in ("Left", "Right")]
        return table[cols].sum(axis=1, min_count=2) if cols[0] in table else table[f"{st}_{v}"]

    def produced_by_all(st):
        return not ({f"{s}-{st}" for s in ("Left", "Right")} | {st}) & absent

    vol = {(st, v): bilateral(st, v) for ty in TYPES.values() for st in ty for v in fsvv.VERSIONS}
    inc = fsvv.incomparable_labels(table)
    redefined = {st for ty in TYPES.values() for st in ty if (st in inc or f"Left-{st}" in inc) and produced_by_all(st)}
    absent |= {f"{s}-{st}" for st in redefined for s in ("Left", "Right")} | redefined
    rows = []
    for ty, structs in TYPES.items():
        for st in structs:
            for pair in fsvv.PAIRS:
                if "FS8" in pair and not produced_by_all(st):
                    continue
                a, b = vol[(st, pair[0])], vol[(st, pair[1])]
                rel = ((a - b).abs() / b * 100).replace([np.inf, -np.inf], np.nan)
                rows.append(pd.DataFrame({"type": ty, "structure": st, "pair": fsvv.label(pair), "rel_diff": rel}))
    rel_sub = pd.concat(rows, ignore_index=True)
    cv_sub = pd.DataFrame([{"type": ty, "structure": st, "version": v,
                            "cv": vol[(st, v)].std() / vol[(st, v)].mean() * 100}
                           for ty, structs in TYPES.items() for st in structs for v in fsvv.VERSIONS
                           if v != "FS8" or produced_by_all(st)])
    _data["by_type_sub"] = (rel_sub, cv_sub)
    return rel_sub, cv_sub


def draw_by_type_subcortical(width=15, height=6, font=8, title_font=10, legend_font=7, box_width=0.8, fliers=False,
                             flier_size=1.5, marker_size=16, yscale="log", legend_cols=5):
    """Figure S5: relative difference between versions (boxes per pair) and between-subject CV (markers) of the
    bilateral subcortical structures, by type."""
    rel_sub, cv_sub = _by_type_subcortical()
    pairs = [fsvv.label(p) for p in fsvv.PAIRS]
    fig, axes = plt.subplots(1, len(TYPES), figsize=(width, height),
                             gridspec_kw={"width_ratios": [len(s) for s in TYPES.values()]})
    for ax, (ty, structs) in zip(axes, TYPES.items()):
        w = box_width / len(pairs)
        for j, pair in enumerate(pairs):
            data = [rel_sub.loc[(rel_sub["structure"] == st) & (rel_sub["pair"] == pair), "rel_diff"].dropna().values
                    for st in structs]
            pos = np.arange(len(structs)) - box_width / 2 + w * (j + 0.5)
            keep = [k for k, x in enumerate(data) if len(x)]
            _boxes(ax, [data[k] for k in keep], pos[keep], w * 0.9, fsvv.PAIR_COLORS[pair], fliers, flier_size)
        for v in fsvv.VERSIONS:
            cv = cv_sub[(cv_sub["type"] == ty) & (cv_sub["version"] == v)].set_index("structure")["cv"]
            x = [k for k, st in enumerate(structs) if st in cv]
            ax.scatter(x, [cv[structs[k]] for k in x], marker=fsvv.VERSION_MARKERS[v], color="black", s=marker_size,
                       zorder=5, label=f"between-subject CV {v}")
        ax.set_xticks(range(len(structs)))
        ax.set_xticklabels([s.replace("_", " ") for s in structs], rotation=45, ha="right", fontsize=font)
        ax.set_xlim(-0.6, len(structs) - 0.4)
        ax.set_yscale(yscale)
        ax.tick_params(axis="y", labelsize=font)
        ax.set_title(ty, fontsize=title_font)
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.5)
    axes[0].set_ylabel("relative difference between versions, |A − B| / B (%)", fontsize=font + 1)
    h, l = axes[0].get_legend_handles_labels()
    _pair_legend(fig, legend_font, legend_cols, (h, l))
    fig.tight_layout(rect=(0, _legend_rect(len(pairs) + len(h), legend_cols, legend_font, height), 1, 1))
    return fig


def _by_lobe_cortical():
    """Per image, the median relative difference (%) of each lobe's regions, bilateral (08_variability_by_type)."""
    if "by_lobe" in _data:
        return _data["by_lobe"]
    table = load_data()["table"]
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
                med = rel[[r for r in fsvv.CORTICAL_ROIS if lobe_of[r] == lobe]].median(axis=1)
                rows.append(pd.DataFrame({"kind": kind, "pair": fsvv.label(pair), "lobe": lobe, "rel_diff": med}))
    _data["by_lobe"] = (pd.concat(rows, ignore_index=True), lobes)
    return _data["by_lobe"]


def draw_by_lobe_cortical(width=15, height=4.5, font=8, title_font=10, legend_font=7, box_width=0.8, fliers=False,
                          flier_size=1.5, yscale="linear", legend_cols=6):
    """Figure S6: relative difference between versions per lobe (median over the lobe's regions), bilateral."""
    rel, lobes = _by_lobe_cortical()
    pairs = [fsvv.label(p) for p in fsvv.PAIRS]
    fig, axes = plt.subplots(1, len(fsvv.CORTICAL_MEASURES), figsize=(width, height))
    for ax, kind in zip(axes, fsvv.CORTICAL_MEASURES):
        d = rel[rel["kind"] == kind]
        w = box_width / len(pairs)
        for j, pair in enumerate(pairs):
            data = [d.loc[(d["lobe"] == l) & (d["pair"] == pair), "rel_diff"].dropna().values for l in lobes]
            _boxes(ax, data, np.arange(len(lobes)) - box_width / 2 + w * (j + 0.5), w * 0.9, fsvv.PAIR_COLORS[pair],
                   fliers, flier_size)
        ax.set_xticks(range(len(lobes)))
        ax.set_xticklabels(lobes, fontsize=font)
        ax.set_xlim(-0.6, len(lobes) - 0.4)
        ax.set_yscale(yscale)
        ax.tick_params(axis="y", labelsize=font)
        ax.set_title(f"{fsvv.MEASURE_NAMES[kind]} (bilateral)", fontsize=title_font)
        ax.grid(axis="y", color="0.85", linewidth=0.5)
    axes[0].set_ylabel("relative difference between versions (%),\nmedian over the lobe's regions", fontsize=font + 1)
    _pair_legend(fig, legend_font, legend_cols)
    fig.tight_layout(rect=(0, _legend_rect(len(pairs), legend_cols, legend_font, height), 1, 1))
    return fig


def _place_legend(fig, pos, dx=0.0, dy=0.0):
    """Move the figure legend(s) drawn below the figure to `pos` (LEGEND_POSITIONS), shifted by (dx, dy).

    above / right / left: outside the axes (one column on the sides); the saved figure (bbox_inches="tight") grows to
    include it. inside <corner>: in that corner of the first panel. The space kept below for the legend is given back
    to the axes when the layout allows it (all axes on a grid).
    dx, dy: shift, in fractions of the figure (below / above / right / left) or of the first panel (inside); > 0 moves
    right / up.
    """
    if not fig.legends or (pos == "below" and not dx and not dy):
        return fig
    to_fig = fig.transFigure.inverted()
    for old in list(fig.legends):  # rebuilt: an existing legend does not re-lay out its columns
        texts, title = old.get_texts(), old.get_title()
        side = pos not in ("above", "below")  # sides and inside: one column, title on two lines
        kw = dict(ncol=1 if side else old._ncols, fontsize=texts[0].get_fontsize() if texts else None,
                  frameon=old.get_frame_on(), title_fontsize=title.get_fontsize(),
                  title=title.get_text().replace("; ", "\n") if side else title.get_text())
        if pos.startswith("inside "):
            # axes coordinates, not fig.axes[0].bbox: a Bbox anchor blows up under bbox_inches="tight" (out of memory)
            kw.update(loc=pos.removeprefix("inside "), bbox_to_anchor=(dx, dy, 1, 1),
                      bbox_transform=fig.axes[0].transAxes)
        else:
            if pos == "below":  # where the figure function put it
                b = old.get_bbox_to_anchor().transformed(to_fig)
                loc, anchor = old._loc, (b.x0, b.y0)
            else:
                loc, anchor = {"above": ("lower center", (0.5, 1.0)), "right": ("center left", (1.0, 0.5)),
                               "left": ("center right", (0.0, 0.5))}[pos]
            kw.update(loc=loc, bbox_to_anchor=(anchor[0] + dx, anchor[1] + dy), bbox_transform=fig.transFigure)
        handles = old.legend_handles
        old.remove()
        fig.legend(handles, [t.get_text() for t in texts], **kw)
    if pos != "below" and all(ax.get_subplotspec() is not None for ax in fig.axes):
        fig.tight_layout()
    return fig


def draw(fig_id, legend_pos="below", legend_dx=0.0, legend_dy=0.0, **params):
    return _place_legend(_draw(fig_id, **params), legend_pos, legend_dx, legend_dy)


def _draw(fig_id, **params):
    if fig_id == "fig1":
        return draw_fig1(**params)
    if fig_id in GLM_FIGURES:
        return draw_glm(fig_id, **params)
    if fig_id in ("figS1", "figS2"):
        return draw_estimates(fig_id, **params)
    if fig_id == "figS4":
        return draw_seg_dice(**params)
    if fig_id == "figS5":
        return draw_by_type_subcortical(**params)
    if fig_id == "figS6":
        return draw_by_lobe_cortical(**params)
    if fig_id == "fig2":
        return draw_fig2(**params)
    if fig_id in ("fig3a", "fig3b"):
        return draw_variability("fig3", part=1 if fig_id == "fig3a" else 2, **params)
    if fig_id == "fig3":
        p = dict(params)
        if p.pop("layout", "boxes") == "enigma":
            return draw_subcortical_variability(**p)
        for k in ("metric", "ratio_max", "ratio_cmap", "lower_height"):
            p.pop(k, None)
        return draw_variability(fig_id, **p)
    if fig_id in ("fig4", "fig5", "fig6"):
        return draw_cortical(fig_id, **params)
    return draw_subcortical(fig_id, **params)
