"""Paper figures with every visual setting as a parameter (Figures 2-8), shared by the tuning notebook
(09_figure_tuning), scripts/paper/figure_variants.py and, once the settings are chosen, 02/06.

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
PAPER_NAMES = {"fig2": "fig2_pvalues_cortical", "fig3": "variability_subcortical", "fig4": "variability_vol",
               "fig5": "variability_surf", "fig6": "variability_ct", "fig7": "fig_subcortical_group",
               "fig8": "fig_subcortical_corr"}
KIND_OF = {"fig4": "vol", "fig5": "surf", "fig6": "ct"}
DEFAULTS = {
    "fig2": dict(width=16, height=22, font=10, title_font=12, legend_font=10, grid_lw=0.3),
    "fig3": dict(layout="enigma", metric="ratio", width=12, height=16, font=10, title_font=13, legend_font=10,
                 fliers=False, xscale="linear", xlim="whiskers", xmax=None, marker_size=30, box_width=0.8,
                 legend_cols=5, ratio_max=2.0, ratio_cmap="RdBu_r", lower_height=0.3),
    "fig4": dict(layout="ratio_heatmap", hemis="lh+rh", width=12, height=18, font=9, title_font=12, legend_font=9,
                 fliers=False, xscale="linear", xlim="whiskers", xmax=None, marker_size=30, box_width=0.8,
                 legend_cols=5, ratio_max=2.0, ratio_cmap="RdBu_r", lobe_split=True, preview_part=1),
    "fig7": dict(style="glass_subcortical", width=9, row_height=2.0, font=12, colorbar_font=10, vmax=5.0, threshold=1.3,
                 cmap="cold_hot", cut_y="10,-5,-20"),
}
DEFAULTS["fig5"] = dict(DEFAULTS["fig4"])
DEFAULTS["fig6"] = dict(DEFAULTS["fig4"])
DEFAULTS["fig8"] = dict(DEFAULTS["fig7"])

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


def load_params():
    saved = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    return {k: {**v, **saved.get(k, {})} for k, v in DEFAULTS.items()}


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


def draw_variability(fig_id, width, height, font=10, title_font=13, legend_font=10, fliers=False, xscale="linear",
                     xlim="whiskers", xmax=None, marker_size=30, box_width=0.8, legend_cols=5):
    """Figure 3 (fig_id "fig3": subcortical) or 4-6 ("fig4"/"fig5"/"fig6": cortical lh and rh)."""
    table = load_data()["table"]
    kw = dict(fontsize=font, showfliers=fliers, xscale=xscale, xlim=xlim if xlim != "auto" else None,
              marker_size=marker_size, legend=False, box_width=box_width)
    if fig_id == "fig3":
        fig, ax = plt.subplots(figsize=(width, height))
        fsvv.plot_variability(table, "aseg", ax=ax, **kw)
        _style_axes(ax, title_font, xmax, xscale)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True)
        for ax, hemi in zip(axes, fsvv.HEMIS):
            fsvv.plot_variability(table, KIND_OF[fig_id], hemi, ax=ax, **kw)
        lims = [a.get_xlim() for a in axes]
        axes[0].set_xlim(min(l[0] for l in lims), max(l[1] for l in lims))
        for ax in axes:
            _style_axes(ax, title_font, xmax, xscale)
    _legend_below(fig, axes[0], legend_font, legend_cols)
    fig.tight_layout(rect=(0, 0.06 if fig_id == "fig3" else 0.035, 1, 1))
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
                  preview_part=1):
    """Figures 4-6 (volume, area, thickness): between-version |difference| vs between-subject SD.

    layout: "ratio_heatmap" (median |diff| / mean SD of the two versions, regions x pairs, grouped by lobe),
    "dots" (median and IQR per pair, SD markers), "boxes_by_lobe" (boxes, one panel per lobe), "boxes" (original).
    """
    kind = KIND_OF[fig_id]
    if layout == "boxes":
        return draw_variability(fig_id, width, height, font, title_font, legend_font, fliers, xscale, xlim, xmax,
                                marker_size, box_width, legend_cols)
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
                                fontsize=font, showfliers=fliers)
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
        _data["meshes"] = {p.stem: nib.freesurfer.read_geometry(str(p)) for p in sorted(d.glob("*.surf"))}
    return _data["meshes"]


def _cmap(name):
    """Matplotlib colour map by name, including nilearn's (e.g. cold_hot)."""
    try:
        return plt.get_cmap(name)
    except ValueError:
        from nilearn.plotting import cm as ncm
        return ncm._cmap_d[name]


def _render(ax, side, azim, values, cmap, norm, threshold=None, gray="0.82"):
    """Draw the structures of one hemisphere, coloured by values[structure] (gray when missing or below threshold)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    light = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])
    cm = _cmap(cmap)
    pts = []
    for name, (v, f) in _meshes().items():
        if not name.startswith(side):
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
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(base * shade[:, None], 0, 1), linewidths=0))
        pts.append(v)
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=0, azim=azim)
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
    pos = sorted({threshold, 2.0, 3.0, vmax} | ({bonf} if bonf else set()))
    pos = [x for x in pos if threshold <= x <= vmax]
    ticks = sorted(set([-x for x in pos] + pos)) if signed else pos
    cb.set_ticks(ticks)
    cb.set_ticklabels([_p_label(abs(x)) + ("*" if bonf and np.isclose(abs(x), bonf) else "") for x in ticks],
                      fontsize=font)
    side = f"  (left: {sides[0]}, right: {sides[1]})" if signed else ""
    star = f"; * Bonferroni threshold (0.05/{n_bonf})" if bonf else ""
    cb.set_label(f"{label}{side}; gray: p ≥ .05{star}", fontsize=font)
    return cb


def draw_subcortical_3d(fig_id, width=16, row_height=1.6, font=12, colorbar_font=10, vmax=5.0, threshold=1.3,
                        cmap="cold_hot"):
    """Figures 7-8 on the 3D subcortical meshes: rows = versions, columns = baseline and rate of change, four views
    each; structures coloured by (signed) -log10(p), gray below the threshold."""
    d = load_data()
    res, signed = (d["anc"], False) if fig_id == "fig7" else (d["corr"], True)
    norm = matplotlib.colors.Normalize(-vmax, vmax)
    times = [("base", "Baseline"), ("long", "Rate of change")]
    fig = plt.figure(figsize=(width, row_height * len(fsvv.VERSIONS) + 1.4))
    gs = fig.add_gridspec(len(fsvv.VERSIONS), 1 + 2 * len(VIEWS_3D),
                          width_ratios=[0.35] + [1] * (2 * len(VIEWS_3D)), wspace=0, hspace=0,
                          left=0.01, right=0.99, top=0.9, bottom=0.12)
    for i, v in enumerate(fsvv.VERSIONS):
        lab = fig.add_subplot(gs[i, 0]); lab.set_axis_off()
        lab.text(0.9, 0.5, v, ha="right", va="center", fontsize=font)
        for t, (time, title) in enumerate(times):
            r = res[(res["version"] == v) & (res["time"] == time)]
            sign = _sign(r, signed)
            vals = {roi: -np.log10(max(p, 1e-300)) * s for roi, p, s in zip(r["roi"], r["p"], sign) if np.isfinite(p)}
            for j, (side, view, azim) in enumerate(VIEWS_3D):
                ax = fig.add_subplot(gs[i, 1 + t * len(VIEWS_3D) + j], projection="3d")
                _render(ax, side, azim, vals, cmap, norm, threshold=threshold)
                if i == 0:
                    ax.set_title(f"{side[0]} {view}", fontsize=font - 2, pad=0)
        if i == 0:
            for t, (_, title) in enumerate(times):
                x0 = gs[0, 1 + t * len(VIEWS_3D)].get_position(fig).x0
                x1 = gs[0, (t + 1) * len(VIEWS_3D)].get_position(fig).x1
                fig.text((x0 + x1) / 2, 0.96, title, ha="center", fontsize=font + 1)
    _p_colorbar(fig, [0.3, 0.05, 0.4, 0.02], cmap, vmax, threshold, signed=True, label=_P_LABEL[fig_id],
                font=colorbar_font, sides=_SIDES[fig_id])
    return fig


def draw_subcortical(fig_id, style="glass_ortho", width=9, row_height=2.0, font=12, colorbar_font=10, vmax=5.0,
                     threshold=1.3, cmap="cold_hot", cut_y="10,-5,-20"):
    """Figure 7 (ANCOVA HC vs PD-non-MCI) or 8 (correlation with UPDRS-III); style glass_subcortical (split
    view cropped on the deep gray matter) / glass_ortho / glass_xz / glass_lyrz / mni_slices / fsaverage_slices."""
    if style == "surface3d":
        return draw_subcortical_3d(fig_id, width=width, row_height=row_height, font=font,
                                   colorbar_font=colorbar_font, vmax=vmax, threshold=threshold, cmap=cmap)
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
        return nib.Nifti1Image(vol, aseg.affine)

    times = [("base", "Baseline"), ("long", "Rate of change")]
    fig, axes = plt.subplots(len(fsvv.VERSIONS), 2, figsize=(width, row_height * len(fsvv.VERSIONS) + 1.2))
    for row, v in zip(axes, fsvv.VERSIONS):
        for ax, (time, title) in zip(row, times):
            img = volume(res[(res["version"] == v) & (res["time"] == time)])
            common = dict(axes=ax, figure=fig, threshold=threshold, vmax=vmax, cmap=cmap, colorbar=False,
                          annotate=False)
            if style == "glass_subcortical":  # split view (left sagittal, coronal, axial, right sagittal), cropped
                disp = plotting.plot_glass_brain(img, display_mode="lyrz", plot_abs=False, black_bg=False, **common)
                _crop_subcortical(disp)
            elif style.startswith("glass"):
                plotting.plot_glass_brain(img, display_mode=style.split("_")[1], plot_abs=False, black_bg=False,
                                          **common)
            else:
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


def draw(fig_id, **params):
    if fig_id == "fig2":
        return draw_fig2(**params)
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
