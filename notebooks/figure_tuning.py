"""Interactive tuning UI of the paper figures (ipywidgets), shared by 09_figure_tuning (main paper) and
10_supp_figure_tuning (supplement).

One tab per figure: one control per setting of paper_figures.DEFAULTS[fig_id], **Draw** to redraw, **Save to paper**
to write the figure under its paper name in outputs/figures and record the settings in figure_params.json,
**Show in paper** to build the LaTeX and show the page(s) of the figure as saved, **Defaults** to reset the controls.
"""
import sys

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display

import paper_figures as pf

sys.path.insert(0, str(pf.fsvv.ROOT / "scripts" / "paper"))
import figures as paper  # noqa: E402  (build the LaTeX and find a figure's page)

PREVIEW_DPI = 50
PAGE_DPI = 200  # resolution of the paper page shown by "Show in paper" (high, so text stays sharp on HiDPI screens)
PAGE_WIDTH = "850px"  # displayed width of that page

# parameter: (min, max, step). Wide on purpose: the figure code accepts any value in these ranges.
RANGES = {
    "width": (2, 60, 0.5), "height": (2, 80, 0.5), "row_height": (0.5, 12, 0.1),
    "font": (3, 40, 1), "title_font": (3, 48, 1), "suptitle_font": (3, 48, 1), "legend_font": (3, 40, 1), "colorbar_font": (3, 40, 1),
    "grid_lw": (0, 3, 0.05), "marker_size": (1, 400, 1), "box_width": (0.1, 1.0, 0.05), "legend_cols": (1, 12, 1),
    "flier_size": (0.1, 30, 0.1), "cross_size": (2, 30, 1),
    "vmax": (1, 20, 0.5), "threshold": (0, 10, 0.05), "ratio_max": (1.1, 10, 0.1), "lower_height": (0.05, 0.9, 0.05),
    "xmin": (0, 0.99, 0.01), "px_per_mm": (2, 30, 1), "wspace": (-0.3, 0.5, 0.01), "hspace": (-0.3, 0.5, 0.01),
    "legend_dx": (-0.5, 0.5, 0.005), "legend_dy": (-0.5, 0.5, 0.005),
}
CHOICES = {
    "xscale": ["linear", "log"], "yscale": ["linear", "log"], "xlim": ["whiskers", "auto"],
    "style": ["surface3d", "glass_lyrz", "glass_subcortical", "glass_ortho", "glass_xz", "mni_slices",
              "fsaverage_slices"],
    "layout": ["ratio_heatmap", "dots", "boxes_by_lobe", "boxes", "enigma"], "hemis": ["lh+rh", "bilateral"],
    "metric": ["ratio", "relative"], "preview_part": [1, 2], "superior_anterior": ["up", "left"],
    "ratio_cmap": ["RdBu_r", "PuOr_r", "coolwarm", "magma_r"],
    "cmap": ["cold_hot", "hot", "RdBu_r", "coolwarm", "viridis"],
    "legend_pos": pf.LEGEND_POSITIONS,
}


def control(name, value):
    style = {"description_width": "110px"}
    layout = w.Layout(width="360px")
    if isinstance(value, bool):
        return w.Checkbox(value=value, description=name, style=style)
    if name in CHOICES:
        options = CHOICES[name] if value in CHOICES[name] else [value] + CHOICES[name]
        return w.Dropdown(options=options, value=value, description=name, style=style, layout=layout)
    if name == "xmax":  # 0 = automatic
        return w.FloatText(value=value or 0, description="xmax (0=auto)", style=style, layout=layout)
    if name in RANGES:
        lo, hi, step = RANGES[name]
        value = min(max(value, lo), hi)
        cls = w.IntSlider if isinstance(value, int) and float(step).is_integer() else w.FloatSlider
        return cls(value=value, min=lo, max=hi, step=step, description=name, continuous_update=False, style=style,
                   layout=layout)
    return w.Text(value=str(value), description=name, style=style, layout=layout)


def panel(fig_id, params):
    ctrls = {k: control(k, v) for k, v in params[fig_id].items()}
    draw_btn = w.Button(description="Draw", button_style="primary", icon="refresh")
    save_btn = w.Button(description="Save to paper", button_style="success", icon="save")
    page_btn = w.Button(description="Show in paper", icon="file-pdf-o")
    reset_btn = w.Button(description="Defaults", icon="undo")
    status = w.HTML()
    out = w.Output()

    def values():
        v = {k: c.value for k, c in ctrls.items()}
        if "xmax" in v:
            v["xmax"] = v["xmax"] or None
        return v

    def on_draw(_=None):
        out.clear_output(wait=True)
        status.value = "drawing…"
        with out:
            fig = pf.draw(fig_id, **values())
            fig.set_dpi(PREVIEW_DPI)
            display(fig)
            plt.close(fig)
        status.value = ""

    def on_save(_):
        status.value = "saving…"
        fig = pf.draw(fig_id, **values())
        path = pf.save_to_paper(fig_id, fig, values())
        plt.close(fig)
        status.value = f"saved <code>{path.name}</code> and its settings"

    def on_page(_):
        """Build the paper with the saved figures and show the page(s) of this figure."""
        name = pf.PAPER_NAMES[fig_id]
        names = [name] if paper.locate(name) else [f"{name}_part1", f"{name}_part2"]
        found = [(n, paper.locate(n)) for n in names if paper.locate(n)]
        if not found:
            status.value = f"<code>{name}</code> is not included in the paper"
            return
        status.value = "building the paper…"
        pages, built = [], {}
        for n, (doc, caption) in found:
            if doc not in built:
                built[doc] = paper.build(doc)
            pdf, ok = built[doc]
            page = paper.find_page(pdf, caption)
            if page:
                pages.append((f"{caption.rstrip(':')} — {doc}.pdf p. {page}", paper.render_page(pdf, page, PAGE_DPI)))
        out.clear_output(wait=True)
        with out:
            for title, png in pages:
                display(w.VBox([w.HTML(f"<b>{title}</b>"), w.Image(value=png, format="png", layout=w.Layout(width=PAGE_WIDTH))]))
        failed = [d for d, (_, ok) in built.items() if not ok]
        status.value = (f"latexmk failed for {', '.join(failed)} (see paper/latex/*.log); showing the last build. "
                        if failed else "") + ("" if pages else "figure page not found in the PDF") + \
            ("" if failed or not pages else "paper page with the <b>saved</b> figure (press Save to paper first)")

    def on_reset(_):
        for k, c in ctrls.items():
            d = pf.DEFAULTS[fig_id][k]
            c.value = d if d is not None else 0

    draw_btn.on_click(on_draw)
    save_btn.on_click(on_save)
    page_btn.on_click(on_page)
    reset_btn.on_click(on_reset)
    grid = w.GridBox(list(ctrls.values()), layout=w.Layout(grid_template_columns="repeat(3, 370px)"))
    return w.VBox([grid, w.HBox([draw_btn, save_btn, page_btn, reset_btn, status]), out])


def show(fig_ids):
    """Tabs of the given figures (pf.MAIN_FIGURES or pf.SUPP_FIGURES)."""
    params = pf.load_params()
    tabs = w.Tab(children=[panel(f, params) for f in fig_ids])
    for i, f in enumerate(fig_ids):
        tabs.set_title(i, pf.TITLES[f])
    display(tabs)
    return tabs
