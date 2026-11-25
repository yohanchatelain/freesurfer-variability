"""Interactive tuning UI of the paper figures (ipywidgets), shared by 09_figure_tuning (main paper) and
10_supp_figure_tuning (supplement).

One tab per figure: one control per setting of paper_figures.DEFAULTS[fig_id], **Draw** to redraw, **Save to paper**
to write the figure under its paper name in outputs/figures and record the settings in figure_params.json,
**Show in paper** to build the LaTeX and show the page(s) of the figure (saving it first if the settings changed since
the last save), **Defaults** to reset the controls.
Draw and Show in paper reuse their last result when nothing changed (same settings; same LaTeX sources and figure files).
"""
import functools
import html
import sys
import traceback

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.core.pylabtools import print_figure
from IPython.display import Image, display

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
    "colorbar_length": (0.2, 2.5, 0.05), "colorbar_thickness": (0.2, 8, 0.1),  # scale factors (1 = as drawn)
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
    **{f"view{i}": list(pf.VIEWS_3D_FIG) for i in (1, 2, 3)},  # Figures 7-8 (surface3d): panel at each position
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
    cache = {"draw": (None, None), "page": (None, None)}  # (key, result) of the last Draw / Show in paper

    def values():
        v = {k: c.value for k, c in ctrls.items()}
        if "xmax" in v:
            v["xmax"] = v["xmax"] or None
        return v

    saved = {"values": values()}  # settings of the figure in the paper: as loaded, then as last saved

    def on_draw(_=None):
        """Draw the preview; settings unchanged since the last Draw: show that preview again."""
        key = repr(sorted(values().items()))
        cached = cache["draw"][0] == key
        if not cached:
            status.value = "drawing…"
            fig = pf.draw(fig_id, **values())
            fig.set_dpi(PREVIEW_DPI)
            cache["draw"] = key, print_figure(fig, "png")
            plt.close(fig)
        out.clear_output(wait=True)
        with out:
            display(Image(data=cache["draw"][1], format="png"))
        status.value = "settings unchanged: last preview" if cached else ""

    def on_save(_):
        status.value = "saving…"
        v = values()
        split = pf.is_split(fig_id, v)  # save_to_paper draws both parts itself
        fig = None if split else pf.draw(fig_id, **v)
        path = pf.save_to_paper(fig_id, fig, v)
        if fig is not None:
            plt.close(fig)
        saved["values"] = v
        status.value = f"saved <code>{path.name}</code> and its settings"

    def on_page(_):
        """Save the figure if its settings changed, build the paper and show the page(s) of this figure."""
        name = pf.PAPER_NAMES[fig_id]
        names = [name] if paper.locate(name) else [f"{name}_part1", f"{name}_part2"]
        found = [(n, paper.locate(n)) for n in names if paper.locate(n)]
        if not found:
            status.value = f"<code>{name}</code> is not included in the paper"
            return
        changed = values() != saved["values"]
        if changed:  # settings changed since the last save: save first, as the paper shows the saved figure
            on_save(None)
        stamp = (paper.sources_stamp(), tuple(found))
        if cache["page"][0] == stamp:  # no LaTeX source or figure changed since the last successful build
            show_pages(cache["page"][1])
            status.value = "nothing changed since the last build: last page shown"
            return
        status.value = "building the paper…"
        pages, built = [], {}
        for n, (doc, caption, k) in found:
            if doc not in built:
                built[doc] = paper.build(doc)
            pdf, ok = built[doc]
            page = paper.find_page(pdf, caption, k)
            if page:
                pages.append((f"{caption.rstrip(':')} — {doc}.pdf p. {page}", paper.render_page(pdf, page, PAGE_DPI)))
        show_pages(pages)
        failed = [d for d, (_, ok) in built.items() if not ok]
        cache["page"] = (stamp, pages) if pages and not failed else (None, None)
        status.value = (f"latexmk failed for {', '.join(failed)} (see paper/latex/*.log); showing the last build. "
                        if failed else "") + ("" if pages else "figure page not found in the PDF") + \
            ("" if failed or not pages else "paper page with the saved figure") + \
            (" (settings changed: saved first)" if changed else "")

    def show_pages(pages):
        out.clear_output(wait=True)
        with out:
            for title, png in pages:
                display(w.VBox([w.HTML(f"<b>{title}</b>"), w.Image(value=png, format="png", layout=w.Layout(width=PAGE_WIDTH))]))

    def on_reset(_):
        for k, c in ctrls.items():
            d = pf.DEFAULTS[fig_id][k]
            c.value = d if d is not None else 0

    def guarded(fn):
        """Errors raised in a button callback go to Jupyter's log, not the notebook (the status would stay at
        "drawing…"): show them in the status and the output area instead."""
        @functools.wraps(fn)
        def run(*args):
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001  (any error of the figure code)
                status.value = f"<b style='color:#c00'>{type(e).__name__}</b>: {html.escape(str(e))}"
                out.clear_output(wait=True)
                with out:
                    traceback.print_exc()
        return run

    draw_btn.on_click(guarded(on_draw))
    save_btn.on_click(guarded(on_save))
    page_btn.on_click(guarded(on_page))
    reset_btn.on_click(guarded(on_reset))
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
