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
# # 09 · Interactive tuning of the paper figures (Figures 2–8)
#
# Run in JupyterLab on a compute node (the figures read participant-level tables). One tab per figure: change the
# settings, press **Draw** to redraw (large figures take a few seconds), then **Save to paper** to write the figure
# under its paper name in `outputs/figures` (PNG + PDF, used by the LaTeX) and record the settings in
# `scripts/paper/figure_params.json`, which `paper_figures.load_params()` reads back the next time.
#
# Preview: drawn at a lower resolution (`PREVIEW_DPI`) to stay fast; the saved files use the full resolution.
# The figure code is in `notebooks/paper_figures.py`.

# %%
# %matplotlib inline
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display

import paper_figures as pf

PREVIEW_DPI = 50
pf.load_data()  # once (a few seconds)
params = pf.load_params()

# %%
RANGES = {  # parameter: (min, max, step)
    "width": (4, 30, 0.5), "height": (4, 40, 0.5), "row_height": (1, 4, 0.1),
    "font": (5, 20, 1), "title_font": (6, 24, 1), "legend_font": (5, 20, 1), "colorbar_font": (5, 20, 1),
    "grid_lw": (0, 1, 0.05), "marker_size": (5, 150, 5), "box_width": (0.3, 1.0, 0.05), "legend_cols": (1, 8, 1),
    "vmax": (2, 10, 0.5), "threshold": (0.5, 3, 0.1), "ratio_max": (1.2, 5, 0.1), "lower_height": (0.15, 0.6, 0.05),
}
CHOICES = {
    "xscale": ["linear", "log"], "xlim": ["whiskers", "auto"],
    "style": ["glass_subcortical", "surface3d", "glass_ortho", "glass_xz", "glass_lyrz", "mni_slices", "fsaverage_slices"],
    "layout": ["ratio_heatmap", "dots", "boxes_by_lobe", "boxes", "enigma"], "hemis": ["lh+rh", "bilateral"],
    "metric": ["ratio", "relative"], "preview_part": [1, 2],
    "ratio_cmap": ["RdBu_r", "PuOr_r", "coolwarm", "magma_r"],
    "cmap": ["cold_hot", "hot", "RdBu_r", "coolwarm", "viridis"],
}
TITLES = {"fig2": "Fig 2 p-values", "fig3": "Fig 3 subcortical", "fig4": "Fig 4 volume", "fig5": "Fig 5 area",
          "fig6": "Fig 6 thickness", "fig7": "Fig 7 group (subcortical)", "fig8": "Fig 8 UPDRS (subcortical)"}


def control(name, value):
    style = {"description_width": "110px"}
    layout = w.Layout(width="360px")
    if isinstance(value, bool):
        return w.Checkbox(value=value, description=name, style=style)
    if name in CHOICES:
        return w.Dropdown(options=CHOICES[name], value=value, description=name, style=style, layout=layout)
    if name == "xmax":  # 0 = automatic
        return w.FloatText(value=value or 0, description="xmax (0=auto)", style=style, layout=layout)
    if name in RANGES:
        lo, hi, step = RANGES[name]
        cls = w.IntSlider if isinstance(value, int) and float(step).is_integer() else w.FloatSlider
        return cls(value=value, min=lo, max=hi, step=step, description=name, continuous_update=False, style=style,
                   layout=layout)
    return w.Text(value=str(value), description=name, style=style, layout=layout)


def panel(fig_id):
    ctrls = {k: control(k, v) for k, v in params[fig_id].items()}
    draw_btn = w.Button(description="Draw", button_style="primary", icon="refresh")
    save_btn = w.Button(description="Save to paper", button_style="success", icon="save")
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

    def on_reset(_):
        for k, c in ctrls.items():
            c.value = pf.DEFAULTS[fig_id][k] if pf.DEFAULTS[fig_id][k] is not None else 0

    draw_btn.on_click(on_draw)
    save_btn.on_click(on_save)
    reset_btn.on_click(on_reset)
    grid = w.GridBox(list(ctrls.values()), layout=w.Layout(grid_template_columns="repeat(3, 370px)"))
    return w.VBox([grid, w.HBox([draw_btn, save_btn, reset_btn, status]), out])


tabs = w.Tab(children=[panel(f) for f in TITLES])
for i, t in enumerate(TITLES.values()):
    tabs.set_title(i, t)
display(tabs)
