# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3.11 (fsvv)
#     language: python
#     name: fsvv
# ---

# %% [markdown]
# # 09 · Interactive tuning of the paper figures (Figures 1–12)
#
# Run in JupyterLab (the figures read participant-level tables: on the cluster, or a local copy of `outputs/`). One
# tab per figure: change the settings, press **Draw** to redraw (large figures take a few seconds), then **Save to
# paper** to write the figure under its paper name in `outputs/figures` (PNG + PDF, used by the LaTeX) and record the
# settings in `scripts/paper/figure_params.json`, which `paper_figures.load_params()` reads back the next time.
#
# Preview: drawn at a lower resolution (`figure_tuning.PREVIEW_DPI`) to stay fast; the saved files use the full
# resolution. The figure code is in `notebooks/paper_figures.py`, the tuning UI (controls and slider ranges) in
# `notebooks/figure_tuning.py`.
#

# %%
# %matplotlib inline
import figure_tuning as ft
import paper_figures as pf

pf.load_data();

# %%
ft.show(pf.MAIN_FIGURES);

# %%
