"""Fixed sets of layouts of Figures 2-8 (outputs/figures/variants/ + index.html), drawn with notebooks/paper_figures.py.

For interactive tuning use notebooks/09_figure_tuning.ipynb instead.
Usage (compute node): python scripts/paper/figure_variants.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402
import paper_figures as pf  # noqa: E402

OUT = fsvv.FIGURES / "variants"
VARIANTS = {
    "fig2": [dict(height=h, font=f, width=16 if f == 8 else 18) for h in (18, 22, 26) for f in (8, 10)],
    "fig3": [dict(width=w_, height=h, font=f, fliers=fl, xscale=sc)
             for (w_, h) in ((9, 16), (11, 20)) for (f, fl, sc) in ((10, False, "linear"), (10, True, "log"),
                                                                    (12, False, "linear"))],
    **{k: [dict(width=w_, height=h, font=f, fliers=fl, xscale=sc)
           for (w_, h) in ((16, 26), (18, 32)) for (f, fl, sc) in ((10, False, "linear"), (10, True, "log"),
                                                                  (12, False, "linear"))]
       for k in ("fig4", "fig5", "fig6")},
    **{k: [dict(style=s, width=wd) for s, wd in (("glass_ortho", 9), ("glass_xz", 7.5), ("mni_slices", 10))]
       for k in ("fig7", "fig8")},
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pf.load_data()
    names = []
    for fig_id, variants in VARIANTS.items():
        for v in variants:
            params = {**pf.DEFAULTS[fig_id], **v}
            name = fig_id + "_" + "_".join(f"{k}{val}" for k, val in v.items())
            fig = pf.draw(fig_id, **params)
            fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
            fig.savefig(OUT / f"{name}.pdf", dpi=200, bbox_inches="tight")
            plt.close(fig)
            names.append(name)
            print("wrote", name, flush=True)
    rows = "\n".join(f'<h3>{n}</h3><img src="{n}.png" style="max-width:100%;border:1px solid #ccc">' for n in names)
    (OUT / "index.html").write_text(f"<!doctype html><meta charset=utf-8><body style='font-family:sans-serif'>"
                                    f"<h1>{len(names)} variants</h1>{rows}")
    print(f"{len(names)} variants in {OUT}")
