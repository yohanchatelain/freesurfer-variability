"""LaTeX tables of the manuscript and the supplement, from the aggregate outputs of notebooks 03, 06 and 07.

Writes <out>/tables/*.tex (\\input by manuscript.tex and supplementary.tex) and <out>/supplementary_data/*.csv
(the tables too long for the PDF: MRIQC correlations). Aggregate results only, no participant-level rows.

Usage: python scripts/paper/make_latex_tables.py [--out paper/latex]
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

TABLES = fsvv.OUT / "tables"
VERSION_NAMES = {"FS5": "FS5", "FS6": "FS6", "FS7": "FS7", "FS8": "FS8"}


def tex(s):
    """Escape a cell for LaTeX."""
    s = "" if pd.isna(s) else str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("<", r"$<$"), ("±", r"$\pm$"), ("*", r"$^{*}$")]:
        s = s.replace(a, b)
    return s


def num(x, fmt="{:.2f}"):
    return "" if pd.isna(x) else fmt.format(x)


def write(path, text):
    path.write_text(text)
    print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def table1(out):
    t = pd.read_csv(TABLES / "table1_descriptive.csv", header=[0, 1], index_col=0)
    rows = [" & ".join([tex(i)] + [tex(v) for v in r]) + r" \\" for i, r in zip(t.index, t.values)]
    groups = [g for _, g in t.columns]
    body = "\n".join(rows)
    write(out / "table1.tex", rf"""\begin{{tabular}}{{lccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{Cross-sectional}} & \multicolumn{{2}}{{c}}{{Longitudinal}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-6}}
 & {' & '.join(groups)} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def stage1(out):
    """Table 2: % of regions with a significant difference, per pair and measure (all, subcortical, lobes)."""
    t = pd.read_csv(TABLES / "table2_stage1_by_lobe.csv", index_col=[0, 1])
    cols = list(t.columns)
    lines = []
    for pair, g in t.groupby(level=0, sort=False):
        for i, ((_, measure), r) in enumerate(g.iterrows()):
            lines.append(" & ".join([pair.replace("FS ", "FS").replace(" vs ", " vs FS") if i == 0 else "", measure]
                                    + [num(r[c], "{:.1f}") for c in cols]) + r" \\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines[:-1])
    head = " & ".join(c.capitalize() for c in cols)
    write(out / "table2.tex", rf"""\begin{{tabular}}{{ll{'r' * len(cols)}}}
\toprule
Pair & Measure & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def clinical(out, name, src, stat):
    """Tables 3-4: ROI rows, (stat, p) per version, baseline and longitudinal panels."""
    t = pd.read_csv(src, header=[0, 1], index_col=[0, 1], dtype=str)  # keep the p strings as written (0.010)
    versions = list(dict.fromkeys(v for v, _ in t.columns))
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{VERSION_NAMES[v]}}}" for v in versions)
    sub = " & ".join(rf"${stat}$ & $p$" for _ in versions)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(versions)))
    lines = []
    for time, g in t.groupby(level=0, sort=False):
        lines.append(rf"\multicolumn{{{1 + 2 * len(versions)}}}{{l}}{{\textit{{{time}}}}} \\")
        for (_, roi), r in g.iterrows():
            cells = []
            for v in versions:
                cells += [num(float(r[(v, stat)]), "{:.3f}"), tex(r[(v, "p")])]
            lines.append(" & ".join([tex(roi)] + cells) + r" \\")
    body = "\n".join(lines)
    write(out / f"{name}.tex", rf"""\begin{{tabular}}{{l{'rr' * len(versions)}}}
\toprule
 & {head} \\
{rules}
ROI & {sub} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def clusters_table(out, name, c):
    """Tables 5-6 (and S8): significant clusters per time and version; versions without one say so."""
    lines = []
    for time in ["Baseline", "Longitudinal"]:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{time}}}}} \\")
        for v in fsvv.VERSIONS:
            g = c[(c["time"] == time) & (c["version"] == v)]
            if g.empty:
                lines.append(rf"{v} & \multicolumn{{6}}{{l}}{{no significant cluster}} \\")
                continue
            for i, (_, r) in enumerate(g.iterrows()):
                lines.append(" & ".join([v if i == 0 else "", tex(r["Region"]), num(r["Size (mm2)"]),
                                         num(r["MNI X"], "{:.1f}"), num(r["MNI Y"], "{:.1f}"),
                                         num(r["MNI Z"], "{:.1f}"), num(r["Max"], "{:.2f}")]) + r" \\")
    body = "\n".join(lines)
    write(out / f"{name}.tex", rf"""\begin{{tabular}}{{llrrrrr}}
\toprule
Version & Region & Size (mm$^2$) & MNI X & MNI Y & MNI Z & Max \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def cluster_frame(clusters, analysis):
    c = clusters[clusters["analysis"] == analysis].copy()
    c["time"] = c["time"].map({"base": "Baseline", "long": "Longitudinal"})
    c["Region"] = c["hemi"].str[0].str.upper() + " " + c["Annot"]
    c = c.sort_values(["time", "version", "hemi", "ClusterNo"])
    return c.rename(columns={"Size(mm^2)": "Size (mm2)", "MNIX": "MNI X", "MNIY": "MNI Y", "MNIZ": "MNI Z"})


def version_tests(out, name, src, stat, caption, label, mark=()):
    """Supplementary longtable: one row per (measure, hemisphere, region), (stat, p) per version pair.

    Regions in `mark` get a dagger (explained in the caption).
    """
    t = pd.read_csv(src, header=[0, 1], index_col=[0, 1, 2], keep_default_na=False, dtype=str)
    pairs = list(dict.fromkeys(p for p, _ in t.columns))
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{p.replace('FS ', '')}}}" for p in pairs)
    sub = " & ".join(rf"${stat}$ & $p$" for _ in pairs)
    ncol = 2 + 2 * len(pairs)
    lines = []
    for (measure, hemi), g in t.groupby(level=[0, 1], sort=False):
        lines.append(rf"\midrule\multicolumn{{{ncol}}}{{l}}{{\textbf{{{measure}{', ' + hemi if hemi else ''}}}}} \\")
        for (_, _, region), r in g.iterrows():
            cells = []
            for p in pairs:
                v = r[(p, stat)]
                cells += [num(float(v), "{:.2f}") if v not in ("", "nan") else "", tex(r[(p, "p")])]
            lines.append(" & ".join([tex(region) + (r"$^{\dagger}$" if region in mark else ""), ""] + cells) + r" \\")
    body = "\n".join(lines)
    write(out / f"{name}.tex", rf"""{{\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{ll{'rr' * len(pairs)}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
Region & & {head} \\
Region & & {sub} \\
\midrule\endfirsthead
\toprule
Region & & {head} \\
Region & & {sub} \\
\midrule\endhead
\bottomrule\endfoot
{body}
\end{{longtable}}}}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT / "paper" / "latex")
    out = Path(ap.parse_args().out)
    tab, data = out / "tables", out / "supplementary_data"
    tab.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)

    table1(tab)
    stage1(tab)
    clinical(tab, "table3", TABLES / "table3_ancova_subcortical.csv", "F")
    clinical(tab, "table4", TABLES / "table4_corr_subcortical.csv", "r")
    clusters = fsvv.read_cluster_summaries()
    clusters_table(tab, "table5", cluster_frame(clusters, "corr"))
    clusters_table(tab, "table6", cluster_frame(clusters, "group"))

    # Supplement. S1-S5: version tests and PD vs HC; S7: Tables 3-4 with QC covariates; S8: Tables 5-6 with QC
    # covariates; S9-S18 / S19-S28: MRIQC correlations (CSV only). S6 and S29-S38 (QPN) wait for the QPN inputs.
    note = (r"Paired t-tests between versions; raw $p$, $^{*}$ significant after Bonferroni correction "
            r"($\alpha = 0.05/187$ for volume, $0.05/148$ for surface area and thickness). $^{\dagger}$Label not "
            r"produced by FS8 (volume 0 in every image): significant by construction in the FS8 comparisons.")
    # Labels FS8 doesn't produce, as flagged by 02_software_variability
    tests = pd.read_csv(fsvv.OUT / "version_paired_ttests.csv")
    absent_fs8 = set(tests.loc[tests["absent_in_FS8"], "key"])
    for name, src, cap in [
        ("tableS1", "tableS1_version_ttests_all.csv", f"Differences between FreeSurfer versions, all images (n = 830). {note}"),
        ("tableS2", "tableS2_version_ttests_PD.csv", f"Differences between FreeSurfer versions, PD images (n = 560). {note}"),
        ("tableS3", "tableS3_version_ttests_HC.csv", f"Differences between FreeSurfer versions, HC images (n = 270). {note}"),
        ("tableS4", "tableS4_pd_vs_hc_variability.csv",
         r"PD vs HC differences in software variability: two-sample t-tests of the relative differences between "
         r"versions, first-visit images (280 PD, 135 HC). Raw $p$, $^{*}$ Bonferroni as in Table S1."),
        ("tableS5", "tableS5_pd_vs_hc_variability_qc_covariates.csv",
         r"PD vs HC differences in software variability adjusted for image quality (MRIQC total SNR) and scanner "
         r"manufacturer: $t$ of the group coefficient of an OLS model, first-visit images. Raw $p$, $^{*}$ Bonferroni "
         r"as in Table S1."),
    ]:
        version_tests(tab, name, TABLES / src, "t", cap, f"tab:{name[5:]}",
                      mark=absent_fs8 if name in ("tableS1", "tableS2", "tableS3") else ())

    for name, src, stat in [("tableS7a", "clinical_ancova_hc_pdnonmci_qc_covariates.csv", "F"),
                            ("tableS7b", "clinical_partial_corr_updrs_qc_covariates.csv", "r")]:
        res = pd.read_csv(fsvv.OUT / src)
        t = pd.concat({"Baseline": fsvv.clinical_table(res, stat, "base"),
                       "Longitudinal": fsvv.clinical_table(res, stat, "long")}, names=["time"])
        t.to_csv(data / f"{name}.csv")
        clinical(tab, name, data / f"{name}.csv", stat)
        (data / f"{name}.csv").unlink()  # in the PDF; the CSV was only the input
    qc_clusters = fsvv.read_cluster_summaries(root=fsvv.OUT / "glm_qc")
    clusters_table(tab, "tableS8a", cluster_frame(qc_clusters, "corr"))
    clusters_table(tab, "tableS8b", cluster_frame(qc_clusters, "group"))

    for src, dst in [("tableS9-S18_mriqc_corr_all.csv", "TablesS9-S18_mriqc_correlations_all_images.csv"),
                     ("tableS19-S28_mriqc_corr_PD.csv", "TablesS19-S28_mriqc_correlations_PD_images.csv")]:
        shutil.copy(TABLES / src, data / dst)
        print("wrote", data / dst)


if __name__ == "__main__":
    main()
