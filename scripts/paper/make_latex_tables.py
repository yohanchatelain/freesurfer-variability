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
    blocks, rules, col = [], [], 2  # one header block per first-level column (sample)
    for sample in dict.fromkeys(s for s, _ in t.columns):
        k = sum(s == sample for s, _ in t.columns)
        blocks.append(rf"\multicolumn{{{k}}}{{c}}{{{sample}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + k - 1}}}")
        col += k
    write(out / "table1.tex", rf"""\begin{{tabular}}{{l{'c' * len(groups)}}}
\toprule
 & {' & '.join(blocks)} \\
{''.join(rules)}
 & {' & '.join(groups)} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def stage1(out):
    """Table 2: % of regions with a significant difference, per measure and pair (all, subcortical, lobes).
    Grouped by measure with a rule between measures, so that the pairs of one measure compare down a column."""
    t = pd.read_csv(TABLES / "table2_stage1_by_lobe.csv", index_col=[0, 1])
    cols = list(t.columns)
    blocks = []
    for measure in dict.fromkeys(t.index.get_level_values(1)):
        g = t.xs(measure, level=1)
        blocks.append("\n".join(" & ".join([measure if i == 0 else "", pair.replace("FS ", "FS").replace(" vs ", " vs FS")]
                                           + [num(r[c], "{:.1f}") for c in cols]) + r" \\"
                                for i, (pair, r) in enumerate(g.iterrows())))
    body = "\n\\midrule\n".join(blocks)
    head = " & ".join(c.capitalize() for c in cols)
    write(out / "table2.tex", rf"""\begin{{tabular}}{{ll{'r' * len(cols)}}}
\toprule
Measure & Pair & {head} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


MRIQC_NAMES = {"cjv": "Coefficient of joint variation", "cnr": "Contrast-to-noise ratio",
               "fber": "Foreground-to-background energy ratio", "qi_1": "Mortamet's quality index 1",
               "qi_2": "Mortamet's quality index 2", "rpve_gm": "Residual partial GM volume effect",
               "snr_gm": "GM signal-to-noise ratio", "snr_total": "Total signal-to-noise ratio",
               "snrd_gm": "Dietrich's GM signal-to-noise ratio", "snrd_total": "Dietrich's total signal-to-noise ratio"}
MRIQC_FILES = [("All images, PPMI (415)", "TablesS9-S18_mriqc_correlations_all_images.csv", "S9--S18"),
               ("PD, PPMI (280)", "TablesS19-S28_mriqc_correlations_PD_images.csv", "S19--S28"),
               ("PD, QPN (201)", "TablesS29-S38_mriqc_correlations_QPN_PD_images.csv", "S29--S38")]


def mriqc_summary(out, data):
    """Printed summary of Tables S9-S38 (supplied as CSV): per sample, MRIQC metric and measure, the number of
    significant correlations over the six version pairs (sig = '*' in the CSV: Bonferroni over the regions and the
    10 metrics), the number of tests and the largest |r|."""
    measures = [["Volume (subcortical)", "Volume"], ["Surface area"], ["Cortical thickness"]]
    lines = []
    for title, fname, tabs in MRIQC_FILES:
        d = pd.read_csv(data / fname, header=[0, 1], index_col=[0, 1, 2, 3])
        d = d[[c for c in d.columns if c[0].startswith("FS")]]
        rows, n_tests = [], None
        for metric, name in MRIQC_NAMES.items():
            cells, counts = [], []
            for kinds in measures:
                sub = d[(d.index.get_level_values(0) == metric) & d.index.get_level_values(1).isin(kinds)]
                p = sub.xs("p", axis=1, level=1).replace("<1e-300", 0).apply(pd.to_numeric, errors="coerce")  # n/a: untested
                sig = sub.xs("sig", axis=1, level=1).fillna("").astype(str)
                r = pd.to_numeric(sub.xs("r", axis=1, level=1).stack(), errors="coerce")
                counts.append(int(p.notna().values.sum()))
                cells += [str(int((sig == "*").values.sum())), f"{r.abs().max():.2f}"]
            n_tests = n_tests or counts
            rows.append(" & ".join([tex(name)] + cells) + r" \\")
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}, Tables {tabs}}}}} \\".replace(",", "{,}"))
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{Tests per metric: {n_tests[0]:,} volume, {n_tests[1]:,} "
                     rf"surface area, {n_tests[2]:,} thickness}}}} \\".replace(",", "{,}"))
        lines += rows
    body = "\n".join(lines)
    write(out / "tableS9_summary.tex", rf"""\begin{{tabular}}{{lrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Volume}} & \multicolumn{{2}}{{c}}{{Surface area}} & \multicolumn{{2}}{{c}}{{Cortical thickness}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
MRIQC metric & Sig. & Max $|r|$ & Sig. & Max $|r|$ & Sig. & Max $|r|$ \\
{body}
\bottomrule
\end{{tabular}}
""")


def p_cell(p, n_tests=fsvv.N_SUBCORTICAL, alpha=fsvv.ALPHA):
    """Exact raw p with 2 significant digits (0.19, 0.084, 0.0034; below 0.001 as 7.5x10^-4), in bold when
    significant after Bonferroni correction (p < alpha / n_tests)."""
    if pd.isna(p):
        return "n/a"
    return p_text(p, p < alpha / n_tests)


def p_text(p, bold):
    """Exact p as in the main-text tables: 2 significant digits, below 0.001 as m x 10^e; bold if `bold`.
    Below 1e-300 (underflow: p = 0 or denormal; '<1e-300' in the CSVs) the bound <10^-300 is given, never 0."""
    if isinstance(p, str):
        p = 0.0 if p.startswith("<") else float(p)
    if p < 1e-300:
        s = r"{<}10^{-300}"
    elif p >= 0.001:
        s = f"{p:#.2g}" if p < 0.995 else "0.99"  # '#': keep trailing zeros (0.0070, 0.010); no e-notation here
    else:
        m, e = f"{p:.1e}".split("e")
        s = rf"{m}{{\times}}10^{{{int(e)}}}"
    return rf"$\mathbf{{{s}}}$" if bold else rf"${s}$"


def clinical(out, name, results, stat, rois=fsvv.CLINICAL_TABLE_ROIS, versions=fsvv.VERSIONS):
    """Tables 3-4 (and S7): ROI rows, (stat, p) per version, baseline and longitudinal panels separated by a rule;
    `results` is a results table of 03 (columns version, roi, time, stat, p), p exact and raw, in bold when
    significant after Bonferroni correction (0.05/14)."""
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{VERSION_NAMES[v]}}}" for v in versions)
    sub = " & ".join(rf"${stat}$ & $p$" for _ in versions)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(versions)))
    d = results.set_index(["time", "roi", "version"])
    lines = []
    for k, (time, title) in enumerate([("base", "Baseline"), ("long", "Longitudinal")]):
        if k:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * len(versions)}}}{{l}}{{\textit{{{title}}}}} \\")
        for roi in rois:
            cells = []
            for v in versions:
                if (time, roi, v) in d.index:
                    r = d.loc[(time, roi, v)]
                    cells += [num(r[stat], "{:.3f}"), p_cell(r["p"])]
                else:
                    cells += ["", "n/a"]
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
        lines.append(r"\cmidrule(lr){1-7}")
        for k, v in enumerate(fsvv.VERSIONS):
            shade = r"\rowcolor{gray!12}" if k % 2 else ""  # alternate versions shaded
            g = c[(c["time"] == time) & (c["version"] == v)]
            if g.empty:
                lines.append(rf"{shade}{v} & \multicolumn{{6}}{{l}}{{no significant cluster}} \\")
                continue
            for i, (_, r) in enumerate(g.iterrows()):
                lines.append(shade + " & ".join([v if i == 0 else "", tex(r["Region"]), num(r["Size (mm2)"]),
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
                v, pv = r[(p, stat)], r[(p, "p")]
                cells += [num(float(v), "{:.2f}") if v not in ("", "nan") else "",
                          p_text(pv, r[(p, "sig")] == "*") if pv not in ("", "nan", "n/a") else "n/a"]
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
    clinical(tab, "table3", pd.read_csv(fsvv.OUT / "clinical_ancova_hc_pdnonmci.csv"), "F")
    clinical(tab, "table4", pd.read_csv(fsvv.OUT / "clinical_partial_corr_updrs.csv"), "r")
    clusters = fsvv.read_cluster_summaries()
    clusters_table(tab, "table5", cluster_frame(clusters, "corr"))
    clusters_table(tab, "table6", cluster_frame(clusters, "group"))

    # Supplement. S1-S5: version tests and PD vs HC; S6: version tests, QPN PD; S7: Tables 3-4 with QC covariates;
    # S8: Tables 5-6 with QC covariates; S9-S18 / S19-S28 / S29-S38 (QPN PD): MRIQC correlations (CSV only).
    note = (r"Paired t-tests between versions; raw $p$, exact; in bold: significant after Bonferroni correction "
            r"($\alpha = 0.05/187$ for volume, $0.05/148$ for surface area and thickness). $^{\dagger}$Label that "
            r"FS8 does not produce, or defines differently (CSF): not tested in the FS8 comparisons (n/a).")
    # FS8 labels left out of the FS8 pairs, as flagged by 02_software_variability (fsvv.drop_incomparable)
    tests = pd.read_csv(fsvv.OUT / "version_paired_ttests.csv")
    for name, d in [("version_paired_ttests.csv", tests), ("qpn_version_paired_ttests.csv",
                                                           pd.read_csv(fsvv.OUT / "qpn_version_paired_ttests.csv"))]:
        if "incomparable_in_FS8" not in d:
            raise SystemExit(f"{name} predates fsvv.drop_incomparable: rerun notebooks 02 and 04")
    flag = "incomparable_in_FS8"
    absent_fs8 = set(tests.loc[tests[flag], "key"])
    for name, src, cap in [
        ("tableS1", "tableS1_version_ttests_all.csv", f"Differences between FreeSurfer versions, all images (n = 830). {note}"),
        ("tableS2", "tableS2_version_ttests_PD.csv", f"Differences between FreeSurfer versions, PD images (n = 560). {note}"),
        ("tableS3", "tableS3_version_ttests_HC.csv", f"Differences between FreeSurfer versions, HC images (n = 270). {note}"),
        ("tableS4", "tableS4_pd_vs_hc_variability.csv",
         r"PD vs HC differences in software variability: two-sample t-tests of the relative differences between "
         r"versions, first-visit images (280 PD, 135 HC). Raw $p$, exact; in bold: significant (Bonferroni as in Table S1). $^{\dagger}$As in "
         r"Table S1: not tested in the FS8 comparisons (n/a)."),
        ("tableS5", "tableS5_pd_vs_hc_variability_qc_covariates.csv",
         r"PD vs HC differences in software variability adjusted for image quality (MRIQC total SNR) and scanner "
         r"manufacturer: $t$ of the group coefficient of an OLS model, first-visit images. Raw $p$, exact; in bold: significant (Bonferroni "
         r"as in Table S1). $^{\dagger}$As in Table S1: not tested in the FS8 comparisons (n/a)."),
    ]:
        version_tests(tab, name, TABLES / src, "t", cap, f"tab:{name[5:]}",
                      mark=absent_fs8)

    # S6: QPN replication sample (PD), from 04_qpn_replication and 07
    qpn_tests = pd.read_csv(fsvv.OUT / "qpn_version_paired_ttests.csv")
    n_qpn_pd = int(pd.read_csv(fsvv.OUT / "qpn_demographics.csv", index_col="dx_group").loc["PD", "n"])
    version_tests(tab, "tableS6", TABLES / "tableS6_version_ttests_QPN.csv", "t",
                  f"Differences between FreeSurfer versions in the QPN replication sample, PD (n = {n_qpn_pd}). {note}",
                  "tab:S6", mark=set(qpn_tests.loc[qpn_tests[flag], "key"]))

    for name, src, stat in [("tableS7a", "clinical_ancova_hc_pdnonmci_qc_covariates.csv", "F"),
                            ("tableS7b", "clinical_partial_corr_updrs_qc_covariates.csv", "r")]:
        clinical(tab, name, pd.read_csv(fsvv.OUT / src), stat)
    qc_clusters = fsvv.read_cluster_summaries(root=fsvv.OUT / "glm_qc")
    clusters_table(tab, "tableS8a", cluster_frame(qc_clusters, "corr"))
    clusters_table(tab, "tableS8b", cluster_frame(qc_clusters, "group"))

    # S40: asymmetry of the version differences (08_variability_by_type)
    a = pd.read_csv(fsvv.OUT / "asymmetry_summary.csv")
    order = {"Volume (subcortical)": 0, "Volume": 1, "Surface area": 2, "Cortical thickness": 3}
    pair_order = {fsvv.label(p): i for i, p in enumerate(fsvv.PAIRS)}
    a = a.sort_values(["measure", "pair"], key=lambda c: c.map(order if c.name == "measure" else pair_order))
    lines = []
    for measure, g in a.groupby("measure", sort=False):
        lines.append(rf"\midrule\multicolumn{{5}}{{l}}{{\textbf{{{tex(measure)}}}}} \\")
        lines += [rf"{tex(r.pair)} & {r.tests} & {r.significant} & {r.left_larger} & {r.right_larger} \\"
                  for r in g.itertuples()]
    body = "\n".join(lines)
    write(tab / "tableS40.tex", rf"""\begin{{tabular}}{{lrrrr}}
\toprule
Version pair & Tests & Significant & Left larger & Right larger \\
{body}
\bottomrule
\end{{tabular}}
""")

    for src, dst in [("tableS9-S18_mriqc_corr_all.csv", "TablesS9-S18_mriqc_correlations_all_images.csv"),
                     ("tableS19-S28_mriqc_corr_PD.csv", "TablesS19-S28_mriqc_correlations_PD_images.csv"),
                     ("tableS29-S38_mriqc_corr_QPN_PD.csv", "TablesS29-S38_mriqc_correlations_QPN_PD_images.csv")]:
        shutil.copy(TABLES / src, data / dst)
        print("wrote", data / dst)
    mriqc_summary(tab, data)


if __name__ == "__main__":
    main()
