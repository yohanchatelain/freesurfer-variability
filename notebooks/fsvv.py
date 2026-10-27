"""Shared code for the FreeSurfer-versions-variability notebooks.

Paths, region lists, FreeSurfer stats parsers, version comparisons, Dice of significant
results, clinical analyses and lightweight (matplotlib) figures.

Column names follow the original notebook so tables can be compared with the published ones:
  measures      "{roi}_{FSx}" (aseg), "{roi}_{hemi}_{measure}_{FSx}" (Destrieux), "TIV_{FSx}"
  differences   "FS{a}_{b}_pct_{key}" (|A-B|/B), "FS{a}_{b}_diff_{key}" (|A-B|)
  test keys     "{roi}" (aseg) or "{hemi}_{roi}_{measure}" (Destrieux)
"""
from __future__ import annotations

import json
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# FSVV_OUT redirects all outputs (e.g. for a validation run)
OUT = Path(os.environ.get("FSVV_OUT", ROOT / "outputs")).resolve()
FIGURES = OUT / "figures"
# PPMI study files used to build the cohorts: the 2025-09 LivingPark snapshot for the revision.
# FSVV_STUDY_FILES=<ROOT>/data/inputs/study_files reproduces the published cohort.
STUDY_FILES = Path(os.environ.get("FSVV_STUDY_FILES", DATA / "study_files_2025-09"))
# FreeSurfer subject folders <root>/FSx/<subject>: the paper's runs (data/FSx) and the runs of the participants
# added for the revision (data/new_participants/light/FSx).
PAPER_ROOT, NEW_ROOT = DATA, DATA / "new_participants" / "light"
_paper_participants = None


def paper_participants():
    """Participants of the paper's cross-sectional cohort (data/tables/download_data.csv)."""
    global _paper_participants
    if _paper_participants is None:
        _paper_participants = set(pd.read_csv(DATA / "tables" / "download_data.csv")["PATNO"])
    return _paper_participants


def subject_dir(version, subject, root=None):
    """FreeSurfer folder of a subject: the first root holding its stats/ (else the first root).

    The paper's participants keep the paper's runs. The others use the revision runs first: data/FS8 also holds
    incomplete FS8 runs of some of them. A participant of both (re-run for the revision) keeps the paper's runs:
    re-running the same image on another cluster gives different results (numerical variability).
    """
    if root:
        roots = [Path(root)]
    else:
        patno = int(re.match(r"sub-(\d+)", subject).group(1))
        roots = [PAPER_ROOT, NEW_ROOT] if patno in paper_participants() else [NEW_ROOT, PAPER_ROOT]
    for r in roots:
        if (r / version / subject / "stats").is_dir():
            return r / version / subject
    return roots[0] / version / subject


def set_output_dir(path):
    """Redirect every output (used to keep a validation run apart from the real outputs)."""
    global OUT, FIGURES
    OUT = Path(path).resolve()
    FIGURES = OUT / "figures"


def out(name):
    """Path of a file written by the notebooks (always under outputs/, never under data/)."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def inp(name, sub="tables"):
    """Path of a file read by the notebooks: regenerated copy in outputs/ if any, else the original in data/."""
    p = OUT / name
    return p if p.exists() else DATA / sub / name


# --------------------------------------------------------------------------- multiple comparisons
# Bonferroni: raw p-values are compared to ALPHA / n_tests. p-values are never multiplied
# (multiplying produced p > 1 in the submitted tables).
ALPHA = 0.05


def fmt_p(p, n_tests=1, alpha=ALPHA):
    """Raw p-value for tables; '*' marks significance after Bonferroni (p < alpha / n_tests)."""
    if p is None or np.isnan(p):
        return "n/a"
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    return s + ("*" if p < alpha / n_tests else "")


# --------------------------------------------------------------------------- versions
VERSIONS = ["FS5", "FS6", "FS7", "FS8"]
# Order of the original notebook (76, 75, 65, 87, 86, 85): newer version first.
PAIRS = [("FS7", "FS6"), ("FS7", "FS5"), ("FS6", "FS5"), ("FS8", "FS7"), ("FS8", "FS6"), ("FS8", "FS5")]


def tag(pair):
    """('FS7', 'FS6') -> '76'."""
    return pair[0][2:] + pair[1][2:]


def prefix(pair):
    """('FS7', 'FS6') -> 'FS7_6' (column prefix of version differences)."""
    return f"{pair[0]}_{pair[1][2:]}"


def label(pair):
    """('FS7', 'FS6') -> 'FS 7 vs 6'."""
    return f"FS {pair[0][2:]} vs {pair[1][2:]}"


# --------------------------------------------------------------------------- regions
ASEG_ROIS = [
    "Left-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Left-Cerebellum-White-Matter", "Left-Cerebellum-Cortex",
    "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum", "3rd-Ventricle", "4th-Ventricle",
    "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "CSF", "Left-Accumbens-area", "Left-VentralDC",
    "Left-vessel", "Left-choroid-plexus", "Right-Lateral-Ventricle", "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter", "Right-Cerebellum-Cortex", "Right-Thalamus", "Right-Caudate",
    "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "Right-VentralDC", "Right-vessel", "Right-choroid-plexus", "5th-Ventricle", "Optic-Chiasm",
    "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior",
]
# FS5 and FS6 call the thalamus "Thalamus-Proper".
ASEG_ALIASES = {"Left-Thalamus-Proper": "Left-Thalamus", "Right-Thalamus-Proper": "Right-Thalamus"}

# Plot order of the subcortical regions (left/right next to each other).
ASEG_PLOT_ORDER = [
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter", "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex", "Left-Thalamus", "Right-Thalamus", "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen", "Left-Pallidum", "Right-Pallidum", "Left-Hippocampus",
    "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala", "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-VentralDC", "Right-VentralDC", "Left-vessel", "Right-vessel", "Left-choroid-plexus",
    "Right-choroid-plexus", "3rd-Ventricle", "4th-Ventricle", "5th-Ventricle", "CSF", "Brain-Stem",
    "Optic-Chiasm", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior",
]

# Destrieux (aparc.a2009s) labels, FS7/FS8 spelling.
DESTRIEUX_ROIS = [
    "G_and_S_frontomargin", "G_and_S_occipital_inf", "G_and_S_paracentral", "G_and_S_subcentral",
    "G_and_S_transv_frontopol", "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal", "G_cingul-Post-ventral", "G_cuneus", "G_front_inf-Opercular",
    "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup",
    "G_Ins_lg_and_S_cent_ins", "G_insular_short", "G_occipital_middle", "G_occipital_sup",
    "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "G_oc-temp_med-Parahip", "G_orbital",
    "G_pariet_inf-Angular", "G_pariet_inf-Supramar", "G_parietal_sup", "G_postcentral", "G_precentral",
    "G_precuneus", "G_rectus", "G_subcallosal", "G_temp_sup-G_T_transv", "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar", "G_temp_sup-Plan_tempo", "G_temporal_inf", "G_temporal_middle",
    "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "Lat_Fis-post", "Pole_occipital", "Pole_temporal",
    "S_calcarine", "S_central", "S_cingul-Marginalis", "S_circular_insula_ant", "S_circular_insula_inf",
    "S_circular_insula_sup", "S_collat_transv_ant", "S_collat_transv_post", "S_front_inf", "S_front_middle",
    "S_front_sup", "S_interm_prim-Jensen", "S_intrapariet_and_P_trans", "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal", "S_occipital_ant", "S_oc-temp_lat", "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral", "S_orbital_med-olfact", "S_orbital-H_Shaped", "S_parieto_occipital",
    "S_pericallosal", "S_postcentral", "S_precentral-inf-part", "S_precentral-sup-part", "S_suborbital",
    "S_subparietal", "S_temporal_inf", "S_temporal_sup", "S_temporal_transverse",
]
# FS5/FS6 use "&" in some Destrieux names (canonical -> on-disk).
DESTRIEUX_OLD_NAMES = {
    "G_and_S_frontomargin": "G&S_frontomargin", "G_and_S_occipital_inf": "G&S_occipital_inf",
    "G_and_S_paracentral": "G&S_paracentral", "G_and_S_subcentral": "G&S_subcentral",
    "G_and_S_transv_frontopol": "G&S_transv_frontopol", "G_and_S_cingul-Ant": "G&S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant": "G&S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post": "G&S_cingul-Mid-Post",
    "G_Ins_lg_and_S_cent_ins": "G_Ins_lg&S_cent_ins", "S_intrapariet_and_P_trans": "S_intrapariet&P_trans",
    "S_oc_middle_and_Lunatus": "S_oc_middle&Lunatus", "S_oc_sup_and_transversal": "S_oc_sup&transversal",
    "S_oc-temp_med_and_Lingual": "S_oc-temp_med&Lingual",
}
DESTRIEUX_CANONICAL = {v: k for k, v in DESTRIEUX_OLD_NAMES.items()}

# Desikan-Killiany (aparc) labels: same 34 names in FS5-8 (R3 major 4).
DK_ROIS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform",
    "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis",
    "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
]
# Lobes of the DK regions (FreeSurfer lobe grouping; cingulate and insula kept apart).
DK_LOBES = {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral", "frontalpole"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "insula": ["insula"],
}
DK_LOBE = {roi: lobe for lobe, rois in DK_LOBES.items() for roi in rois}
# Lobes of the Destrieux regions for the Stage I text (per hemisphere). Reconstructed for the revision: the
# submitted percentages imply 8 limbic, 24 frontal, 11 temporal, 10 occipital and 10 parietal regions; the 11 others
# (insula and sulci on lobe boundaries) are only counted in the totals.
DESTRIEUX_LOBES = {
    "limbic": ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post", "G_cingul-Post-dorsal",
               "G_cingul-Post-ventral", "G_subcallosal", "S_pericallosal", "G_oc-temp_med-Parahip"],
    "frontal": ["G_and_S_frontomargin", "G_and_S_paracentral", "G_and_S_subcentral", "G_and_S_transv_frontopol",
                "G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup",
                "G_orbital", "G_precentral", "G_rectus", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_central",
                "S_front_inf", "S_front_middle", "S_front_sup", "S_orbital_lateral", "S_orbital_med-olfact",
                "S_orbital-H_Shaped", "S_precentral-inf-part", "S_precentral-sup-part", "S_suborbital"],
    "temporal": ["G_oc-temp_lat-fusifor", "G_temp_sup-G_T_transv", "G_temp_sup-Lateral", "G_temp_sup-Plan_polar",
                 "G_temp_sup-Plan_tempo", "G_temporal_inf", "G_temporal_middle", "Pole_temporal", "S_temporal_inf",
                 "S_temporal_sup", "S_temporal_transverse"],
    "occipital": ["G_and_S_occipital_inf", "G_cuneus", "G_occipital_middle", "G_occipital_sup",
                  "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus",
                  "S_oc_sup_and_transversal", "S_occipital_ant"],
    "parietal": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar", "G_parietal_sup", "G_postcentral", "G_precuneus",
                 "S_cingul-Marginalis", "S_interm_prim-Jensen", "S_intrapariet_and_P_trans", "S_postcentral",
                 "S_subparietal"],
}
DESTRIEUX_LOBE = {roi: lobe for lobe, rois in DESTRIEUX_LOBES.items() for roi in rois}

# Cortical atlas used by the measure extraction, tests and figures (see set_atlas): stats file, labels, on-disk names
ATLASES = {
    "destrieux": ("aparc.a2009s", DESTRIEUX_ROIS, DESTRIEUX_CANONICAL),
    "dk": ("aparc", DK_ROIS, {}),
}
HEMIS = ["lh", "rh"]
CORTICAL_MEASURES = ["vol", "surf", "ct"]
MEASURE_NAMES = {"vol": "Volume", "surf": "Surface area", "ct": "Cortical thickness"}
MEASURE_UNITS = {"vol": "mm$^3$", "surf": "mm$^2$", "ct": "mm"}


def set_atlas(name="destrieux", hemis=("lh", "rh")):
    """Select the cortical atlas (and hemispheres) of every function below; the default is the paper's Destrieux.

    hemis=("bi",) tests the bilateral measures added by bilateral_measures instead of lh/rh.
    N_TESTS (Bonferroni) follows: volume = 39 aseg + n_hemis x n_regions, surface/thickness = n_hemis x n_regions
    (Destrieux lh/rh: 187, 148, 148).
    """
    global ATLAS, CORTICAL_STATS, CORTICAL_ROIS, CORTICAL_CANONICAL, CORTICAL_HEMIS, N_TESTS
    ATLAS = name
    CORTICAL_STATS, CORTICAL_ROIS, CORTICAL_CANONICAL = ATLASES[name]
    CORTICAL_HEMIS = list(hemis)
    n = len(CORTICAL_HEMIS) * len(CORTICAL_ROIS)
    N_TESTS = {"vol": len(ASEG_ROIS) + n, "surf": n, "ct": n}


set_atlas()

# Clinical analyses
CLINICAL_ROIS = [r for r in ASEG_ROIS if r not in ("Right-vessel", "5th-Ventricle")]  # as in the original analysis
CLINICAL_TABLE_ROIS = [  # rows of Tables 2-3
    "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum", "Left-Hippocampus", "Left-Amygdala",
    "Left-Accumbens-area", "Right-Thalamus", "Right-Caudate", "Right-Putamen", "Right-Pallidum",
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
]
N_SUBCORTICAL = len(CLINICAL_TABLE_ROIS)
# Not estimated by FS8 in the longitudinal stream for some subjects.
EXCLUDED_LONGITUDINAL = {("FS8", "Left-vessel"), ("FS8", "Left-choroid-plexus"),
                         ("FS8", "Right-choroid-plexus"), ("FS8", "Optic-Chiasm")}

MRIQC_METRICS = ["cjv", "cnr", "fber", "qi_1", "qi_2", "rpve_gm", "snr_gm", "snr_total", "snrd_gm", "snrd_total"]
# MRIQC 22.0.1 outputs (first visit of each participant): the paper's, then the revision's (scripts/mriqc)
MRIQC_ROOTS = [DATA / "source_files" / "clinical" / "mriqc" / "output", OUT / "mriqc" / "output"]


def mriqc_file(patno, event):
    """MRIQC JSON of a participant's first visit: the first of MRIQC_ROOTS holding it."""
    rel = Path(f"sub-{patno}") / f"ses-{event}" / "anat" / f"sub-{patno}_T1w.json"
    for root in MRIQC_ROOTS:
        if (root / rel).exists():
            return root / rel
    return MRIQC_ROOTS[0] / rel


# Image-quality and scanner covariates (reviewers: site and SNR). PPMI site isn't in the study files we have, so the
# scanner manufacturer of the first visit stands in for it (Siemens = reference).
QC_COVARIATES = ["snr_total", "scanner_GE", "scanner_Philips"]


def qc_covariates(download):
    """One row per participant (index PATNO): MRIQC snr_total and manufacturer dummies of the first visit.

    `download` is download_data.csv of 01_cohort; its first row per participant is the visit MRIQC was run on.
    """
    first = download.drop_duplicates("PATNO").set_index("PATNO")
    snr = {p: json.load(open(f))["snr_total"] if (f := mriqc_file(p, e)).exists() else np.nan
           for p, e in first["EVENT_ID"].items()}
    maker = first["Imaging Protocol"].str.extract(r"Manufacturer=([^;]+)")[0].str.upper().str.split().str[0]
    return pd.DataFrame({"snr_total": pd.Series(snr), "scanner_GE": (maker == "GE").astype(int),
                         "scanner_Philips": (maker == "PHILIPS").astype(int)}).rename_axis("PATNO")


def measure_keys():
    """All tested measures as (key, kind, column_stem): aseg volumes then cortical (set_atlas) vol/surf/ct."""
    keys = [(roi, "vol", roi) for roi in ASEG_ROIS]
    for m in CORTICAL_MEASURES:
        for roi in CORTICAL_ROIS:
            for hemi in CORTICAL_HEMIS:
                keys.append((f"{hemi}_{roi}_{m}", m, f"{roi}_{hemi}_{m}"))
    return keys


# --------------------------------------------------------------------------- FreeSurfer stats files
def read_aseg_stats(path):
    """aseg.stats -> (eTIV, {structure: volume_mm3}); thalamus names harmonized across versions."""
    tiv, vols = np.nan, {}
    with open(path) as f:
        for line in f:
            if "Estimated Total Intracranial Volume" in line:
                tiv = float(line.split(",")[3])
            elif not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 5:
                    vols[ASEG_ALIASES.get(parts[4], parts[4])] = float(parts[3])
    return tiv, vols


def read_aparc_stats(path):
    """?h.<atlas>.stats -> {canonical label: (surface, volume, thickness)} (names harmonized for the set_atlas atlas)."""
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                data[CORTICAL_CANONICAL.get(parts[0], parts[0])] = (float(parts[2]), float(parts[3]), float(parts[4]))
    return data


def extract_measures(subjects, versions=VERSIONS, aparc=True, suffix="", root=None):
    """Wide table of aseg (+ cortical, set_atlas) measures for FreeSurfer subject folders in <root>/FSx (default SUBJECT_ROOTS).

    Missing stats files give NaN. `suffix` is appended to every column (e.g. "_first").
    """
    rows = []
    for subj in subjects:
        row = {}
        for v in versions:
            sdir = subject_dir(v, subj, root) / "stats"
            try:
                tiv, vols = read_aseg_stats(sdir / "aseg.stats")
            except FileNotFoundError:
                tiv, vols = np.nan, {}
            row[f"TIV_{v}{suffix}"] = tiv
            for roi in ASEG_ROIS:
                row[f"{roi}_{v}{suffix}"] = vols.get(roi, np.nan)
            if not aparc:
                continue
            for hemi in HEMIS:
                try:
                    ap = read_aparc_stats(sdir / f"{hemi}.{CORTICAL_STATS}.stats")
                except FileNotFoundError:
                    ap = {}
                for roi in CORTICAL_ROIS:
                    surf, vol, ct = ap.get(roi, (np.nan, np.nan, np.nan))
                    row[f"{roi}_{hemi}_vol_{v}{suffix}"] = vol
                    row[f"{roi}_{hemi}_surf_{v}{suffix}"] = surf
                    row[f"{roi}_{hemi}_ct_{v}{suffix}"] = ct
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(list(subjects), name="PATNO_id"))


def bilateral_measures(measures, versions=VERSIONS, suffix=""):
    """Bilateral cortical measures (hemi "bi"): mean of lh and rh for volume and thickness, sum for surface area."""
    cols = {}
    for roi in CORTICAL_ROIS:
        for kind in CORTICAL_MEASURES:
            for v in versions:
                lh, rh = (measures[f"{roi}_{h}_{kind}_{v}{suffix}"] for h in HEMIS)
                cols[f"{roi}_bi_{kind}_{v}{suffix}"] = lh + rh if kind == "surf" else (lh + rh) / 2
    return pd.DataFrame(cols, index=measures.index)


def version_differences(table, pairs=PAIRS):
    """Absolute relative (pct) and absolute (diff) differences between versions, per subject."""
    cols = {}
    for kind in ("pct", "diff"):
        for key, _, stem in measure_keys():
            for a, b in pairs:
                va, vb = table[f"{stem}_{a}"], table[f"{stem}_{b}"]
                d = (va - vb) if kind == "diff" else (va - vb) / vb
                cols[f"{prefix((a, b))}_{kind}_{key}"] = d.abs()
    return pd.DataFrame(cols, index=table.index)


# --------------------------------------------------------------------------- version comparisons
def paired_version_tests(table, pairs=PAIRS):
    """Paired t-tests between versions for every measure -> tidy DataFrame (pair, key, kind, t, p)."""
    rows = []
    for key, kind, stem in measure_keys():
        for a, b in pairs:
            t, p = stats.ttest_rel(table[f"{stem}_{a}"], table[f"{stem}_{b}"], nan_policy="omit")
            rows.append((tag((a, b)), key, kind, float(t), float(p)))
    return pd.DataFrame(rows, columns=["pair", "key", "kind", "t", "p"])


def first_visits(table, download):
    """Rows of `table` (index PATNO_id) that are the first visit of their participant in download_data.csv.

    One image per participant, the one MRIQC was run on (BL, or the first usable later visit). A participant whose
    first-visit image was excluded after QC is dropped rather than replaced by the follow-up.
    """
    return table.loc[table.index.isin(download.drop_duplicates("PATNO")["PATNO_id"])]


def absent_labels(table, version="FS8"):
    """Volume keys that `version` never produces (0 or missing in every image, e.g. FS8 vessel, choroid plexus).

    Their relative difference is 1 for every image, so the paired version tests flag them as significant.
    """
    return [key for key, kind, stem in measure_keys()
            if kind == "vol" and f"{stem}_{version}" in table and (table[f"{stem}_{version}"].fillna(0) == 0).all()]


def group_variability_tests(table, group1, group2, pairs=PAIRS):
    """Two-sample t-tests (group1 vs group2) of the relative version differences."""
    rows = []
    for key, kind, _ in measure_keys():
        for pair in pairs:
            col = f"{prefix(pair)}_pct_{key}"
            t, p = stats.ttest_ind(table.loc[group1, col], table.loc[group2, col], nan_policy="omit")
            rows.append((tag(pair), key, kind, float(t), float(p)))
    return pd.DataFrame(rows, columns=["pair", "key", "kind", "t", "p"])


def group_variability_models(table, group1, group2, covar=(), pairs=PAIRS):
    """OLS `pct ~ group + covar` of the relative version differences: t and p of the group1 vs group2 coefficient.

    Without covariates this is the pooled two-sample t-test of group_variability_tests (same t and p).
    Rows with a missing value in the outcome or a covariate are dropped per model.
    """
    import statsmodels.api as sm
    rows = []
    sub = table.loc[group1 | group2]
    base = pd.concat([group1.loc[sub.index].astype(float).rename("group"), sub[list(covar)].astype(float)], axis=1)
    for key, kind, _ in measure_keys():
        for pair in pairs:
            col = f"{prefix(pair)}_pct_{key}"
            d = pd.concat([sub[col], base], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if np.allclose(d[col], d[col].iloc[0]):  # label absent in one version (FS8: vessel, choroid plexus, ...)
                rows.append((tag(pair), key, kind, len(d), np.nan, np.nan))
                continue
            fit = sm.OLS(d[col], sm.add_constant(d.drop(columns=col))).fit()
            rows.append((tag(pair), key, kind, len(d), float(fit.tvalues["group"]), float(fit.pvalues["group"])))
    return pd.DataFrame(rows, columns=["pair", "key", "kind", "n", "t", "p"])


def qc_correlations(table, metrics=MRIQC_METRICS, pairs=PAIRS, replace_inf=False):
    """Pearson correlation between MRIQC metrics and relative version differences."""
    rows = []
    for key, kind, _ in measure_keys():
        for m in metrics:
            for pair in pairs:
                col = f"{prefix(pair)}_pct_{key}"
                d = table[[col, m]]
                if replace_inf:
                    d = d.replace([np.inf, -np.inf], np.nan)
                d = d.dropna()
                r, p = stats.pearsonr(d[col], d[m]) if len(d) > 2 else (np.nan, np.nan)
                rows.append((tag(pair), key, kind, m, float(r), float(p)))
    return pd.DataFrame(rows, columns=["pair", "key", "kind", "metric", "r", "p"])


def significant_sets(tests, alpha=ALPHA):
    """{(pair, kind): set of keys with p < alpha / N_TESTS[kind]} (Bonferroni by dividing alpha)."""
    sets = {}
    for (pair, kind), g in tests.groupby(["pair", "kind"]):
        sets[(pair, kind)] = set(g.loc[g["p"] < alpha / N_TESTS[kind], "key"])
    return sets


def proportion_significant(sets, pairs=PAIRS):
    """Share of regions with a significant version difference, per pair and measure."""
    return pd.DataFrame({MEASURE_NAMES[k]: [len(sets[(tag(p), k)]) / N_TESTS[k] for p in pairs]
                         for k in CORTICAL_MEASURES}, index=[label(p) for p in pairs])


def proportion_by_lobe(tests, pairs=PAIRS, alpha=ALPHA):
    """% of significant regions per pair and measure: all regions, subcortical (volume) and each Destrieux lobe."""
    t = tests.assign(sig=tests["p"] < alpha / tests["kind"].map(N_TESTS))
    sub = t["key"].isin(ASEG_ROIS)
    roi = t["key"].str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    t["group"] = np.where(sub, "subcortical", roi.map(DESTRIEUX_LOBE))
    rows = {}
    for pair in pairs:
        for kind in CORTICAL_MEASURES:
            g = t[(t["pair"] == tag(pair)) & (t["kind"] == kind)]
            rows[(label(pair), MEASURE_NAMES[kind])] = {"all": g["sig"].mean() * 100,
                                                       **(g.groupby("group")["sig"].mean() * 100).to_dict()}
    cols = ["all", "subcortical", *DESTRIEUX_LOBES]
    return pd.DataFrame.from_dict(rows, orient="index")[cols].rename_axis(["pair", "measure"])


def dice_between_sets(sets, kind, pairs=PAIRS):
    """Sørensen-Dice of the sets of significant regions between version comparisons (diagonal = 1)."""
    tags = [tag(p) for p in pairs]
    m = pd.DataFrame(1.0, index=[label(p) for p in pairs], columns=[label(p) for p in pairs])
    for (i, a), (j, b) in combinations(enumerate(tags), 2):
        sa, sb = sets[(a, kind)], sets[(b, kind)]
        d = 2 * len(sa & sb) / (len(sa) + len(sb)) if (sa or sb) else np.nan
        m.iloc[i, j] = m.iloc[j, i] = d
    return m


# --------------------------------------------------------------------------- segmentation Dice (mri_seg_overlap)
# Read order matters: labels shared by several segmentations keep the value of the last file read.
# aseg is read last so that whole-structure labels (e.g. Cerebral-Cortex, which in aparc+aseg and
# aparc.a2009s+aseg only holds leftover unparcellated voxels, Dice ~ 0) keep their aseg value.
# The original notebook read aseg first (published dice_results.csv: Cerebral-Cortex Dice ~0.5-0.7).
SEG_FILES = ["aparc2009", "aparc", "aseg"]


# mri_seg_overlap JSONs: computed for the revision (scripts/dice), then the paper's
DICE_ROOTS = [OUT / "dice", DATA / "stats" / "dice"]


def dice_file(pair, subj, seg):
    """Dice JSON of one image, version pair and segmentation: the first of DICE_ROOTS holding it."""
    name = f"dice_{tag(pair)}_{subj}.{seg}.json"
    for root in DICE_ROOTS:
        if (root / name).exists():
            return root / name
    return DICE_ROOTS[-1] / name


def read_seg_dice(subjects, pairs=PAIRS, files=SEG_FILES):
    """Per-label Dice between versions from the JSONs of DICE_ROOTS -> wide table 'dice_FS{tag}_{label}'.

    Labels present in several segmentations keep their aseg value (see SEG_FILES).
    """
    rows = []
    for subj in subjects:
        row = {}
        for pair in pairs:
            for f in files:
                with open(dice_file(pair, subj, f)) as fh:
                    d = json.load(fh)
                for k, v in d["measures"]["dice"]["labels"].items():
                    row[f"dice_FS{tag(pair)}_{d['names'][k]}"] = v
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(list(subjects), name="PATNO_id"))


def seg_dice_long(dice_wide):
    """Wide Dice table -> long (comparison, hemi, roi, dice); only lateralized labels have a hemi."""
    d = dice_wide.melt(var_name="col", value_name="dice").dropna(subset=["dice"])
    ex = d["col"].str.extract(r"^dice_FS(\d)(\d)_(.*)$")
    d["comparison"] = "FS " + ex[0] + " vs " + ex[1]
    raw = ex[2]
    d["hemi"] = raw.str.extract(r"^(Left|Right|ctx_lh|ctx_rh)")[0].map(
        {"Left": "lh", "ctx_lh": "lh", "Right": "rh", "ctx_rh": "rh"})
    d["roi"] = raw.str.replace(r"^(Left-|Right-|ctx_lh_|ctx_rh_)", "", regex=True)
    return d.drop(columns="col")


# --------------------------------------------------------------------------- vertex-wise GLM results
CLUSTER_COLUMNS = ["ClusterNo", "Max", "VtxMax", "Size(mm^2)", "MNIX", "MNIY", "MNIZ", "CWP", "CWPLow",
                   "CWPHi", "NVtxs", "WghtVtx", "Annot"]


def glm_contrast_dir(version, analysis, time, hemi, root=None):
    """Contrast folder of one GLM: <root>/FSx/results_{analysis}_{time}_*_{hemi}/con_* (scripts/glm/run_glm.sbatch).

    root defaults to FSVV_GLM_RESULTS, else <outputs>/glm (revision cohort). The paper's results use the harmonized
    names <root>/FSx/{analysis}_{time}_{hemi}_con: FSVV_GLM_RESULTS=<repo>/data/glm_results.
    """
    root = Path(root or os.environ.get("FSVV_GLM_RESULTS", OUT / "glm"))
    harmonized = root / version / f"{analysis}_{time}_{hemi}_con"
    if harmonized.exists():
        return harmonized
    found = sorted(root.glob(f"{version}/results_{analysis}_{time}_*_{hemi}/con_*"))
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} contrast folders for {version} {analysis} {time} {hemi} in {root}")
    return found[0]


def read_cluster_summaries(versions=VERSIONS, analyses=("group", "corr"), times=("base", "long"), root=None):
    """Significant clusters (mri_glmfit-sim, cluster-wise p < .05) of every GLM -> one table (see glm_contrast_dir)."""
    frames = []
    for v in versions:
        for a in analyses:
            for t in times:
                for hemi in HEMIS:
                    f = glm_contrast_dir(v, a, t, hemi, root) / "cache.th13.abs.sig.cluster.summary"
                    # FS5's mri_glmfit-sim has no WghtVtx column: take the names from the file's header line
                    names = next(l for l in open(f) if l.startswith("# ClusterNo"))[1:].split()
                    d = pd.read_csv(f, sep=r"\s+", comment="#", header=None, names=names)
                    frames.append(d.assign(version=v, analysis=a, time=t, hemi=hemi))
    cols = ["version", "analysis", "time", "hemi"] + CLUSTER_COLUMNS
    return pd.concat(frames, ignore_index=True)[cols]


# --------------------------------------------------------------------------- clinical analyses
def clinical_volumes(stage_two, versions=VERSIONS, rois=CLINICAL_ROIS, aparc=False):
    """Subcortical volumes at both visits (longitudinal stream) and % change, for HC and PD-non-MCI.

    Rows whose aseg.stats is missing for any version or visit are dropped (returned separately).
    With aparc=True the cortical measures (set_atlas) are extracted too, and `rois` may hold their column stems
    (e.g. "superiorfrontal_lh_ct") to get their % change.
    """
    t = stage_two.loc[stage_two["dx_group"].isin(["HC", "PD-non-MCI"])].copy()
    base = "sub-" + t["PATNO"].astype(str)
    t["first_id"] = base + "_ses-" + t["EVENT_ID"] + ".long." + base + "_base"
    t["second_id"] = base + "_ses-" + t["NEXT_VISIT"] + ".long." + base + "_base"
    ok = [all((subject_dir(v, s) / "stats" / "aseg.stats").exists() for v in versions for s in (f, s2))
          for f, s2 in zip(t["first_id"], t["second_id"])]
    dropped = t.loc[[not o for o in ok], ["first_id", "second_id"]]
    t = t.loc[ok].reset_index(drop=True)
    first = extract_measures(t["first_id"], versions, aparc=aparc, suffix="_first").reset_index(drop=True)
    second = extract_measures(t["second_id"], versions, aparc=aparc, suffix="_second").reset_index(drop=True)
    if aparc and CORTICAL_HEMIS == ["bi"]:
        first = pd.concat([first, bilateral_measures(first, versions, "_first")], axis=1)
        second = pd.concat([second, bilateral_measures(second, versions, "_second")], axis=1)
    t = pd.concat([t, first, second], axis=1)
    change = {f"{roi}_{v}_change_pct": (t[f"{roi}_{v}_second"] - t[f"{roi}_{v}_first"]) / t[f"{roi}_{v}_first"] * 100
              for roi in rois for v in versions}
    return pd.concat([t, pd.DataFrame(change)], axis=1), dropped


def clinical_partial_correlations(table, versions=VERSIONS, rois=CLINICAL_ROIS, extra_covar=()):
    """Partial correlation (Pearson) of volumes with UPDRS-III in PD-non-MCI (baseline and change).

    Covariates: age, sex (+ time between visits for the change), plus `extra_covar` (e.g. QC_COVARIATES); "{v}" in a
    covariate name is replaced by the version (e.g. "TIV_{v}_first" for each version's own eTIV).
    """
    from pingouin import partial_corr
    pdn = table.loc[table["dx_group"] == "PD-non-MCI"]
    rows = []
    for time, x_fmt, y, covar in [
        ("base", "{roi}_{v}_first", "NP3TOT", ["AGE_AT_VISIT", "SEX"]),
        ("long", "{roi}_{v}_change_pct", "NP3TOT_change", ["AGE_AT_VISIT", "SEX", "durationT2_T1_y"]),
    ]:
        for roi in rois:
            for v in versions:
                if time == "long" and (v, roi) in EXCLUDED_LONGITUDINAL:
                    continue
                r = partial_corr(data=pdn, x=x_fmt.format(roi=roi, v=v), y=y, covar=[*covar, *(c.format(v=v) for c in extra_covar)],
                                 method="pearson")
                ci = r["CI95"] if "CI95" in r else r["CI95%"]
                p = r["p_val"] if "p_val" in r else r["p-val"]
                rows.append((v, roi, time, int(r["n"].iloc[0]), float(r["r"].iloc[0]), ci.iloc[0], float(p.iloc[0])))
    return pd.DataFrame(rows, columns=["version", "roi", "time", "n", "r", "ci95", "p"])


def clinical_ancova(table, versions=VERSIONS, rois=CLINICAL_ROIS, extra_covar=()):
    """ANCOVA HC vs PD-non-MCI on volumes (baseline) and % change (longitudinal); covariates as in
    clinical_partial_correlations."""
    from pingouin import ancova
    rows = []
    for time, dv_fmt, covar in [
        ("base", "{roi}_{v}_first", ["AGE_AT_VISIT", "SEX"]),
        ("long", "{roi}_{v}_change_pct", ["AGE_AT_VISIT", "SEX", "durationT2_T1_y"]),
    ]:
        for roi in rois:
            for v in versions:
                if time == "long" and (v, roi) in EXCLUDED_LONGITUDINAL:
                    continue
                a = ancova(data=table, dv=dv_fmt.format(roi=roi, v=v), between="dx_group",
                           covar=[*covar, *(c.format(v=v) for c in extra_covar)])
                pcol = "p_unc" if "p_unc" in a else "p-unc"
                rows.append((v, roi, time, float(a["F"].iloc[0]), float(a[pcol].iloc[0])))
    return pd.DataFrame(rows, columns=["version", "roi", "time", "F", "p"])


def clinical_table(results, stat, time, rois=CLINICAL_TABLE_ROIS, versions=VERSIONS, n_tests=N_SUBCORTICAL):
    """Paper table: one row per ROI, (stat, p) per version; p raw with '*' after Bonferroni (alpha/n_tests, 14)."""
    d = results.loc[results["time"] == time].set_index(["roi", "version"])
    cols = {}
    for v in versions:
        cols[(v, stat)] = [round(d.loc[(r, v), stat], 3) if (r, v) in d.index else np.nan for r in rois]
        cols[(v, "p")] = [fmt_p(d.loc[(r, v), "p"], n_tests) if (r, v) in d.index else "n/a" for r in rois]
    return pd.DataFrame(cols, index=pd.Index(rois, name="ROI"))


# --------------------------------------------------------------------------- cohort helpers
VISIT2MONTH = {"BL": 0, "V01": 3, "V02": 6, "V03": 9, "V04": 12, "V05": 18, "V06": 24, "V07": 30, "V08": 36,
               "V09": 42, "V10": 48, "V11": 54, "V12": 60, "V13": 72, "V14": 84, "V15": 96, "V16": 108,
               "V17": 120, "V18": 132, "V19": 144, "V20": 156}


def find_visit_pairs(months):
    """Visit pairs whose time difference is closest to `months` ({visit: next_visit})."""
    best, hist = float("inf"), {}
    for (k, v), (k_, v_) in combinations(VISIT2MONTH.items(), 2):
        d = abs(abs(v - v_) - months)
        if d <= best:
            best = d
            hist.setdefault(d, {})[k] = k_
    return hist[best]


def pair_visits(df, months, keep, drop_cols=("AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS")):
    """Join each visit with the visit `months` later (suffix _NX) and keep rows where keep(row_df) is True."""
    v = df.copy()
    v["NEXT_VISIT"] = v["EVENT_ID"].map(find_visit_pairs(months))
    v = v.merge(v.drop(list(drop_cols), axis=1), left_on=["PATNO", "NEXT_VISIT"],
                right_on=["PATNO", "EVENT_ID"], suffixes=(None, "_NX")).drop_duplicates()
    return v.loc[keep(v)]


# --------------------------------------------------------------------------- figures (matplotlib, static)
PAIR_COLORS = dict(zip([label(p) for p in PAIRS], ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]))
VERSION_COLORS = dict(zip(VERSIONS, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))
VERSION_MARKERS = dict(zip(VERSIONS, ["x", "^", "o", "D"]))


def savefig(fig, name, dpi=200):
    """Save a figure to outputs/figures (PNG, and PDF for the paper)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi, bbox_inches="tight")  # dpi of the rasterized parts


def _grouped_boxes(ax, data, regions, groups, colors, horizontal=True, width=0.8):
    """Box plots of data[group][region] (lists of values), grouped by region."""
    n = len(groups)
    w = width / n
    for gi, g in enumerate(groups):
        pos = np.arange(len(regions)) + (gi - (n - 1) / 2) * w
        vals = [np.asarray(data[g][r], dtype=float) for r in regions]
        vals = [v[~np.isnan(v)] for v in vals]
        bp = ax.boxplot(vals, positions=pos, widths=w * 0.9, vert=not horizontal, patch_artist=True,
                        showfliers=True, flierprops=dict(markersize=1.5, alpha=0.4, markeredgewidth=0),
                        medianprops=dict(color="black", linewidth=0.8), whiskerprops=dict(linewidth=0.6),
                        capprops=dict(linewidth=0.6), boxprops=dict(linewidth=0.6), manage_ticks=False)
        for b in bp["boxes"]:
            b.set_facecolor(colors[g])
            b.set_alpha(0.85)
    ticks = np.arange(len(regions))
    (ax.set_yticks if horizontal else ax.set_xticks)(ticks)
    (ax.set_yticklabels if horizontal else ax.set_xticklabels)(regions, fontsize=7,
                                                               **({} if horizontal else {"rotation": 90}))
    (ax.set_ylim if horizontal else ax.set_xlim)(-0.6, len(regions) - 0.4)
    ax.grid(axis="x" if horizontal else "y", linestyle=":", linewidth=0.5)


def plot_variability(table, kind, hemi=None, pairs=PAIRS, versions=VERSIONS, ax=None, xlim=None):
    """Between-version absolute differences (boxes, one colour per pair) vs between-subject SD (markers).

    kind="aseg" for subcortical volumes, or "vol"/"surf"/"ct" for the cortical atlas (set_atlas) with hemi="lh"/"rh"/"bi".
    """
    import matplotlib.pyplot as plt
    if kind == "aseg":
        regions = ASEG_PLOT_ORDER
        diff_col = lambda p, r: f"{prefix(p)}_diff_{r}"
        est_col = lambda v, r: f"{r}_{v}"
        unit = MEASURE_UNITS["vol"]
        title = "Subcortical volumes"
    else:
        regions = CORTICAL_ROIS
        diff_col = lambda p, r: f"{prefix(p)}_diff_{hemi}_{r}_{kind}"
        est_col = lambda v, r: f"{r}_{hemi}_{kind}_{v}"
        unit = MEASURE_UNITS[kind]
        title = f"{MEASURE_NAMES[kind]} ({hemi})"
    groups = [label(p) for p in pairs]
    data = {label(p): {r: table[diff_col(p, r)].values for r in regions} for p in pairs}
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.22 * len(regions) + 1.5))
    _grouped_boxes(ax, data, regions, groups, PAIR_COLORS, horizontal=True)
    y = np.arange(len(regions))
    for v in versions:
        sd = [table[est_col(v, r)].std() for r in regions]
        ax.scatter(sd, y, marker=VERSION_MARKERS[v], s=14, color="black", linewidths=0.8, zorder=5,
                   label=f"SD {v}")
    ax.invert_yaxis()
    ax.set_xlabel(f"{MEASURE_NAMES['vol'] if kind == 'aseg' else MEASURE_NAMES[kind]} ({unit})")
    ax.set_title(title, fontsize=10)
    if xlim:
        ax.set_xlim(*xlim)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[g]) for g in groups]
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles + h2, groups + l2, fontsize=7, ncol=2, loc="lower right", frameon=True,
              title="between-version |diff| / between-subject SD", title_fontsize=7)
    return ax


def plot_estimates(table, kind, hemi=None, versions=VERSIONS, ax=None):
    """Distribution of the raw estimates per version (boxes), per region."""
    import matplotlib.pyplot as plt
    if kind == "aseg":
        regions, col, unit = ASEG_PLOT_ORDER, (lambda v, r: f"{r}_{v}"), MEASURE_UNITS["vol"]
        title = "Subcortical volumes"
    else:
        regions, col, unit = CORTICAL_ROIS, (lambda v, r: f"{r}_{hemi}_{kind}_{v}"), MEASURE_UNITS[kind]
        title = f"{MEASURE_NAMES[kind]} ({hemi})"
    data = {v: {r: table[col(v, r)].values for r in regions} for v in versions}
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.25 * len(regions) + 2, 4))
    _grouped_boxes(ax, data, regions, versions, VERSION_COLORS, horizontal=False)
    ax.set_ylabel(unit)
    ax.set_title(title, fontsize=10)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=VERSION_COLORS[v]) for v in versions], versions,
              fontsize=7, ncol=4, loc="upper right")
    return ax


def plot_seg_dice(dice_long, hemi, ax=None):
    """Segmentation Dice between versions per region (boxes per comparison) for one hemisphere."""
    import matplotlib.pyplot as plt
    d = dice_long.loc[dice_long["hemi"] == hemi]
    regions = sorted(d["roi"].unique())
    groups = [label(p) for p in PAIRS]
    data = {g: {r: d.loc[(d["comparison"] == g) & (d["roi"] == r), "dice"].values for r in regions} for g in groups}
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.16 * len(regions) + 1.5))
    _grouped_boxes(ax, data, regions, groups, PAIR_COLORS, horizontal=True)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sørensen–Dice coefficient")
    ax.set_title(f"Segmentation overlap between versions ({hemi})", fontsize=10)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[g]) for g in groups], groups, fontsize=7,
              loc="lower left", ncol=2)
    return ax
