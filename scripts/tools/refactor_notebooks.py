"""Refactor the copied notebooks to read from ../data and write to ../outputs.

Usage: python refactor_notebooks.py <src.ipynb> <dst.ipynb> <kind> <report.txt>
kind: main | mindboggle | clinical-FS5 | clinical-FS6 | clinical-FS7 | scratch

Only kernel code cells are rewritten. Cells starting with a cell magic (%%writefile, %%sbatch,
%%bash) are pipeline scripts executed outside the kernel and are left untouched.
"""
import copy
import json
import re
import sys

src, dst, kind, report_path = sys.argv[1:5]

CONFIG = '''# ---- Paths (revision workspace) ----
# Inputs are read from ../data (symlinks to /lustre09/project/6049200/vinuyans, see ../data/README.md).
# Everything this notebook writes goes to ../outputs. Never write into ../data: the links point to
# the original files, which are writable.
from pathlib import Path

DATA = Path("../data").resolve()
OUT = Path("../outputs").resolve()
OUT.mkdir(exist_ok=True)
INPUT_SUBDIR = "{subdir}"  # default ../data subdirectory for files written by the original pipeline


def out(name):
    """Path of a file written by this notebook (always under ../outputs)."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def inp(name, sub=None):
    """Path of a file read by this notebook: regenerated copy in ../outputs if any, else the original in ../data."""
    p = OUT / name
    return str(p if p.exists() else DATA / (sub or INPUT_SUBDIR) / name)


# Multiple comparisons (Bonferroni): raw p-values are compared to ALPHA / n_tests.
# p-values are never multiplied (this produced p > 1 in the submitted tables).
ALPHA = 0.05
N_SUBCORTICAL = 14  # bilateral subcortical ROIs tested in the clinical analyses (Tables 2-3)


def fmt_p(p, n_tests=1, alpha=ALPHA):
    """Raw p-value for tables; '*' marks significance after Bonferroni (p < alpha / n_tests)."""
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    return s + ("*" if p < alpha / n_tests else "")
'''

WARNING_MD = (
    "> **Revision workspace note.** The `%%writefile` / `%%sbatch` cells in the processing sections are the "
    "original pipeline. They were run in `/lustre09/project/6049200/vinuyans` and use paths relative to that "
    "directory. **Don't run them from this workspace**: `preprocess_FS8*.py` deletes existing subject folders "
    "(`shutil.rmtree`), and `data/FS8` links to the original outputs. Re-runs go through `../scripts/` with "
    "outputs on `$SCRATCH`."
)

SUBDIR = {
    "main": "tables",
    "mindboggle": "tables",
    "scratch": "tables",
    "clinical-FS5": "glm_FS5-7/FS5",
    "clinical-FS6": "glm_FS5-7/FS6",
    "clinical-FS7": "glm_FS5-7/FS7",
}[kind]

GLM_FILE = r"(?:qdec_long_groups\.dat|[^\"'/]*\.fsgd|con_[^\"'/]*\.mtx|[^\"'/]*\.mgh)"
Q = r"([\"'])"

RULES = [
    # --- p-values: Bonferroni by dividing alpha, never multiplying p (R3 major 1, R2 p = 0) ---
    (r"str\(\s*\(\s*((?:corr_cov|ancova)\[[^\]]+\])\s*\*\s*14\s*\)\.round\(\s*decimals=3\s*\)\s*\)",
     r"fmt_p(\1, n_tests=N_SUBCORTICAL)"),
    # --- writes -> ../outputs ---
    (r"open\(\s*" + Q + r"([^\"']+)\1(\.format\([^()]*\))(\s*,\s*[\"'](?:w|wb|a)[\"'])", r'open(out("\2"\3)\4'),
    (r"open\((\w+),\s*([\"'])rb\2\)", r'open(inp(\1), "rb")'),
    (r"^(\s*)out = Path\(\s*" + Q + r"([^\"']+)\2\s*\)", r'\1out_path = Path(out("\3"))'),
    (r"open\(out,", r"open(out_path,"),
    (r"\{out\}", r"{out_path}"),
    (r"open\(\s*" + Q + r"(?:\.\./)?([^\"']+)\1(\s*,\s*[\"'](?:w|wb|a)[\"'])", r'open(out("\2")\3'),
    (r"\.to_csv\(\s*" + Q + r"(?:\.\./)?([^\"']+)\1", r'.to_csv(out(\1\2\1)'),
    (r"os\.path\.abspath\(\s*" + Q + r"\.\./([^\"']+)\1\s*\)", r"out('\2')"),
    (r"save_path=" + Q + r"([^\"']+)\1", r'save_path=out("\2")'),
    (r"\.(write_image|write_html|savefig)\(\s*" + Q + r"([^\"']+)\2", r'.\1(out("\3")'),
    (r"iio\.mimsave\(\s*f" + Q + r"([^\"']+)\1", r'iio.mimsave(out(f"\2")'),
    # --- specific inputs ---
    (Q + r"\./boutiques/([^\"']+)\1", r'str(DATA / "boutiques" / "\2")'),
    (r"license=" + Q + r"FS_license/license\.txt\1", r'license=str(DATA / "FS_license" / "license.txt")'),
    (r"f?" + Q + r"/home/vinuyans/links/projects/rrg-glatard/vinuyans/data/FS8\1", r'str(DATA / "FS8")'),
    (Q + r"data/FS8\1", r'str(DATA / "FS8")'),
    # GLM cluster summaries -> harmonized links ../data/glm_results/FSx/{group,corr}_{base,long}_{hemi}_con
    (r"\"stats/results/(FS\d)/results_(group|corr)_[^\"]*/cache\.th13\.abs\.sig\.cluster\.summary\"",
     r'str(DATA) + "/glm_results/\1/\2_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary"'),
    (r"\"stats/FS5_data/sub-\"", r'str(DATA) + "/FS5/sub-"'),
    (r"\"stats/dice/", r'str(DATA) + "/stats/dice/'),
    (r"f\"data/", r'f"{DATA}/'),
    (r"(?<![\w\"])\"data/", r'str(DATA) + "/'),
    (r"f\"mriqc/(output\w*)/", r'f"{DATA}/source_files/clinical/mriqc/\1/'),
    (r"(?<!out\()f\"qpn/", r'f"{DATA}/qpn/'),
    (r"(?<![\w\"(])\"qpn/", r'str(DATA) + "/qpn/'),
    (r"read_csv\(\s*\"qpn/", r'read_csv(str(DATA) + "/qpn/'),
    (r"Path\(f\"segm/", r'Path(f"{DATA}/segm/'),
    (r"os\.path\.join\(\s*utils\.study_files_dir", r'os.path.join(str(DATA / "inputs" / "study_files")'),
    (r"'mindboggle101_atlases\.txt'", r'str(DATA / "mindboggle" / "mindboggle101_atlases.txt")'),
    # --- generic reads of local files (no directory component) ---
    (r"read_csv\(\s*" + Q + r"(?:\.\./)?(" + GLM_FILE + r")\1", r'read_csv(inp("\2", "glm")'),
    (r"read_csv\(\s*" + Q + r"(?:\.\./)?([^\"'/]+)\1", r'read_csv(inp("\2")'),
    (r"open\(\s*" + Q + r"(?:\.\./)?([^\"'/]+)\1(\s*(?:,\s*[\"']r[\"'])?\s*[,)])", r'open(inp("\2")\3'),
    # path constants in the scratch notebook
    (r"^(\s*\w*OUT\w*\s*=\s*)" + Q + r"([^\"'/]+\.json)\2", r'\1out("\3")'),
    (r"^(\s*\w*(?:JSON|INPUT|FILE)\w*\s*=\s*)" + Q + r"([^\"'/]+\.json)\2", r'\1inp("\3")'),
]
RULES = [(re.compile(p, re.M), r) for p, r in RULES]

# path-like literals left after rewriting, reported for manual review
LEFTOVER = re.compile(r"[\"'][^\"'\n]*(?:\.csv|\.json|\.txt|\.dat|\.mtx|\.fsgd|\.mgh|\.png|\.gif|/)[^\"'\n]*[\"']")


GUARD = (
    'raise RuntimeError("Disabled in the revision workspace: this original pipeline step copies FreeSurfer "\n'
    '                   "outputs or creates one folder per subject (file quota). The data is already linked in ../data.")\n\n'
)


def is_magic(src_text):
    return src_text.lstrip().startswith("%%")


nb = json.load(open(src))
cells = nb["cells"]
report = []

for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    text = "".join(c["source"])
    if is_magic(text):
        continue
    new = text
    if re.search(r"os\.makedirs\(\s*'(?:stats|data)/FS|\.\./freesurfer/data_FS", text):
        new = GUARD + text
    for pat, rep in RULES:
        new, n = pat.subn(rep, new)
    if new != text:
        c["source"] = new.splitlines(keepends=True)
        report.append(f"[cell {i}] rewritten")
    for m in LEFTOVER.finditer(new):
        s = m.group(0)
        if "{DATA}" in s or s.startswith(("\"/glm_results", "\"/stats", "\"/FS5", "\"/qpn", "\"/{")):
            continue
        report.append(f"[cell {i}] leftover literal: {s[:120]}")

if kind == "main":
    # FS8 whole-brain result cells: copy the FS7 ones (markdown header + 4 code cells).
    start = next(i for i, c in enumerate(cells)
                 if c["cell_type"] == "markdown" and "FS 7 - whole-brain results" in "".join(c["source"]))
    end = next(i for i in range(start + 1, len(cells))
               if cells[i]["cell_type"] == "markdown" and "Vertex-wise results" in "".join(cells[i]["source"]))
    fs8 = []
    for c in cells[start:end]:
        c8 = copy.deepcopy(c)
        c8["source"] = "".join(c8["source"]).replace("FS 7", "FS 8").replace("FS7", "FS8").splitlines(keepends=True)
        if c8["cell_type"] == "code":
            c8["outputs"], c8["execution_count"] = [], None
        c8.get("metadata", {}).pop("id", None)
        c8.pop("id", None)
        fs8.append(c8)
    cells[end:end] = fs8
    report.append(f"inserted {len(fs8)} FS8 whole-brain result cells at {end}")
    # warning before the processing sections
    pre = next(i for i, c in enumerate(cells)
               if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("# Image preprocessing"))
    cells.insert(pre + 1, {"cell_type": "markdown", "metadata": {}, "source": WARNING_MD.splitlines(keepends=True)})

if kind.startswith("clinical"):
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": WARNING_MD.splitlines(keepends=True)})

cells.insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": CONFIG.replace("{subdir}", SUBDIR).splitlines(keepends=True)})

for c in cells:  # ids are optional in nbformat 4.4 and must be unique in 4.5
    c.pop("id", None)
if nb.get("nbformat_minor", 0) >= 5:
    nb["nbformat_minor"] = 4

json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
open(report_path, "w").write("\n".join(report) + "\n")
print(dst, "rewritten cells:", sum("rewritten" in r for r in report),
      "leftovers:", sum("leftover" in r for r in report))
