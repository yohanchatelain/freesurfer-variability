#!/usr/bin/env python3
"""Paper figure workflow: figures [install|run|show|save|commit].

  install   create .venv (Python 3.11, uv) with requirements.txt + JupyterLab and register the "fsvv" kernel
  run       open the tuning notebooks (09: Figures 1-12, 10: S1-S6); press "Save to paper" in each tab
  show [F]  build the paper and open it at figure F (number: 7, S3; or image name: fig7_corr_base)
  save      copy the saved PDFs from outputs/figures into paper/latex/figures (tracked copies used by the LaTeX)
  commit    stage the figures and their settings, check for participant identifiers, commit and push

Standard library only: run it with any Python (alias figures='python <repo>/scripts/paper/figures.py').
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
LATEX = ROOT / "paper" / "latex"
PAPER_FIGURES = LATEX / "figures"
NOTEBOOKS = ["09_figure_tuning.ipynb", "10_supp_figure_tuning.ipynb"]
TRACKED = ["paper/latex/figures", "scripts/paper/figure_params.json", "notebooks/paper_figures.py",
           "notebooks/figure_tuning.py"]
IDENTIFIERS = re.compile(r"sub-[0-9]+|PATNO|PPMI_[0-9]+|\bI[0-9]{6,8}\b")
DATA_FILE = re.compile(r"\.(csv|tsv|json|dat|txt|nii|nii\.gz|mgz)$")
ALLOWED_DATA = {"scripts/paper/figure_params.json"}  # settings only, no participant data


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kw)


def git(*args):
    # errors="replace": `diff --text` also prints the binary figure PDFs, which are not valid UTF-8
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
                          errors="replace").stdout


def install(_):
    if not shutil.which("uv"):
        sys.exit("uv not found: install it first (https://docs.astral.sh/uv/)")
    if not VENV.exists():
        sh("uv", "venv", VENV, "--python", "3.11")
    py = VENV / "bin" / "python"
    sh("uv", "pip", "install", "--python", py, "-r", ROOT / "requirements.txt", "jupyterlab", "ipykernel")
    sh(py, "-m", "ipykernel", "install", "--user", "--name", "fsvv", "--display-name", "Python 3.11 (fsvv)")


def run(_):
    jupyter = VENV / "bin" / "jupyter"
    if not jupyter.exists():
        sys.exit("no .venv: run `figures install` first")
    sh(jupyter, "lab", *NOTEBOOKS, cwd=ROOT / "notebooks")


def save(_):
    copied = []
    for dst in sorted(PAPER_FIGURES.glob("*.pdf")):
        src = OUTPUT_FIGURES / dst.name
        if src.exists() and src.read_bytes() != dst.read_bytes():
            shutil.copy2(src, dst)
            copied.append(dst.name)
    new = sorted({p.name for p in OUTPUT_FIGURES.glob("*.pdf")} - {p.name for p in PAPER_FIGURES.glob("*.pdf")})
    added = [n for n in new if locate(n)]  # new figures that manuscript.tex or supplementary.tex include
    for n in added:
        shutil.copy2(OUTPUT_FIGURES / n, PAPER_FIGURES / n)
    print("\n".join([f"updated {n}" for n in copied] + [f"added {n}" for n in added])
          or "all figures already up to date")
    unused = [n for n in new if n not in added]
    if unused:
        print(f"\nnot copied (not included by the LaTeX): {', '.join(unused)}")


def _figure_number(name):
    """(document, caption prefix, k) of the figure that includes the image `name` (counting \\begin{figure}): the
    figure's caption is the k-th (from 0) of the PDF starting with that prefix (several "(continued)" parts)."""
    for doc, prefix in (("manuscript", ""), ("supplementary", "S")):
        tex = (LATEX / f"{doc}.tex").read_text()
        n = k = 0
        for env in re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
            continued = "\\ContinuedFloat" in env
            n, k = (n, k + 1) if continued else (n + 1, -1)
            if re.search(r"\\includegraphics(\[[^]]*\])?\{" + re.escape(name) + r"\}", env):
                return doc, f"Figure {prefix}{n}:" + (" (continued)" if continued else ""), max(k, 0)
    return None


def locate(fig):
    """(document, caption prefix, k) of a figure number (7, S3) or image name (fig7_corr_base): the figure's caption
    is the k-th of the PDF starting with that prefix (see find_page); None if not included."""
    if re.fullmatch(r"S?\d+", fig, re.I):
        return ("supplementary" if fig[0] in "sS" else "manuscript"), f"Figure {fig.upper()}:", 0
    return _figure_number(Path(fig).stem)


def build(doc):
    """Build paper/latex/<doc>.tex with latexmk; returns (PDF path, True if the build succeeded).

    latexmk only tracks the image files the last run read. A figure saved for the first time in outputs/figures
    (searched before paper/latex/figures by \\graphicspath) is not one of them, so force a rerun (-g) whenever a
    figure is newer than the PDF.
    """
    pdf = LATEX / f"{doc}.pdf"
    built = pdf.stat().st_mtime if pdf.exists() else 0
    figures = [*OUTPUT_FIGURES.glob("*.pdf"), *OUTPUT_FIGURES.glob("*.png"), *PAPER_FIGURES.glob("*")]
    force = ["-g"] if any(f.stat().st_mtime > built for f in figures) else []
    ok = subprocess.run(["latexmk", "-quiet", *force, f"{doc}.tex"], cwd=LATEX, capture_output=True).returncode == 0
    return pdf, ok


SOURCES = {".tex", ".bib", ".bst", ".cls", ".sty", ".pdf", ".png", ".jpg", ".jpeg", ".eps"}


def sources_stamp():
    """Fingerprint of the files a build reads (LaTeX sources, saved and tracked figures): it changes whenever one of
    them is added, edited or removed, so an unchanged stamp means the last built PDF is still current."""
    files = [f for f in [*LATEX.rglob("*"), *OUTPUT_FIGURES.glob("*")]
             if f.is_file() and f.suffix.lower() in SOURCES and not (f.parent == LATEX and f.suffix == ".pdf")]
    return hash(tuple(sorted((str(f), f.stat().st_mtime_ns) for f in files)))


def find_page(pdf, caption, k=0):
    """1-based page of the PDF with the k-th (from 0) line starting with `caption`, or None."""
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    pages = [i for i, p in enumerate(text.split("\f"), 1)
             for _ in re.findall(r"^\s*" + re.escape(caption), p, re.M)]
    return pages[k] if k < len(pages) else None


def render_page(pdf, page, dpi=80):
    """PNG bytes of one page of the PDF (pdftoppm)."""
    return subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", str(dpi), "-f", str(page), "-l", str(page),
                           pdf], capture_output=True, check=True).stdout


def show(args):
    found = locate(args.figure) if args.figure else ("manuscript", None, 0)
    if found is None:
        sys.exit(f"{args.figure}: not included by manuscript.tex or supplementary.tex")
    doc, caption, k = found
    pdf, ok = build(doc)
    if not ok:
        print(f"latexmk failed: see paper/latex/{doc}.log (opening the last built PDF)")
    page = find_page(pdf, caption, k) if caption else None
    if caption:
        print(f"{caption.rstrip(':')} on page {page}" if page else f"{caption.rstrip(':')} not found in {pdf.name}")
    cmd = (["evince", "-i", str(page), pdf] if page and shutil.which("evince") else ["xdg-open", pdf])
    subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def commit(args):
    git("add", *[p for p in TRACKED if (ROOT / p).exists()])
    staged = git("diff", "--cached", "--name-only").split()
    if not staged:
        sys.exit("nothing to commit (did you run `figures save`?)")
    data = [f for f in staged if DATA_FILE.search(f) and f not in ALLOWED_DATA]
    hits = [l for l in git("diff", "--cached", "--text").splitlines()
            if l.startswith("+") and IDENTIFIERS.search(l)]
    if data or hits:
        print("refusing to commit (possible participant data):", *data, *hits[:20], sep="\n  ")
        sys.exit("unstage with `git restore --staged <file>` and check AGENTS.md")
    print("staged:", *staged, sep="\n  ")
    git("commit", "-m", args.message)
    if not args.no_push:
        sh("git", "push", cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(prog="figures", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install", help="create .venv and register the fsvv kernel").set_defaults(func=install)
    sub.add_parser("run", help="open the tuning notebooks in JupyterLab").set_defaults(func=run)
    s = sub.add_parser("show", help="build the paper and open it at a figure")
    s.add_argument("figure", nargs="?", help="figure number (7, S3) or image name (fig7_corr_base)")
    s.set_defaults(func=show)
    sub.add_parser("save", help="copy saved PDFs into paper/latex/figures").set_defaults(func=save)
    c = sub.add_parser("commit", help="check, commit and push the figures")
    c.add_argument("-m", "--message", default="Figure tuning")
    c.add_argument("--no-push", action="store_true")
    c.set_defaults(func=commit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
