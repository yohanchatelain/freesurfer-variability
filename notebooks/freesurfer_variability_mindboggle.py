# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# ---- Paths (revision workspace) ----
# Inputs are read from ../data (symlinks to $PROJECT_DIR, see ../data/README.md).
# Everything this notebook writes goes to ../outputs. Never write into ../data: the links point to
# the original files, which are writable.
from pathlib import Path

DATA = Path("../data").resolve()
OUT = Path("../outputs").resolve()
OUT.mkdir(exist_ok=True)
INPUT_SUBDIR = "tables"  # default ../data subdirectory for files written by the original pipeline


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


# %%
import pandas as pd
import json

df = pd.read_csv(str(DATA / "mindboggle" / "mindboggle101_atlases.txt"), header=None, names=["PATNO_id"])

df["File name"] = df["PATNO_id"] + "labels"

# %%
df = pd.read_csv(inp("stage_two_clinical.csv"))
# Load text file (one PATNO per line)
with open(inp("orig.txt"), "r") as f:
    patnos = {line.strip().rstrip(",") for line in f}

# Make PATNO comparable
df["PATNO_id"] = df["PATNO_id"].astype(str)

# Remove rows not in TXT
df_filtered = df[df["PATNO_id"].isin(patnos)].copy()

# Find PATNOs in TXT that are missing from dataframe
missing_patnos = patnos - set(df_filtered["PATNO_id"])

print(f"Original dataframe rows: {len(df)}")
print(f"Filtered dataframe rows: {len(df_filtered)}")
print(f"Missing PATNOs from dataframe: {len(missing_patnos)}")

print(sorted(missing_patnos))
