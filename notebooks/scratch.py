# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv (3.11.4)
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


# %% [markdown]
# # Check Run Completion

# %% [markdown]
# ## Step 1 Completion

# %% [markdown]
# ### Config

# %%
JSON_PATH = inp("json_data.json")   # path to your subject list
FS_ROOT   = str(DATA / "FS8")         # root FreeSurfer output directory

# %% [markdown]
# ### Imports

# %%
import json
from pathlib import Path
import pandas as pd

# %% [markdown]
# ### Required Files

# %%
# Files that must exist for a successful run
REQUIRED_FILES = [
    "scripts/recon-all.done",
    "mri/aparc+aseg.mgz",
    "mri/brain.mgz",
    "mri/wm.mgz",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
    "stats/aseg.stats",
]

# Explicit error flag written by recon-all on failure
FAILURE_FLAGS = ["scripts/recon-all.error"]

# Keywords to search for in the status log
LOG_FILE       = "scripts/recon-all-status.log"
ERROR_KEYWORDS = ["ERROR", "FAILED", "Segmentation fault", "killed", "Killed"]

# %%
with open(JSON_PATH) as fh:
    data = json.load(fh)

patno_ids = list(data["PATNO_id"].values())
fs_root   = Path(FS_ROOT)

print(f"Loaded {len(patno_ids)} subjects")
print(f"FreeSurfer root: {fs_root.resolve()}")


# %%
def check_subject(patno_id: str) -> dict:
    subject_dir = fs_root / patno_id   # adjust if your dir names differ

    if not subject_dir.exists():
        return {"subject": patno_id, "status": "MISSING_DIR",
                "missing_files": [], "error_flags": [], "log_errors": []}

    error_flags   = [f for f in FAILURE_FLAGS   if (subject_dir / f).exists()]
    missing_files = [f for f in REQUIRED_FILES  if not (subject_dir / f).exists()]

    log_errors = []
    log_path = subject_dir / LOG_FILE
    if log_path.exists():
        with open(log_path, errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if any(kw in line for kw in ERROR_KEYWORDS):
                    log_errors.append(f"line {i}: {line.rstrip()}")

    if error_flags or log_errors:
        status = "ERROR"
    elif "scripts/recon-all.done" in missing_files:
        status = "INCOMPLETE"
    elif missing_files:
        status = "PARTIAL"
    else:
        status = "OK"

    return {"subject": patno_id, "status": status,
            "missing_files": missing_files,
            "error_flags": error_flags,
            "log_errors": log_errors[:5]}   # cap log lines per subject


results = [check_subject(pid) for pid in patno_ids]
print(f"Checked {len(results)} subjects.")

# %%
df = pd.DataFrame([{
    "subject":       r["subject"],
    "status":        r["status"],
    "missing_files": len(r["missing_files"]),
    "error_flags":   len(r["error_flags"]),
    "log_errors":    len(r["log_errors"]),
} for r in results])

# Colour-code by status
STATUS_COLOURS = {
    "OK":          "background-color: #d4edda",
    "INCOMPLETE":  "background-color: #fff3cd",
    "PARTIAL":     "background-color: #fde8c8",
    "ERROR":       "background-color: #f8d7da",
    "MISSING_DIR": "background-color: #e2e3e5",
}

def colour_row(row):
    style = STATUS_COLOURS.get(row["status"], "")
    return [style] * len(row)

df.style.apply(colour_row, axis=1)

# %%
summary = df.groupby("status").size().rename("count").reset_index()
summary["pct"] = (summary["count"] / len(df) * 100).round(1)
print(summary.to_string(index=False))

# %%
failed = [r for r in results if r["status"] != "OK"]

if not failed:
    print("✓ All subjects completed successfully — nothing to report.")
else:
    for r in failed:
        print(f"\n{'='*60}")
        print(f"  Subject : {r['subject']}")
        print(f"  Status  : {r['status']}")
        if r["error_flags"]:
            print("  Error flags:")
            for f in r["error_flags"]:
                print(f"    ⚑  {f}")
        if r["missing_files"]:
            print("  Missing files:")
            for f in r["missing_files"]:
                print(f"    ✗  {f}")
        if r["log_errors"]:
            print("  Log errors (first 5):")
            for line in r["log_errors"]:
                print(f"    ⚠  {line}")

# %%
failed_ids = [r["subject"] for r in results if r["status"] != "OK"]

if failed_ids:
    # Preserve all original keys for failed subjects, re-indexed from 0
    failed_indices = [i for i, pid in enumerate(patno_ids) if pid in set(failed_ids)]

    failed_json = {}
    for key in data.keys():
        failed_json[key] = {
            str(new_i): data[key][str(orig_i)]
            for new_i, orig_i in enumerate(failed_indices)
            if str(orig_i) in data[key]
        }

    out_path = Path(out("failed_subjects.json"))
    with open(out_path, "w") as fh:
        json.dump(failed_json, fh, indent=4)
    print(f"Wrote {len(failed_ids)} failed subject(s) to {out_path}")
else:
    print("No failed subjects — nothing to export.")

# %% [markdown]
# # Step 2

# %%
# #!/usr/bin/env python3

import json
from pathlib import Path

# Paths
JSON_FILE = inp("json_data_base.json")      # Input JSON
FS8_DIR = Path(str(DATA / "FS8"))       # FS8 subjects directory
OUTPUT_JSON = out("failed_subjects_base.json")

# Files that should exist for a completed FS8 base reconstruction
REQUIRED_FILES = [
    "mri/brain.mgz",
    "surf/lh.white",
    "surf/rh.white",
    "stats/aseg.stats",
]

# Load JSON
with open(JSON_FILE, "r") as f:
    data = json.load(f)

failed_indices = []

# Check each base subject
for idx, base in data["PATNO_base"].items():
    base_dir = FS8_DIR / base

    if not base_dir.is_dir():
        failed_indices.append(idx)
        continue

    if any(not (base_dir / f).exists() for f in REQUIRED_FILES):
        failed_indices.append(idx)

# Keep only failed subjects in every field
failed_data = {}
for key in data:
    failed_data[key] = {
        idx: data[key][idx]
        for idx in failed_indices
    }

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(failed_data, f, indent=4)

print(f"Found {len(failed_indices)} failed base subjects.")
print(f"Saved to {OUTPUT_JSON}")

# %%
import os
import json

subjects_dir = str(DATA / "FS8")

with open(inp("json_data_long.json")) as f:
    data = json.load(f)

def check_subject(subject_path):
    required_files = [
        "scripts/recon-all.done",
        "stats/aseg.stats",
        "surf/lh.white",
        "surf/rh.white"
    ]

    missing = []
    for rel in required_files:
        path = os.path.join(subject_path, rel)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(rel)

    return missing

success = []
failed = []
missing_subjects = []

for i in data["visit"]:
    tp = data["visit"][i]
    base = data["PATNO_base"][i]

    subject = f"{tp}.long.{base}"
    subject_path = os.path.join(subjects_dir, subject)

    if not os.path.exists(subject_path):
        missing_subjects.append(subject)
        continue

    missing_files = check_subject(subject_path)

    if len(missing_files) == 0:
        success.append(subject)
    else:
        failed.append((subject, missing_files))

# ---------------- REPORT ----------------
print("\n===== FS8 LONG QC (4-file check) =====")
print(f"Total expected: {len(data['visit'])}")
print(f"Success: {len(success)}")
print(f"Failed: {len(failed)}")
print(f"Missing folders: {len(missing_subjects)}")

print("\n--- FAILED SUBJECTS ---")
for s, files in failed:
    print(f"\n{s}")
    for f in files:
        print(f"  - missing: {f}")

print("\n--- MISSING SUBJECTS ---")
for s in missing_subjects:
    print(s)

# Save structured output
failed_long = {
    "success": success,
    "failed": [{"subject": s, "missing": f} for s, f in failed],
    "missing_folders": missing_subjects
}

with open(out("failed_subjects_long.json"), "w") as f:
    json.dump(failed_long, f, indent=4)

print("\nSaved: failed_subjects_long.json")

# %% [markdown]
# ## Qcache

# %%
import os
import json

SUBJECTS_DIR = str(DATA / "FS8")

INPUT_JSON = inp("json_data_long.json")
OUTPUT_JSON = out("failed_subjects_long_qcache.json")

with open(INPUT_JSON) as f:
    subject_map = json.load(f)

failed = {k: {} for k in subject_map}

n = len(subject_map["visit"])

for i in range(n):
    tp = subject_map["visit"][str(i)]
    base = subject_map["PATNO_base"][str(i)]

    subject = f"{tp}.long.{base}"
    subject_dir = os.path.join(SUBJECTS_DIR, subject)

    ok = True

    if not os.path.isdir(subject_dir):
        ok = False
    else:
        log = os.path.join(subject_dir, "scripts", "recon-all.log")

        if not os.path.exists(log):
            ok = False
        else:
            with open(log, "r", errors="ignore") as f:
                log_text = f.read()

            if "finished without error" not in log_text:
                ok = False

        surf = os.path.join(subject_dir, "surf")

        required_files = [
            "lh.thickness.fwhm25.fsaverage.mgh",
            "rh.thickness.fwhm25.fsaverage.mgh",
        ]

        for fname in required_files:
            if not os.path.exists(os.path.join(surf, fname)):
                ok = False
                break

    if not ok:
        for key in subject_map:
            failed[key][str(len(failed[key]))] = subject_map[key][str(i)]

with open(OUTPUT_JSON, "w") as f:
    json.dump(failed, f, indent=4)

print(f"Checked {n} subjects.")
print(f"Failed: {len(failed['visit'])}")
print(f"Results written to {OUTPUT_JSON}")
