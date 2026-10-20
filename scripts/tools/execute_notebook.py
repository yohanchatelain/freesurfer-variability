"""Execute a notebook cell by cell, skipping pipeline cells, and report errors.

Usage: python exec_nb.py <in.ipynb> <executed_out.ipynb> <report.json> [out_subdir]
Run from the notebook's directory (relative ../data, ../outputs paths).
"""
import json
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src, dst, report_path = sys.argv[1:4]
out_subdir = sys.argv[4] if len(sys.argv) > 4 else None

nb = nbformat.read(src, as_version=4)
import os
OVERRIDES = json.load(open(os.environ['EXEC_OVERRIDES'])) if os.environ.get('EXEC_OVERRIDES') else {}

SKIP_LINE = re.compile(r"^(\s*)(.*(utils\.notebook_init\(\)|get_study_files\(|run\.mri_metadata\(\)|"
                       r"LivingParkUtils\(\)|ppmi\.Downloader\(|^%load_ext|^%matplotlib widget|Throw new error).*)$")

in_pipeline = False
skipped = []
for i, c in enumerate(nb.cells):
    text = c.source
    if c.cell_type == "markdown":
        head = text.strip().splitlines()[0] if text.strip() else ""
        if head.startswith("# Image preprocessing"):
            in_pipeline = True
        elif head.startswith("# ") and in_pipeline and "preprocessing" not in head.lower() \
                and not head.startswith("# Freesurfer") and not head.startswith("# Longitudinal"):
            in_pipeline = False
        continue
    c.outputs = []
    if text.lstrip().startswith("%%") or in_pipeline:
        c.source = "# [exec] pipeline cell skipped\n"
        skipped.append(i)
        continue
    lines = []
    for ln in text.splitlines():
        lines.append(SKIP_LINE.sub(r"\1# [exec] skipped: \2", ln) if SKIP_LINE.match(ln) else ln)
    c.source = "\n".join(lines)
    if str(i) in OVERRIDES:
        c.source = OVERRIDES[str(i)]
    if i == 0 and out_subdir:
        c.source = c.source.replace('OUT = Path("../outputs").resolve()',
                                    f'OUT = Path("../outputs/{out_subdir}").resolve()')
        c.source = c.source.replace("OUT.mkdir(exist_ok=True)", "OUT.mkdir(parents=True, exist_ok=True)")

client = NotebookClient(nb, timeout=7200, kernel_name="fsvv", allow_errors=True,
                        resources={"metadata": {"path": "."}})
report = {"skipped_pipeline_cells": skipped, "cells": []}
with client.setup_kernel():
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code" or c.source.startswith("# [exec] pipeline cell skipped"):
            continue
        t0 = time.time()
        try:
            client.execute_cell(c, i)
        except CellExecutionError as e:  # not raised with allow_errors, kept for safety
            pass
        dt = time.time() - t0
        err = next((o for o in c.outputs if o.get("output_type") == "error"), None)
        entry = {"cell": i, "sec": round(dt, 1), "first_line": c.source.strip().splitlines()[0][:80] if c.source.strip() else ""}
        if err:
            entry["error"] = f"{err['ename']}: {err['evalue'][:300]}"
        report["cells"].append(entry)
        print(f"[{i}] {dt:6.1f}s {'ERROR ' + entry['error'][:150] if err else 'ok'}", flush=True)
        json.dump(report, open(report_path, "w"), indent=1)

nbformat.write(nb, dst)
n_err = sum("error" in e for e in report["cells"])
print(f"done: {len(report['cells'])} cells executed, {n_err} errors, {len(skipped)} pipeline cells skipped")
