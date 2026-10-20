"""Git clean filter: strip outputs and execution counts from a notebook read on stdin.

Notebook outputs can show PPMI participant-level data, which must never be committed.
Registered in .gitattributes; enable once per clone with:
    git config filter.strip-outputs.clean "python3 scripts/tools/strip_notebook_outputs.py"
    git config filter.strip-outputs.smudge cat
    git config filter.strip-outputs.required true
"""
import json
import sys

nb = json.load(sys.stdin)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell.get("metadata", {}).pop("execution", None)
json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")
