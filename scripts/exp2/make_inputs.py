"""Subject list and T1w inputs for the Mindboggle-101 runs (Exp 2).

Writes subjects.csv (array index = task) next to this script and copies each t1weighted.nii.gz to
$FSVV_MB_WORK/inputs/<subject>.nii.gz (inside the folder bosh mounts in the container).
Usage: python make_inputs.py
"""
import os
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MB = ROOT / "data/mindboggle/subjects"
WORK = Path(os.environ.get("FSVV_MB_WORK", Path(os.environ["SCRATCH"]) / "fsvv/mindboggle"))

subjects = sorted(p.name for p in MB.iterdir() if (p / "t1weighted.nii.gz").exists())
df = pd.DataFrame({"subject": subjects, "dataset": [s.rsplit("-", 1)[0] for s in subjects]})
df.index.name = "task"
df.to_csv(HERE / "subjects.csv")
(WORK / "inputs").mkdir(parents=True, exist_ok=True)
for s in subjects:
    dst = WORK / "inputs" / f"{s}.nii.gz"
    if not dst.exists():
        shutil.copy2(MB / s / "t1weighted.nii.gz", dst)
print(f"{len(subjects)} subjects -> {HERE / 'subjects.csv'}, inputs in {WORK / 'inputs'}")
print(df["dataset"].value_counts().to_string())
