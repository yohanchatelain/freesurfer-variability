"""Build the job lists for the participants of the extended clinical cohort that FS5-7 never processed.

Cohort: vinuyans' `stage_two_clinical.csv` (455 PD-non-MCI/HC with UPDRS-III OFF at two visits, used for the
FS8 runs). The 286 participants without a FS5-7 base template are new; the 170 FS8 base templates that exist for
them all failed (their cross-sectional runs were never done), so they are re-run in every version, FS8 included.

Usage (login node, seconds): python make_lists.py
Writes lists/ next to this script:
  new_participants.csv   one row per participant (task index = row number), visits, image IDs, years between visits
  loni_image_ids.txt     comma-separated image IDs to paste in the LONI IDA search (PPMI download)
  qdec/<PATNO>.dat       per-participant qdec table for long_mris_slopes
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
LISTS = HERE / "lists"

cohort = pd.read_csv(DATA / "tables" / "stage_two_clinical.csv")
done = {int(d.name.split("_")[0][4:]) for d in (DATA / "FS7").iterdir() if d.name.endswith("_base")}
new = cohort[~cohort["PATNO"].isin(done)].sort_values("PATNO").reset_index(drop=True)

jobs = pd.DataFrame({
    "PATNO": new["PATNO"],
    "dx_group": new["dx_group"],
    "visit1": new["EVENT_ID"],
    "visit2": new["EVENT_ID_NX"],
    "image_id1": new["Image ID"].astype(int),
    "image_id2": new["Image ID_NX"].astype(int),
    "years": new["durationT2_T1_y"],
    "field_strength": new["Imaging Protocol"].str.extract(r"Field Strength=([\d.]+)")[0],
})
jobs["base"] = "sub-" + jobs["PATNO"].astype(str) + "_base"
jobs["tp1"] = "sub-" + jobs["PATNO"].astype(str) + "_ses-" + jobs["visit1"]
jobs["tp2"] = "sub-" + jobs["PATNO"].astype(str) + "_ses-" + jobs["visit2"]

LISTS.mkdir(exist_ok=True)
jobs.to_csv(LISTS / "new_participants.csv", index_label="task")
ids = pd.concat([jobs["image_id1"], jobs["image_id2"]]).astype(str)
(LISTS / "loni_image_ids.txt").write_text(",".join(ids) + "\n")

(LISTS / "qdec").mkdir(exist_ok=True)
for r in jobs.itertuples():
    (LISTS / "qdec" / f"{r.PATNO}.dat").write_text(
        "fsid fsid-base years dx_group\n"
        f"{r.tp1} {r.base} 0.0 {r.dx_group}\n"
        f"{r.tp2} {r.base} {r.years} {r.dx_group}\n")

print(f"{len(jobs)} new participants ({jobs['dx_group'].value_counts().to_dict()}), {len(ids)} images")
print("visit pairs:", (jobs["visit1"] + "-" + jobs["visit2"]).value_counts().to_dict())
print("field strength:", jobs["field_strength"].value_counts().to_dict())
