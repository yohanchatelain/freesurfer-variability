
# Step 3. "-long" longitudinal processing (FreeSurfer 8 recon-all -long)

import os
import json
import shutil

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function

freesurfer = function("./boutiques/freesurfer_8_recon_long.json")

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open("json_data_long.json") as fin:
    subject_map = json.load(fin)

# Output location for FreeSurfer subjects
subjects_dir = "/home/vinuyans/links/projects/rrg-glatard/vinuyans/data/FS8"
os.environ["SUBJECTS_DIR"] = subjects_dir

# Extract inputs
tp = subject_map["visit"][task_id]
base = subject_map["PATNO_base"][task_id]

# ---- Safety: ensure base exists ----
base_path = os.path.join(subjects_dir, base)
if not os.path.exists(base_path):
    raise RuntimeError(f"Base subject not found: {base_path}")

# ---- Safety: remove stale LONG output if it exists ----
# recon-all -long outputs use: TP + "_long_" + BASE
long_subject = f"{tp}.long.{base}"
long_path = os.path.join(subjects_dir, long_subject)

if os.path.exists(long_path):
    print(f"Removing old long subject: {long_subject}")
    shutil.rmtree(long_path)

# # Run longitudinal recon-all -long
out_fs = freesurfer(
    "--imagepath",
    "freesurfer_8.2.0.sif",
    tp=tp,
    base=base,
    license="FS_license/license.txt",
)

print(out_fs)
