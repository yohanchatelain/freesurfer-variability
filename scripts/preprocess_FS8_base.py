
# Step 2. Create an unbiased template from all time points
# for each subject using FreeSurfer 8 longitudinal processing.

import os
import json
import shutil

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function

freesurfer = function("./boutiques/freesurfer_8_recon_base.json")

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

# with open("json_data_base.json") as fin:
with open("failed_subjects_base.json") as fin:
    subject_map = json.load(fin)

# Output location for FreeSurfer subjects
subjects_dir = "/home/vinuyans/links/projects/rrg-glatard/vinuyans/data/FS8"
os.environ["SUBJECTS_DIR"] = subjects_dir

base_subject = subject_map["PATNO_base"][task_id]
base_path = os.path.join(subjects_dir, base_subject)

# Remove any previous base subject to avoid recon-all errors
if os.path.exists(base_path):
    print(f"Removing old base subject: {base_subject}")
    shutil.rmtree(base_path)

out_fs = freesurfer(
    "--imagepath",
    "freesurfer_8.2.0.sif",
    tp1=subject_map["first_visit"][task_id],
    tp2=subject_map["second_visit"][task_id],
    outputdir=base_subject,
    license="FS_license/license.txt"
)

print(out_fs)
