
# save preprocessing script to submit jobs to the server later
# copy your FreeSurfer license to FS_license/license.txt
# or update the license path below

import os
import json
import shutil

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function

# zid = "zenodo.20837336"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_2_0.json")

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open("failed_subjects.json") as fin:
    subject_map = json.load(fin)

# Extract info from json
subjid = subject_map["PATNO_id"][task_id]
input_file = subject_map["File name"][task_id]

# Set output directory for FreeSurfer outputs
subjects_dir = f"/home/vinuyans/links/projects/rrg-glatard/vinuyans/data/FS8"
sub_path = os.path.join(subjects_dir, subjid)
os.environ["SUBJECTS_DIR"] = subjects_dir
# Removing old subject folder if it exists to avoid conflicts
if os.path.exists(sub_path):
    print(f"Removing old subject folder: {subjid}")
    shutil.rmtree(sub_path)

out_fs = freesurfer(
    "--imagepath",
    "freesurfer_8.2.0.sif",
    input=input_file,
    qcache_flag=True,
    license="FS_license/license.txt",
    subjid=subjid,
)
print(out_fs)
