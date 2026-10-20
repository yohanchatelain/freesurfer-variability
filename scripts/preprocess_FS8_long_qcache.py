
# save proprocessing script to submit jobs to the server later 

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920876"
#bosh(["exec", "prepare", zid])
freesurfer = function("./boutiques/freesurfer_8_recon_qcache.json")

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)


out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id],
                                        license="FS_license/license.txt",
                                       )
print(out_fs)
