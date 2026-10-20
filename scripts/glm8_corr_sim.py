
# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920896"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mri_glmfit_sim_abs.json")

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        dir='results_corr_{time}_{group}_{hemi}'.format(time=time, group=group, hemi=hemi),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )
            print(out_fs)
