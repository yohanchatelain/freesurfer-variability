
# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920892"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mri_glmfit_1con.json")

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_long_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_long.mtx'
                                       )
        print(out_fs)
