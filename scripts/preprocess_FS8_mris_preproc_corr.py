
# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920888"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mris_preproc.json")

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi, group=group),
                                       )
        print(out_fs)
