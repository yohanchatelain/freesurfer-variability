
# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920888"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mris_preproc.json")

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                       )
        print(out_fs)
