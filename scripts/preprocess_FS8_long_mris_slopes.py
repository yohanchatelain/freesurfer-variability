
# Prepare the data with long_mris_slopes for longitudinal two stage model

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
# zid = "zenodo.7920880"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_long_mris_slopes.json")

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                        license="FS_license/license.txt",
                                        qdec='qdec_long_groups.dat',
                                        meas='thickness',
                                        hemi=hemi,
                                        time='years',
                                        stack_avg='{hemi}.long.thickness-avg.stack.mgh'.format(hemi=hemi),
                                        stack_rate='{hemi}.long.thickness-rate.stack.mgh'.format(hemi=hemi),
                                        stack_pc1fit='{hemi}.long.thickness-pc1fit.stack.mgh'.format(hemi=hemi),
                                        stack_pc1='{hemi}.long.thickness-pc1.stack.mgh'.format(hemi=hemi),
                                        stack_spc='{hemi}.long.thickness-spc.stack.mgh'.format(hemi=hemi),
                                       )
