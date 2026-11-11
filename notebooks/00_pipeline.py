# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.11 (fsvv)
#     language: python
#     name: fsvv
# ---

# %% [markdown]
# # 00 · Processing pipeline (reference only — do not run)
#
# FreeSurfer 5.3 / 6.0.1 / 7.3.2 / 8.2.0 processing of the PPMI cohort as it was run in `$PROJECT_DIR` (paths in these cells are relative to that directory):
#
# 1. cross-sectional `recon-all` (all images) → `data/FSx/sub-*_ses-*`
# 2. longitudinal stream (base template, longitudinal time points, `-qcache`) → `data/FSx/*.long.*`
# 3. `long_mris_slopes`, FSGD files, `mris_preproc`, `mri_glmfit`, `mri_glmfit-sim` → `data/glm_results`
# 4. `mri_seg_overlap` Dice between versions → `data/stats/dice`
#
# Outputs are already available through `data/` (see `data/README.md`). **Do not execute these cells from this workspace**: `preprocess_FS8*.py` deletes existing subject folders (`shutil.rmtree`) and the data links point to the original outputs. Kernel cells that build FSGD files need the cohort variables of `01_cohort`.

# %% [markdown]
# # Image preprocessing
#
# Data is preprocessed using Freesurfer's recon-all.
#
# Analyses were performed on Compute Canada servers (Advanced Research Computing facilities provided by the Compute Canada Federation). If you don't have Compute Canada account you may be able to request one [here](https://ccdb.computecanada.ca/security/login).
#
# Otherwise, please use any other available server or your local machine to run the analyses. You may need to adjust the following code depending on the method you use.
#
# <a href=https://github.com/boutiques> Boutiques</a> is used to manage Freesurfer's functions within a container. 

# %% [markdown]
# > **Revision workspace note.** The `%%writefile` / `%%sbatch` cells in the processing sections are the original pipeline. They were run in `$PROJECT_DIR` and use paths relative to that directory. **Don't run them from this workspace**: `preprocess_FS8*.py` deletes existing subject folders (`shutil.rmtree`), and `data/FS8` links to the original outputs. Re-runs go through `../scripts/` with outputs on `$SCRATCH`.

# %%
# %load_ext slurm_magic

# %%
import pandas as pd
import json

# read csv (single column, no header assumed)
df = pd.read_csv(inp("subjects.csv"), header=None, names=["id"])

# split into PATNO and EVENT_ID
df["PATNO"] = df["id"].str.extract(r"sub-(\d+)_ses-")[0]
df["EVENT_ID"] = df["id"].str.extract(r"_ses-(.*)")[0]

# build subject ID if you still want it
df["PATNO_id"] = "sub-" + df["PATNO"] + "_ses-" + df["EVENT_ID"]

# build file path
df["File name"] = (
    str(DATA) + "/FS5/sub-"
    + df["PATNO"]
    + "_ses-"
    + df["EVENT_ID"]
    + "/mri/orig.mgz"
)

# keep only needed columns
small_df = df[["PATNO_id", "File name"]]

# convert to JSON (same structure style you were using)
json_data = small_df.to_json()

meta = json.loads(json_data)

with open(out("json_data.json"), "w") as fout:
    json.dump(meta, fout, indent=4)

print("Saved json_data.json")

# %% [markdown]
# ## Step 1 - preprocess all images cross-sectionally

# %% [markdown]
# ## Freesurfer 5 - preprocessing step 1 - cross-sectional

# %%
# %%writefile preprocess_FS5.py

# save proprocessing script to submit jobs to the server later 
# # copy your FreeSurfer license to FS_license/license.txt or update the license path below

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7738457"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])
license = str(os.environ["FS_LICENSE"]) 

with open('json_data.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                    '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-633
# #!/bin/bash
#SBATCH --job-name=FS5_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc.%a.out
#SBATCH --error=logs/FS5_preproc.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5.py

# %% [markdown]
# ## Freesurfer 6 - preprocessing step 1 - cross-sectional

# %%
# %%writefile preprocess_FS6.py

# save proprocessing script to submit jobs to the server later 
# # copy your FreeSurfer license to FS_license/license.txt or update the license path below

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7542266"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-633
# #!/bin/bash
#SBATCH --job-name=FS6_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc.%a.out
#SBATCH --error=logs/FS6_preproc.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS6.py

# %% [markdown]
# ## Freesurfer 7 - preprocessing step 1 - cross-sectional

# %%
# %%writefile preprocess_FS7.py

# save proprocessing script to submit jobs to the server later 
# # copy your FreeSurfer license to FS_license/license.txt or update the license path below

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7545769"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-633
# #!/bin/bash
#SBATCH --job-name=FS7_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_preproc.%a.out
#SBATCH --error=logs/FS7_preproc.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7.py

# %% [markdown]
# ## Freesurfer 8 - preprocessing step 1 - cross-sectional

# %%
# %%writefile preprocess_FS8.py

# save preprocessing script to submit jobs to the server later
# # copy your FreeSurfer license to FS_license/license.txt
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
subjects_dir = f"$PROJECT_DIR/data/FS8"
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
    env={"SUBJECTS_DIR": subjects_dir},
)
print(out_fs)

# %%
# %%sbatch --array=0-2
# #!/bin/bash
#SBATCH --job-name=FS8_preproc
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/step1/FS8_preproc.%a.out
#SBATCH --error=logs/step1/FS8_preproc.%a.err
#SBATCH --time=6:00:00
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8.py


# %% [markdown]
# # Longitudinal preprocessing
#
#
# Data from the clinical cohort are preprocessed using longitudinal stream implemented in Freesurfer. <a href=https://surfer.nmr.mgh.harvard.edu/fswiki/LongitudinalProcessing> Click here for details.</a> 

# %%
# combine HC + PD UPDRS cohorts

hc_only = cohort[cohort["dx_group"] == "HC"]

stage_two = pd.concat([pd_clinical, hc_only])

# get IDs for the first and second visit
stage_two["first_visit"] = (
    "sub-" + stage_two["PATNO"].astype(str) + "_ses-" + stage_two["EVENT_ID"]
)
stage_two["second_visit"] = (
    "sub-" + stage_two["PATNO"].astype(str) + "_ses-" + stage_two["NEXT_VISIT"]
)

# exclude failed images
# failed_long = pd.read_csv(inp("failed_long.csv"))

# for i in failed_long["PATNO"]:
#     stage_two = stage_two[stage_two["PATNO"].astype(str).str.contains(f"{i}") == False]

# %%
stage_two.to_csv(out("stage_two_clinical.csv"), index=False)

# %% [markdown]
# ## Freesurfer 5 - preprocessing step 2 - base template

# %%
# save df with a single input for each subject as json
# double check if the visit pairs are correct!

import json

data_to_process = stage_two
data_to_process = data_to_process.reset_index()
data_to_process["PATNO"] = data_to_process["PATNO"].astype(str)
data_to_process["PATNO_base"] = "sub-" + data_to_process["PATNO"] + "_base"
small_df = data_to_process[["first_visit", "second_visit", "PATNO_base"]]
json_data = small_df.to_json()
meta = json.loads(json_data)
with open(out("json_data_base.json"), "w") as fout:
    json.dump(meta, fout, indent=4)

# %%
# %%writefile preprocess_FS5_base.py

# Step 2. create an unbiased template from all time points for each subject and process it with recon-all:

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7868966"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])
license = str(os.environ["FS_LICENSE"]) 

with open('json_data_base.json') as fin:
    subject_map = json.load(fin)

    
    
out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                    '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        tp1=subject_map["first_visit"][task_id],
                                        tp2=subject_map["second_visit"][task_id],
                                        outputdir=subject_map["PATNO_base"][task_id],
                                       )
print(out_fs)

# %%
# %%sbatch --array=0-249
# #!/bin/bash
#SBATCH --job-name=FS5_base
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_base.%a.out
#SBATCH --error=logs/FS5_preproc_base.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_base.py

# %% [markdown]
# ## Freesurfer 5 - preprocessing step 3 - longitudinally processed timepoints

# %%
# save json fine for longitudinal processing
# first arg = all timepoints, second arg = base directory
# double check if the visit and base are are correct

df_long1 = pd.DataFrame(small_df).rename(columns={"first_visit": "visit"})
df_long2 = pd.DataFrame(small_df).rename(columns={"second_visit": "visit"})
df_both = [df_long1, df_long2]
df_long = pd.concat(df_both, ignore_index=True)
df_long = df_long[["visit", "PATNO_base"]]
json_data = df_long.to_json()
meta = json.loads(json_data)
with open(out("json_data_long.json"), "w") as fout:
    json.dump(meta, fout, indent=4)

# %%
# %%writefile preprocess_FS5_long.py

# Step 3. "-long" longitudinally process all timepoints (recon-all -long):

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7884225"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])
license = str(os.environ["FS_LICENSE"]) 

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                    '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id]
                                       )


print(out_fs)

# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS5_long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_long.%a.out
#SBATCH --error=logs/FS5_preproc_long.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_long.py

# %% [markdown]
# ## Freesurfer 5 - preprocessing step 4 - Qcache¶

# %%
# %%writefile preprocess_FS5_long_qcache.py

# save proprocessing script to submit jobs to the server later 

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7884255"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])
license = str(os.environ["FS_LICENSE"]) 

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                    '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id],
                                       )


# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS5_longQcache
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_longQcache.%a.out
#SBATCH --error=logs/FS5_preproc_longQcache.%a.err
#SBATCH --time=2:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_long_qcache.py

# %% [markdown]
# ## FS5 - Prepare the vertex data with long_mris_slopes for longitudinal two stage model

# %%
# Create longitudinal QDEC table

stage_two["PATNO_base"] = "sub-" + stage_two["PATNO"].astype(str) + "_base"

df_1 = stage_two[["first_visit", "PATNO_base", "durationT2_T1_y", "dx_group"]]
df_1.rename(columns={"first_visit": "visit"}, inplace=True)
df_2 = stage_two[["second_visit", "PATNO_base", "durationT2_T1_y", "dx_group"]]
df_2.rename(columns={"second_visit": "visit"}, inplace=True)

df_all = [df_1, df_2]
df_all = pd.concat(df_all, ignore_index=True)
df_all = df_all.sort_values("visit")
df_all = df_all.reset_index()

qdec_table = df_all

qdec_table["fsid"] = qdec_table["visit"]
qdec_table["fsid-base"] = qdec_table["PATNO_base"]
qdec_table["years"] = np.where(
    qdec_table.index % 2 == 0, 0, qdec_table["durationT2_T1_y"]
)
qdec_table = qdec_table[["fsid", "fsid-base", "years", "dx_group"]]

qdec_table.to_csv(out("qdec_long_groups.dat"), sep=" ", index=False)

# %%
# %%writefile preprocess_FS5_long_mris_slopes.py

# Prepare the data with long_mris_slopes for longitudinal two stage model

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893178"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                    '--imagepath', 'ansokol-freesurfer_5.3.simg',
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


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_slopes
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_slopes.out
#SBATCH --error=logs/FS5_mris_slopes.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_long_mris_slopes.py

# %% [markdown]
# ## FS 5 - Create FSGD file for the group analysis - baseline - t-test

# %%
df_pdmci_first = stage_two.loc[stage_two["dx_group"] == "PD-MCI"]
df_pdnonmci_first = stage_two.loc[stage_two["dx_group"] == "PD-non-MCI"]
df_hc_first = stage_two.loc[stage_two["dx_group"] == "HC"]

cohort_df_PDnonMCI_HC = pd.concat([df_hc_first, df_pdnonmci_first])

# %%
# PD-non-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDnonMCI_HC[
    ["first_visit", "AGE_AT_VISIT", "SEX", "dx_group"]
]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "first_visit", "dx_group", "AGE_AT_VISIT", "SEX"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDnonMCI_HC_baseline.txt"), sep="\t", index=False, header=None
)

# generate file header
with open(out("fsgd_group_desc_PDnonMCI_HC_baseline.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex \n"
    )


with open(out("fsgd_cort_group_HC_PDnonMCI_baseline.fsgd"), "wb") as outfile:
    for f in [
        "fsgd_group_desc_PDnonMCI_HC_baseline.txt",
        "fsgd_group_cort_PDnonMCI_HC_baseline.txt",
    ]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())

# %% [markdown]
# ## FS 5 - Stack images for group analyses - baseline - t-test

# %%
# %%writefile preprocess_FS5_mris_preproc_group_baseline_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_preproc_group_baseline_ttest.out
#SBATCH --error=logs/FS5_mris_preproc_group_baseline_ttest.err
#SBATCH --time=0:20:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_baseline_ttest.py

# %% [markdown]
# ## FS 5 - Create FSGD file for the group analysis - longitudinal - t-test

# %%
# PD-non-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDnonMCI_HC[
    ["PATNO_base", "AGE_AT_VISIT", "SEX", "dx_group", "durationT2_T1_y"]
]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "PATNO_base", "dx_group", "AGE_AT_VISIT", "SEX", "durationT2_T1_y"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDnonMCI_HC_long.txt"), sep="\t", index=False, header=None
)

# generate file header
with open(out("fsgd_group_desc_PDnonMCI_HC_long.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex durationT2_T1_y\n"
    )


with open(out("fsgd_cort_group_HC_PDnonMCI_long.fsgd"), "wb") as outfile:
    for f in [
        "fsgd_group_desc_PDnonMCI_HC_long.txt",
        "fsgd_group_cort_PDnonMCI_HC_long.txt",
    ]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())

# %% [markdown]
# ## FS 5 - Stack images for group analyses - longitudinal - t-test

# %%
# %%writefile preprocess_FS5_mris_preproc_group_long_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_preproc_group_long_ttest.out
#SBATCH --error=logs/FS5_mris_preproc_group_long_ttest.err
#SBATCH --time=0:20:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_long_ttest.py

# %% [markdown]
# ## Create FSGD file for the correlational analysis - baseline

# %%
cohort_df_PD = pd.concat([df_pdnonmci_first, df_pdmci_first])

fsgd_cortThick_mcat = cohort_df_PD[
    ["first_visit", "AGE_AT_VISIT", "SEX", "NP3TOT", "dx_group"]
]
fsgd_cortThick_mcat["Input"] = "Input"
fsgd_cortThick_mcat = fsgd_cortThick_mcat[
    ["Input", "first_visit", "dx_group", "NP3TOT", "AGE_AT_VISIT", "SEX"]
]


fsgd_cortThick_mcat_PDnonMCI = fsgd_cortThick_mcat.loc[
    fsgd_cortThick_mcat["dx_group"] == "PD-non-MCI"
]
fsgd_cortThick_mcat_PDnonMCI["dx_group"] = "PDnonMCI"

# %%
# generate fsgd files for correlational analysis

for group in ["PDnonMCI"]:

    # generate sample matrix
    exec(
        "fsgd_cortThick_mcat_%s.to_csv(out('fsgd_corr_%s_cort.txt'), sep='\t', index=False, header=None)"
        % (group, group)
    )

    # generate file header
    with open(out("fsgd_corr_{group}_desc.txt".format(group=group)), "w") as f:
        f.write(
            "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName NP3TOT\nClass {group}\nVariables NP3TOT age sex \n".format(
                group=group
            )
        )

    # generate fsgd file
    with open(out("fsgd_corr_{group}_group.fsgd".format(group=group)), "wb") as outfile:
        for f in [
            "fsgd_corr_{group}_desc.txt".format(group=group),
            "fsgd_corr_{group}_cort.txt".format(group=group),
        ]:
            with open(inp(f), "rb") as infile:
                outfile.write(infile.read())

# %% [markdown]
# ## FS 5 - Stack images for correlational analyses - baseline

# %%
# %%writefile preprocess_FS5_mris_preproc_corr.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.10.mgh'.format(hemi=hemi, group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_preproc_corr.out
#SBATCH --error=logs/FS5_mris_preproc_corr.err
#SBATCH --time=0:20:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_corr.py

# %% [markdown]
# ## Create FSGD file for the correlational analysis - longitudinal

# %%
fsgd_cortThick_mcat = stage_two[
    [
        "PATNO_base",
        "durationT2_T1_y",
        "dx_group",
        "AGE_AT_VISIT",
        "SEX",
        "NP3TOT_change",
    ]
]
fsgd_cortThick_mcat["Input"] = "Input"
fsgd_cortThick_mcat = fsgd_cortThick_mcat[
    [
        "Input",
        "PATNO_base",
        "dx_group",
        "NP3TOT_change",
        "AGE_AT_VISIT",
        "SEX",
        "durationT2_T1_y",
    ]
]


fsgd_cortThick_mcat_PDnonMCI = fsgd_cortThick_mcat.loc[
    fsgd_cortThick_mcat["dx_group"] == "PD-non-MCI"
]
fsgd_cortThick_mcat_PDnonMCI["dx_group"] = "PDnonMCI"

# %%
# generate fsgd files for correlational analysis

for group in ["PDnonMCI"]:

    # generate sample matrix
    exec(
        "fsgd_cortThick_mcat_%s.to_csv(out('fsgd_corr_%s_cort_long.txt'), sep='\t', index=False, header=None)"
        % (group, group)
    )

    # generate file header
    with open(out("fsgd_corr_{group}_desc_long.txt".format(group=group)), "w") as f:
        f.write(
            "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName NP3TOT\nClass {group}\nVariables NP3TOT age sex durationT2_T1_y\n".format(
                group=group
            )
        )

    # generate fsgd file
    with open(out("fsgd_corr_{group}_group_long.fsgd".format(group=group)), "wb") as outfile:
        for f in [
            "fsgd_corr_{group}_desc_long.txt".format(group=group),
            "fsgd_corr_{group}_cort_long.txt".format(group=group),
        ]:
            with open(inp(f), "rb") as infile:
                outfile.write(infile.read())

# %% [markdown]
# ## FS5 - Stack images for correlational analyses - longitudinal

# %%
# %%writefile preprocess_FS_mris_preproc_corr_long.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi, group=group),
                                       )



# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS_mris_preproc_corr_long.out
#SBATCH --error=logs/FS_mris_preproc_corr_long.err
#SBATCH --time=0:20:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS_mris_preproc_corr_long.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - baseline - t-test

# %%
# create group contrasts

with open(out("con_group_HC_PDnonMCI_base.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0")

# %%
# %%writefile glm_FS5_group_base_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893796"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_base_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}_base.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_gr_base
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_group_base_ttest.out
#SBATCH --error=logs/FS5_glm_group_base_test.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_base_ttest.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - longitudinal - t-test

# %%
# create group contrasts

with open(out("con_group_HC_PDnonMCI_long.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0")

# %%
# %%writefile glm_FS5_group_long_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893796"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}_long.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_gr_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_group_long_ttest.out
#SBATCH --error=logs/FS5_glm_group_long_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_long_ttest.py

# %% [markdown]
# ## FS 5 - Run GLM model for the correlational analysis - baseline

# %%
# create contrast for UDPRS scores

with open(out("con_corr_UPDRS_base.mtx"), "w") as f:
    f.write("0 1 0 0")

# %%
# %%writefile glm5_corr_base.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893796"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_corr_{group}_{hemi}_base'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.10.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_base.mtx'
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_corr.out
#SBATCH --error=logs/FS5_glm_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_corr_base.py

# %% [markdown]
# ## FS5 - Run GLM model for the correlational analysis - longitudinal

# %%
# create contrast for UPDRS scores

with open(out("con_corr_UPDRS_long.mtx"), "w") as f:
    f.write("0 1 0 0 0")

# %%
# %%writefile glm_corr_long.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893796"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_corr_{group}_{hemi}_long'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_long.mtx'
                                       )



# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_corr_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_corr_long.out
#SBATCH --error=logs/FS5_glm_corr_long.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_corr_long.py

# %% [markdown]
# ## FS 5 - Correction for multiple comparison (with mri_glmfit-sim) - t-test

# %%
# %%writefile glm_FS5_group_sim_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893807"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        for time in ["baseline", "long"]:
            out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                                '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        dir='results_group_{time}_{group}_{hemi}'.format(time=time,hemi=hemi,group=group),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_group_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_sim_group_ttest.out
#SBATCH --error=logs/FS5_glm_sim_group_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_sim_ttest.py

# %% [markdown]
# ## FS 5 - Correction for multiple comparison (with mri_glmfit-sim) - correlation

# %%
# %%writefile glm5_corr_sim.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893807"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        for ses in ["long", "base"]:
            out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        dir='results_corr_{group}_{hemi}_{ses}'.format(group=group, hemi=hemi, ses=ses),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_corr_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_sim_corr.out
#SBATCH --error=logs/FS5_glm_sim_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_corr_sim.py

# %% [markdown]
# # Freesurfer 6

# %% [markdown]
# ## Freesurfer 6 - preprocessing step 2 - base template

# %%
# %%writefile preprocess_FS6_base.py

# Step 2. create an unbiased template from all time points for each subject and process it with recon-all:

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900700"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_base.json') as fin:
    subject_map = json.load(fin)

    
    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                    license="FS_license/license.txt",
                                        tp1=subject_map["first_visit"][task_id],
                                        tp2=subject_map["second_visit"][task_id],
                                        outputdir=subject_map["PATNO_base"][task_id],
                                       )
print(out_fs)

# %%
# %%sbatch --array=0-249
# #!/bin/bash
#SBATCH --job-name=FS6_base
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_base.%a.out
#SBATCH --error=logs/FS6_preproc_base.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_base.py

# %% [markdown]
# ## Freesurfer 6 - preprocessing step 3 - longitudinally processed timepoints

# %%
# %%writefile preprocess_FS6_long.py

# Step 3. "-long" longitudinally process all timepoints (recon-all -long):

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900706"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                    license="FS_license/license.txt",
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id]
                                       )


print(out_fs)

# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS6_long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_long.%a.out
#SBATCH --error=logs/FS6_preproc_long.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_long.py

# %% [markdown]
# ## Freesurfer 6 - preprocessing step 4 - Qcache

# %%
# %%writefile preprocess_FS6_long_qcache.py

# save proprocessing script to submit jobs to the server later 

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900708"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                    license="FS_license/license.txt",
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id],
                                       )


# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS6_longQcache
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_longQcache.%a.out
#SBATCH --error=logs/FS6_preproc_longQcache.%a.err
#SBATCH --time=2:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_long_qcache.py

# %% [markdown]
# ## FS 6 - Prepare the vertex data with long_mris_slopes for longitudinal two stage model

# %%
# %%writefile preprocess_FS6_long_mris_slopes.py

# Prepare the data with long_mris_slopes for longitudinal two stage model

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900742"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
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


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_slopes
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_slopes.out
#SBATCH --error=logs/FS6_mris_slopes.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_long_mris_slopes.py

# %% [markdown]
# ## FS 6 - Stack images for group analyses - baseline - t-test

# %%
# %%writefile preprocess_FS6_mris_preproc_group_baseline_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_group_baseline_ttest.out
#SBATCH --error=logs/FS6_mris_preproc_group_baseline_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_group_baseline_ttest.py

# %% [markdown]
# ## FS 6 - Stack images for group analyses - longitudinal - t-test

# %%
# %%writefile preprocess_FS6_mris_preproc_group_long_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_group_long_ttest.out
#SBATCH --error=logs/FS6_mris_preproc_group_long_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_group_long_ttest.py

# %% [markdown]
# ## FS 6 - Stack images for correlational analyses - baseline

# %%
# %%writefile preprocess_FS6_mris_preproc_corr.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi, group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_corr.out
#SBATCH --error=logs/FS6_mris_preproc_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_corr.py

# %% [markdown]
# ## FS 6 - Stack images for correlational analyses - longitudinal

# %%
# %%writefile preprocess_FS6_mris_preproc_corr_long.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi, group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_corr_long.out
#SBATCH --error=logs/FS6_mris_preproc_corr_long.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_corr_long.py

# %% [markdown]
# ## FS 6 - Run GLM model for the group analysis - baseline - t-test

# %%
# %%writefile glm_FS6_group_base_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900725"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_base_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}_base.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_gr_base
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_group_base_ttest.out
#SBATCH --error=logs/FS6_glm_group_base_test.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS6_group_base_ttest.py

# %% [markdown]
# ## FS 6 - Run GLM model for the group analysis - longitudinal - t-test

# %%
# %%writefile glm_FS6_group_long_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900725"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}_long.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_gr_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_group_long_ttest.out
#SBATCH --error=logs/FS6_glm_group_long_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS6_group_long_ttest.py

# %% [markdown]
# ## FS 6 - Run GLM model for the correlational analysis - baseline

# %%
# %%writefile glm6_corr.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900725"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_base_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_base.mtx'
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_corr.out
#SBATCH --error=logs/FS6_glm_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_corr.py

# %% [markdown]
# ## FS 6 - Run GLM model for the correlational analysis - longitudinal

# %%
# %%writefile glm6_corr_long.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900725"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_long_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_long.mtx'
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_corr_long.out
#SBATCH --error=logs/FS6_glm_corr_long.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_corr_long.py

# %% [markdown]
# ## FS 6 - Correction for multiple comparison (with mri_glmfit-sim) - t-test

# %%
# %%writefile glm_FS6_group_sim_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900735"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                                license="FS_license/license.txt",
                                        dir='results_group_{time}_{group}_{hemi}'.format(time=time,hemi=hemi,group=group),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_group_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_sim_group_ttest.out
#SBATCH --error=logs/FS6_glm_sim_group_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS6_group_sim_ttest.py

# %% [markdown]
# ## FS 6 - Correction for multiple comparison (with mri_glmfit-sim) - correlation

# %%
# %%writefile glm6_corr_sim.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900735"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        dir='results_corr_{time}_{group}_{hemi}'.format(time=time, group=group, hemi=hemi),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_corr_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_sim_corr.out
#SBATCH --error=logs/FS6_glm_sim_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_corr_sim.py

# %% [markdown]
# # Freesurfer 7.3.2

# %% [markdown]
# ## Freesurfer 7 - preprocessing step 2 - base template

# %%
# %%writefile preprocess_FS7_base.py

# Step 2. create an unbiased template from all time points for each subject and process it with recon-all:

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7916240"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_base.json') as fin:
    subject_map = json.load(fin)

out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                        tp1=subject_map["first_visit"][task_id],
                                        tp2=subject_map["second_visit"][task_id],
                                        license="FS_license/license.txt",
                                        outputdir=subject_map["PATNO_base"][task_id],
                                       )

# %%
# %%sbatch --array=0-249
# #!/bin/bash
#SBATCH --job-name=FS7_base
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_preproc_base.%a.out
#SBATCH --error=logs/FS7_preproc_base.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_base.py

# %% [markdown]
# ## Freesurfer 7 - preprocessing step 3 - longitudinally processed timepoints

# %%
# %%writefile preprocess_FS7_long.py

# Step 3. "-long" longitudinally process all timepoints (recon-all -long):

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920788"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id],
                                        license="FS_license/license.txt",
                                       )


# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS7_long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_preproc_long.%a.out
#SBATCH --error=logs/FS7_preproc_long.%a.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_long.py

# %% [markdown]
# ## Freesurfer 7 - preprocessing step 4 - Qcache

# %%
# %%writefile preprocess_FS7_long_qcache.py

# save proprocessing script to submit jobs to the server later 

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920876"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

task_id = str(os.environ["SLURM_ARRAY_TASK_ID"])

with open('json_data_long.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                        tp=subject_map["visit"][task_id],
                                        base=subject_map["PATNO_base"][task_id],
                                        license="FS_license/license.txt",
                                       )

# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS7_preproc_longQcache
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_preproc_longQcache.%a.out
#SBATCH --error=logs/FS7_preproc_longQcache.%a.err
#SBATCH --time=2:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_long_qcache.py

# %% [markdown]
# ## FS 7 - Prepare the vertex data with long_mris_slopes for longitudinal two stage model

# %%
# %%writefile preprocess_FS7_long_mris_slopes.py

# Prepare the data with long_mris_slopes for longitudinal two stage model

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920880"
from boutiques.descriptor2func import function
#bosh(["exec", "prepare", zid])
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
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


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS7_mris_slopes
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_mris_slopes.out
#SBATCH --error=logs/FS7_mris_slopes.err
#SBATCH --time=10:0:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_long_mris_slopes.py

# %% [markdown]
# ## FS 7 - Stack images for group analyses - baseline - t-test

# %%
# %%writefile preprocess_FS7_mris_preproc_group_baseline_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920888"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS7_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_mris_preproc_group_baseline_ttest.out
#SBATCH --error=logs/FS7_mris_preproc_group_baseline_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_group_baseline_ttest.py

# %% [markdown]
# ## FS 7 - Stack images for group analyses - longitudinal - t-test

# %%
# %%writefile preprocess_FS7_mris_preproc_group_long_ttest.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920888"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS7_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_mris_preproc_group_long_ttest.out
#SBATCH --error=logs/FS7_mris_preproc_group_long_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_group_long_ttest.py

# %% [markdown]
# ## FS 7 - Stack images for correlational analyses - baseline

# %%
# %%writefile preprocess_FS7_mris_preproc_corr.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920888"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi, group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS7_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_mris_preproc_corr.out
#SBATCH --error=logs/FS7_mris_preproc_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_corr.py

# %% [markdown]
# ## FS 7 - Stack images for correlational analyses - longitudinal

# %%
# %%writefile preprocess_FS7_mris_preproc_corr_long.py

# Concatenate images from correlation analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920888"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi, group=group),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS7_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_mris_preproc_corr_long.out
#SBATCH --error=logs/FS7_mris_preproc_corr_long.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_corr_long.py

# %% [markdown]
# ## FS 7 - Run GLM model for the group analysis - baseline - t-test

# %%
# %%writefile glm_FS7_group_base_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920892"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_base_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}_base.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_gr_base
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_group_base_ttest.out
#SBATCH --error=logs/FS7_glm_group_base_test.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_base_ttest.py

# %% [markdown]
# ## FS 7 - Run GLM model for the group analysis - longitudinal - t-test

# %%
# %%writefile glm_FS7_group_long_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920892"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}_long.mtx'.format(group=group) 
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_gr_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_group_long_ttest.out
#SBATCH --error=logs/FS7_glm_group_long_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_long_ttest.py

# %% [markdown]
# ## FS 7 - Run GLM model for the correlational analysis - baseline

# %%
# %%writefile glm7_corr.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920892"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_base_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_base.mtx'
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_corr.out
#SBATCH --error=logs/FS7_glm_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr.py

# %% [markdown]
# ## FS 7 - Run GLM model for the correlational analysis - longitudinal

# %%
# %%writefile glm7_corr_long.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920892"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:   
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_long_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_long.mtx'
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_corr.out
#SBATCH --error=logs/FS7_glm_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr_long.py

# %% [markdown]
# ## FS 7 - Correction for multiple comparison (with mri_glmfit-sim) - t-test

# %%
# %%writefile glm_FS7_group_sim_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920896"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                license="FS_license/license.txt",
                                        dir='results_group_{time}_{group}_{hemi}'.format(time=time,hemi=hemi,group=group),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_group_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_sim_group_ttest.out
#SBATCH --error=logs/FS7_glm_sim_group_ttest.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_sim_ttest.py

# %% [markdown]
# ## FS 7 - Correction for multiple comparison (with mri_glmfit-sim) - correlation

# %%
# %%writefile glm7_corr_sim.py

# GLM model for the correlational analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7920896"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for group in ["PDnonMCI"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                            license="FS_license/license.txt",
                                        dir='results_corr_{time}_{group}_{hemi}'.format(time=time, group=group, hemi=hemi),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm7_corr_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_glm_sim_corr.out
#SBATCH --error=logs/FS7_glm_sim_corr.err
#SBATCH --time=0:10:0

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr_sim.py

# %% [markdown]
# # Freesurfer 8

# %% [markdown]
# ## Freesurfer 8 - preprocessing step 2 - base template

# %%
# %load_ext slurm_magic

# %%
# %%writefile preprocess_FS8_base.py

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
subjects_dir = "$PROJECT_DIR/data/FS8"
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

# %%
# %%sbatch --array=64
# #!/bin/bash
#SBATCH --job-name=FS8_base
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/step2/FS8_preproc_base.%a.out
#SBATCH --error=logs/step2/FS8_preproc_base.%a.err
#SBATCH --time=4:0:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_base.py

# %% [markdown]
# ## Freesurfer 8 - preprocessing step 3 - longitudinally processed timepoints

# %%
# %%writefile preprocess_FS8_long.py

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
subjects_dir = "$PROJECT_DIR/data/FS8"
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

# %%
# %%sbatch --array=0-495
# #!/bin/bash
#SBATCH --job-name=FS8_long
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/step3/FS8_preproc_long.%a.out
#SBATCH --error=logs/step3/FS8_preproc_long.%a.err
#SBATCH --time=4:0:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_long.py

# %% [markdown]
# ## Freesurfer 8 - preprocessing step 4 - Qcache

# %%
# %%writefile preprocess_FS8_long_qcache.py

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

# %%
# %%sbatch --array=1-495
# #!/bin/bash
#SBATCH --job-name=FS8_longQcache
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/step4/FS8_preproc_longQcache.%a.out
#SBATCH --error=logs/step4/FS8_preproc_longQcache.%a.err
#SBATCH --time=4:0:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_long_qcache.py

# %% [markdown]
# ## FS 8 - Prepare the vertex data with long_mris_slopes for longitudinal two stage model

# %%
# %%writefile preprocess_FS8_long_mris_slopes.py

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


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS8_mris_slopes
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_mris_slopes.out
#SBATCH --error=logs/FS8_mris_slopes.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_long_mris_slopes.py

# %% [markdown]
# ## FS 8 - Stack images for group analyses - baseline - t-test

# %%
# %%writefile preprocess_FS8_mris_preproc_group_baseline_ttest.py

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
                                        cachein="thickness.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                       )
        print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS8_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_mris_preproc_group_baseline_ttest.out
#SBATCH --error=logs/FS8_mris_preproc_group_baseline_ttest.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_mris_preproc_group_baseline_ttest.py

# %% [markdown]
# ## FS 8 - Stack images for group analyses - longitudinal - t-test

# %%
# %%writefile preprocess_FS8_mris_preproc_group_long_ttest.py

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

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS8_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_mris_preproc_group_long_ttest.out
#SBATCH --error=logs/FS8_mris_preproc_group_long_ttest.err
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_mris_preproc_group_long_ttest.py

# %% [markdown]
# ## FS 8 - Stack images for correlational analyses - baseline

# %%
# %%writefile preprocess_FS8_mris_preproc_corr.py

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

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS8_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_mris_preproc_corr.out
#SBATCH --error=logs/FS8_mris_preproc_corr.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_mris_preproc_corr.py

# %% [markdown]
# ## FS 8 - Stack images for correlational analyses - longitudinal

# %%
# %%writefile preprocess_FS8_mris_preproc_corr_long.py

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
                                        cachein="long.thickness-rate.fwhm10.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.10.mgh'.format(hemi=hemi, group=group),
                                       )
        print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS8_mris_preproc_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_mris_preproc_corr_long.out
#SBATCH --error=logs/FS8_mris_preproc_corr_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python preprocess_FS8_mris_preproc_corr_long.py

# %% [markdown]
# ## FS 8 - Run GLM model for the group analysis - baseline - t-test

# %%
# create group contrasts

with open(out("con_group_HC_PDnonMCI_base.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0")

# %%
# %%writefile glm_FS8_group_base_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920892"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mri_glmfit_1con.json")

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_base_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}_base.mtx'.format(group=group) 
                                       )
        print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_gr_base
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_group_base_ttest.out
#SBATCH --error=logs/FS8_glm_group_base_test.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm_FS8_group_base_ttest.py

# %% [markdown]
# ## FS 8 - Run GLM model for the group analysis - longitudinal - t-test

# %%
# create group contrasts

with open(out("con_group_HC_PDnonMCI_long.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0")

# %%
# %%writefile glm_FS8_group_long_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920892"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mri_glmfit_1con.json")

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.10.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}_long.mtx'.format(group=group) 
                                       )
        print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_group_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_group_long_ttest.out
#SBATCH --error=logs/FS8_glm_group_long_ttest.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm_FS8_group_long_ttest.py

# %% [markdown]
# ## FS 8 - Run GLM model for the correlational analysis - baseline

# %%
# create contrast for UDPRS scores

with open(out("con_corr_UPDRS_base.mtx"), "w") as f:
    f.write("0 1 0 0")

# %%
# %%writefile glm8_corr.py

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
                                        outdir='results_corr_base_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.10.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_UPDRS_base.mtx'
                                       )
        print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_corr
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_corr.out
#SBATCH --error=logs/FS8_glm_corr.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm8_corr.py

# %% [markdown]
# ## FS 8 - Run GLM model for the correlational analysis - longitudinal

# %%
# create contrast for UPDRS scores

with open(out("con_corr_UPDRS_long.mtx"), "w") as f:
    f.write("0 1 0 0 0")

# %%
# %%writefile glm8_corr_long.py

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

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_corr_long
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_corr_long.out
#SBATCH --error=logs/FS8_glm_corr_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm8_corr_long.py

# %% [markdown]
# ## FS 8 - Correction for multiple comparison (with mri_glmfit-sim) - t-test

# %%
# %%writefile glm_FS8_group_sim_ttest.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function
# zid = "zenodo.7920896"
# freesurfer = function(zid)
freesurfer = function("./boutiques/freesurfer_8_mri_glmfit_sim_abs.json")

for group in ["HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            out_fs = freesurfer('--imagepath', 'freesurfer_8.2.0.sif',
                                license="FS_license/license.txt",
                                        dir='results_group_{time}_{group}_{hemi}'.format(time=time,hemi=hemi,group=group),
                                        CACHE_abs='1.3',
                                        cwp="0.05"
                                       )
            print(out_fs)

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_group_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_sim_group_ttest.out
#SBATCH --error=logs/FS8_glm_sim_group_ttest.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm_FS8_group_sim_ttest.py

# %% [markdown]
# ## FS 8 - Correction for multiple comparison (with mri_glmfit-sim) - correlation

# %%
# %%writefile glm8_corr_sim.py

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

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm8_corr_sim
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS8_glm_sim_corr.out
#SBATCH --error=logs/FS8_glm_sim_corr.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

source .venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load apptainer
export SUBJECTS_DIR="$PROJECT_DIR/data/FS8"
python glm8_corr_sim.py

# %% [markdown]
# ## Whole-brain Sørensen–Dice coefficients

# %%
# %%writefile whole-brain.sh
# #!/bin/bash
#SBATCH --job-name=dice_overlap
#SBATCH --output=logs/dice_%x_%j.out
#SBATCH --error=logs/dice_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=<account>

# Usage:
# sbatch run_mri_seg_overlap.sbatch FS8 FS7

module load apptainer
sif_path=$PROJECT_DIR/freesurfer_8.2.0.sif
WORKDIR=$(pwd)

if [ $# -ne 2 ]; then
    echo "Usage: sbatch $0 <version1> <version2>"
    exit 1
fi

version1=$1
version2=$2
tag=$(echo "${version1}${version2}" | sed 's/FS//g')

# echo "Comparing $version1 vs $version2"

# mkdir -p stats/dice
# mkdir -p logs

mapfile -t subj < orig.txt


for sub in "${subj[@]}";
do

    echo "Processing $sub: $version1 vs $version2"

    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aparc.a2009s+aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aparc.a2009s+aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aparc2009.json


    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aparc+aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aparc+aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aparc.json


    apptainer exec \
        --cleanenv \
        -B ${WORKDIR}:${WORKDIR} \
        -B ${WORKDIR}/FS_license/license.txt:/license.txt \
         --env FS_LICENSE=/license.txt \
        ${sif_path} \
        mri_seg_overlap \
        ${WORKDIR}/data/${version1}/${sub}/mri/aseg.mgz \
        ${WORKDIR}/data/${version2}/${sub}/mri/aseg.mgz \
        -o ${WORKDIR}/stats/dice/dice_${tag}_${sub}.aseg.json


done

# %%
# %%writefile whole-brain-all.sh

# #!/bin/bash

versions=("FS5" "FS6" "FS7" "FS8")


# Submit unique pairs only

for ((i=0; i<${#versions[@]}; i++))
do
    for ((j=i+1; j<${#versions[@]}; j++))
    do
        v1=${versions[$j]}
        v2=${versions[$i]}

        echo "Submitting $v1 vs $v2"

        sbatch whole-brain.sh $v1 $v2

    done
done

# %%
import json

for subj in aseg_table["PATNO_id"]:
    for pair in ["76", "75", "65", "87", "86", "85"]:
        for file in ["aseg", "aparc", "aparc2009"]:
            with open(
                str(DATA) + "/stats/dice/dice_{pair}_{subj}.{file}.json".format(
                    pair=pair, subj=subj, file=file
                ),
                "r",
            ) as f:
                data = json.load(f)

                for key in data["measures"]["dice"]["labels"]:

                    name = data["names"]["{key}".format(key=key)]
                    dice = data["measures"]["dice"]["labels"]["{key}".format(key=key)]
                    aseg_table.loc[
                        aseg_table["PATNO_id"] == subj,
                        "dice_FS{pair}_".format(pair=pair) + name,
                    ] = dice

# %%
dice_table = aseg_table.filter(like="dice_")

mean = dice_table.mean()
sd = dice_table.std(numeric_only=True)

dice_results = pd.DataFrame({
    "metric": mean.index,
    "mean": mean.values,
    "sd": sd.values
})
aseg_table.to_csv(out("aseg_table4.csv"), index=False)
dice_results.to_csv(out("dice_results.csv"), index=False)

# %%
"""
Visualize dice_results.csv (mean/sd of Dice coefficients comparing
FreeSurfer versions, e.g. FS5/6/7/8) per brain structure.

Expects metric names like: dice_FS76_Left-Thalamus
  -> comparison = "FS76" (e.g. FS7 vs FS6)
  -> structure  = "Left-Thalamus"

Adjust the regex in parse_metric() if your naming convention differs
(e.g. dice_FS7-FS6_Left-Thalamus, dice_LeftThalamus_FS76, etc.)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- 1. Load and parse ----------------------------------------------------

df = pd.read_csv(inp("dice_results.csv"))  # columns: metric, mean, sd

def parse_metric(metric):
    m = re.match(r"dice_(FS\d+)_(.+)", metric)
    if not m:
        return pd.Series({"comparison": None, "structure": None})
    return pd.Series({"comparison": m.group(1), "structure": m.group(2)})

df[["comparison", "structure"]] = df["metric"].apply(parse_metric)
df = df.dropna(subset=["comparison", "structure"])

comparisons = sorted(df["comparison"].unique())
structures = sorted(df["structure"].unique())

# ---- 2. Grouped bar chart with error bars ---------------------------------

def grouped_bar_chart(df, save_path=out("dice_grouped_bar2.png")):
    structures = sorted(df["structure"].unique())
    comparisons = sorted(df["comparison"].unique())
    n_struct = len(structures)
    n_comp = len(comparisons)
    width = 0.8 / n_comp

    fig, ax = plt.subplots(figsize=(max(8, n_struct * 0.5), 6))
    x = np.arange(n_struct)

    for i, comp in enumerate(comparisons):
        sub = df[df["comparison"] == comp].set_index("structure").reindex(structures)
        ax.bar(
            x + i * width - (n_comp - 1) * width / 2,
            sub["mean"],
            width,
            yerr=sub["sd"],
            capsize=3,
            label=comp,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=90)
    ax.set_ylabel("Dice coefficient")
    ax.set_title("Dice coefficient (mean ± SD) by structure and FS version comparison")
    ax.legend(title="Comparison")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")

# ---- 3. Heatmap -------------------------------------------------------------

def heatmap(df, save_path=out("dice_heatmap2.png")):
    pivot = df.pivot(index="structure", columns="comparison", values="mean")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=ax, label="Mean Dice")
    ax.set_title("Mean Dice coefficient: structure x FS version comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    grouped_bar_chart(df)
    heatmap(df)

# %% [markdown]
# # Quality control of preprocessed images

# %%
import imageio as iio
from pathlib import Path

for stage in ["clinical", "comput"]:
    for ver in ["FS5", "FS6", "FS7"]:
        for view in ["axial", "sagittal", "coronal"]:
            images = list()
            for file in Path(f"{DATA}/segm/{stage}/{ver}/{view}").iterdir():
                if not file.is_file():
                    continue

                images.append(iio.imread(file))
                iio.mimsave(out(f"segm/{stage}/{ver}/segm_{stage}_{view}.gif"), images, duration=1
                )

# %% [markdown]
# ## Computational data
#
# ## FS5
#
# ### FS 5 axial view
# ![SegmentLocal](images/segm_comput_FS5_axial.gif "segment")
#
# ### FS 5 coronal view
# ![SegmentLocal](images/segm_comput_FS5_coronal.gif "segment")
#
# ### FS 5 sagittal view
# ![SegmentLocal](images/segm_comput_FS5_sagittal.gif "segment")
#
# ## FS6
#
# ### FS 6 axial view
# ![SegmentLocal](images/segm_comput_FS6_axial.gif "segment")
#
# ### FS 6 coronal view
# ![SegmentLocal](images/segm_comput_FS6_coronal.gif "segment")
#
# ### FS 6 sagittal view
# ![SegmentLocal](images/segm_comput_FS6_sagittal.gif "segment")
#
# ## FS7
#
# ### FS 7 axial view
# ![SegmentLocal](images/segm_comput_FS7_axial.gif "segment")
#
# ### FS 7 coronal view
# ![SegmentLocal](images/segm_comput_FS7_coronal.gif "segment")
#
# ### FS 7 sagittal view
# ![SegmentLocal](images/segm_comput_FS7_sagittal.gif "segment")
#
#
# ## Clinical data
#
# ## FS5
#
# ### FS 5 axial view
# ![SegmentLocal](images/segm_clinical_FS5_axial.gif "segment")
#
# ### FS 5 coronal view
# ![SegmentLocal](images/segm_clinical_FS5_coronal.gif "segment")
#
# ### FS 5 sagittal view
# ![SegmentLocal](images/segm_clinical_FS5_sagittal.gif "segment")
#
# ## FS6
#
# ### FS 6 axial view
# ![SegmentLocal](images/segm_clinical_FS6_axial.gif "segment")
#
# ### FS 6 coronal view
# ![SegmentLocal](images/segm_clinical_FS6_coronal.gif "segment")
#
# ### FS 6 sagittal view
# ![SegmentLocal](images/segm_clinical_FS6_sagittal.gif "segment")
#
# ## FS7
#
# ### FS 7 axial view
# ![SegmentLocal](images/segm_clinical_FS7_axial.gif "segment")
#
# ### FS 7 coronal view
# ![SegmentLocal](images/segm_clinical_FS7_coronal.gif "segment")
#
# ### FS 7 sagittal view
# ![SegmentLocal](images/segm_clinical_FS7_sagittal.gif "segment")
