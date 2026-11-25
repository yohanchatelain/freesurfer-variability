# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Freesurfer variability in Parkinson's disease neuroimaging data
#
# ## Introduction
#
# This notebook examine software variability using neuroimaging data from patients with Parkinson's disease.
#
# T1-weighted brain images are taken from replication studies on four papers:
#
# A. Hanganu et al., <a href=https://doi.org/10.1093/brain/awu036> “Mild cognitive impairment is linked with faster rate of cortical thinning in patients with Parkinson’s disease longitudinally,”</a> Brain, vol. 137, no. 4, pp. 1120–1129, 2014.<br>
# Replication notebook with cohort: https://github.com/LivingPark-MRI/hanganu-etal-2014
#
# E. Mak et al.,  <a href=https://doi.org/10.1093/brain/awv211> “Baseline and longitudinal grey matter changes in newly diagnosed Parkinson’s disease: ICICLE-PD study,” </a> Brain, vol. 138, no. 10, pp. 2974–2986, 2015.<br>
# Replication notebook with cohort: https://github.com/LivingPark-MRI/mak-etal-2015
#
# <br> <br>
# Data processed with three FreeSurfer versions will be compared (5.3, 6.0.1, and 7.3.2).
#
# In the first step data from all participants will be preprocessed in the cross-sectional stream. Variability in the cortical thickness and subcortical structures volumes will be tested.
#
# In the second step data will be analyzed separately for each paper to test whether the clinical results would vary between different software versions.

# %% [markdown]
# ## Initial setup
#
# Let's initialize the notebook directory and software dependencies:

# %%
import os
import shutil
import livingpark_utils
from livingpark_utils import download

utils = livingpark_utils.LivingParkUtils()
random_seed = 2
utils.notebook_init()

downloader = download.ppmi.Downloader(utils.study_files_dir, headless=False)

import warnings

warnings.filterwarnings("ignore")

# %%
import pandas as pd

# %% [markdown]
# ## Cohort preparation
#
# We will build a cohort from the four studies described in the beginning of the notebook. Participant data are imported from the other notebooks.

# %%
df_Mak = pd.read_csv('cohort_Mak_2015.csv') 
df_Hanganu = pd.read_csv('cohort_Hanganu.csv')
df_Rahayel = pd.read_csv('cohort_Rahayel.csv')

df_Mak_download = df_Mak[['PATNO', 'Description', 'EVENT_ID']]
df_Hanganu_download = df_Hanganu[['PATNO', 'Description', 'EVENT_ID']]
df_Rahayel_download = df_Rahayel[['PATNO', 'Description', 'EVENT_ID']]

download_data = pd.concat([df_Mak_download, df_Hanganu_download, df_Rahayel_download], ignore_index=True)
download_data = download_data.drop_duplicates(subset=['PATNO', 'EVENT_ID'])
download_data["PATNO_id"] = "sub-" + download_data["PATNO"].astype(str) + "_ses-" + download_data["EVENT_ID"]
download_data


# %%
download_data.to_csv("download_data.csv", index=False)

# %% [markdown]
# ## Download data

# %%
# download MRI data

utils.download_missing_nifti_files(download_data, link_in_outputs=True)

# %% [markdown]
# ## Preprocessing
#
# Preprocessing is performed on an external server. <a href=https://github.com/boutiques> Boutiques</a> is used to manage Freesurfer's functions within a container. 

# %%
# %load_ext slurm_magic

# %%
# save df with all timepoints as json

import json

download_data["PATNO_id"] = download_data["PATNO_id"].astype(str)

data_to_process = download_data.reset_index()
small_df = data_to_process[["PATNO_id", "File name"]]
json_data = small_df.to_json()
meta = json.loads(json_data)
with open("json_data.json", "w") as fout:
    json.dump(meta, fout, indent=4)

# %% [markdown]
# ## preprocessing step 1 - cross-sectional
# ### Freesurfer 5.3

# %%
# prepare node for computation. it includes downloading the container

# ! module load singularity

from boutiques import bosh

zid = "zenodo.7579820"
bosh(["exec", "prepare", zid])
freesurfer = boutiques.descriptor2func.function(zid)

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

with open('json_data.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{os.getcwd()}/FS_license/license.txt:/usr/local/freesurfer/license.txt',
                    '--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_5.3.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-285
# #!/bin/bash
#SBATCH --job-name=FS5_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS5_preproc.out
#SBATCH --error=FS5_preproc.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS5.py

# %%
# %%writefile preprocess_FS5_err2.py

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

with open('json_data_err5_2.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{os.getcwd()}/FS_license/license.txt:/usr/local/freesurfer/license.txt',
                    '--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_5.3.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-1
# #!/bin/bash
#SBATCH --job-name=FS5
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS5_preproc_err2.out
#SBATCH --error=FS5_preproc_err2.err
#SBATCH --time=168:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS5_err2.py

# %%
# %%writefile preprocess_FS5_mak2nd.py

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

with open('json_data_mak2nd.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{os.getcwd()}/FS_license/license.txt:/usr/local/freesurfer/license.txt',
                    '--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_5.3.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-1
# #!/bin/bash
#SBATCH --job-name=FS5
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS5_preproc_mak2nd.out
#SBATCH --error=FS5_preproc_mak2nd.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS5_mak2nd.py

# %%
# %%writefile preprocess_FS5_mak_err.py

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

with open('json_data_err5_mak.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('-v', f'{os.getcwd()}/FS_license/license.txt:/usr/local/freesurfer/license.txt',
                    '--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_5.3.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-4
# #!/bin/bash
#SBATCH --job-name=FS5_err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS5_preproc_mak_err.out
#SBATCH --error=FS5_preproc_mak_err.err
#SBATCH --time=168:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS5_mak_err.py

# %% [markdown]
# ### FreeSurfer 6.0.1

# %%
# prepare node for computation. it includes downloading the container

# ! module load singularity

from boutiques import bosh

zid = "zenodo.7542266"
bosh(["exec", "prepare", zid])
freesurfer = boutiques.descriptor2func.function(zid)

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

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_6.0.1.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-191
# #!/bin/bash
#SBATCH --job-name=FS6_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS6_preproc.out
#SBATCH --error=FS6_preproc.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS6.py

# %%
# %%writefile preprocess_FS6_err3.py

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

with open('json_data_err6_3.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_6.0.1.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-1
# #!/bin/bash
#SBATCH --job-name=FS6
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS6_preproc_err3.out
#SBATCH --error=FS6_preproc_err3.err
#SBATCH --time=168:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS6_err3.py

# %%
# %%writefile preprocess_FS6_mak2nd.py

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

with open('json_data_mak2nd.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_6.0.1.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-111 
# #!/bin/bash
#SBATCH --job-name=FS6_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS6_preproc_mak2nd.out
#SBATCH --error=FS6_preproc_mak2nd.err
#SBATCH --time=12:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS6_mak2nd.py

# %% [markdown]
# ### FreeSurfer 7.3.2

# %%
# prepare node for computation. it includes downloading the container

# ! module load singularity

from boutiques import bosh

zid = "zenodo.7545769"
bosh(["exec", "prepare", zid])
freesurfer = boutiques.descriptor2func.function(zid)

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

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_7.3.2.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-191
# #!/bin/bash
#SBATCH --job-name=FS7_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS7_preproc.out
#SBATCH --error=FS7_preproc.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7.py

# %%
# %%writefile preprocess_FS7_err.py

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

with open('json_data_err.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_7.3.2.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-1
# #!/bin/bash
#SBATCH --job-name=FS7_preproc
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS7_preproc_err.out
#SBATCH --error=FS7_preproc_err.err
#SBATCH --time=20:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_err.py

# %%
# %%writefile preprocess_FS7_mak2nd.py

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

with open('json_data_mak2nd.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', '$PROJECT_DIR/freesurfer/ansokol-freesurfer_7.3.2.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-111
# #!/bin/bash
#SBATCH --job-name=FS7_mak
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=FS7_preproc_mak2nd.out
#SBATCH --error=FS7_preproc_mak2nd.err
#SBATCH --time=12:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_mak2nd.py

# %%

# %%

# %%

# %%

# %%
# # copy preprocessed FS7 data

os.makedirs('data_FS7', exist_ok=True)

for pat in download_data["PATNO_id"]:
    shutil.copytree('{pat}'.format(pat = pat), 'data_FS7/{pat}'.format(pat = pat))

# %%
