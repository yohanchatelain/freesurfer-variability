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

# %%
# ---- Paths (revision workspace) ----
# Inputs are read from ../data (symlinks to $PROJECT_DIR, see ../data/README.md).
# Everything this notebook writes goes to ../outputs. Never write into ../data: the links point to
# the original files, which are writable.
from pathlib import Path

DATA = Path("../data").resolve()
OUT = Path("../outputs").resolve()
OUT.mkdir(exist_ok=True)
INPUT_SUBDIR = "glm_FS5-7/FS7"  # default ../data subdirectory for files written by the original pipeline


def out(name):
    """Path of a file written by this notebook (always under ../outputs)."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def inp(name, sub=None):
    """Path of a file read by this notebook: regenerated copy in ../outputs if any, else the original in ../data."""
    p = OUT / name
    return str(p if p.exists() else DATA / (sub or INPUT_SUBDIR) / name)


# Multiple comparisons (Bonferroni): raw p-values are compared to ALPHA / n_tests.
# p-values are never multiplied (this produced p > 1 in the submitted tables).
ALPHA = 0.05
N_SUBCORTICAL = 14  # bilateral subcortical ROIs tested in the clinical analyses (Tables 2-3)


def fmt_p(p, n_tests=1, alpha=ALPHA):
    """Raw p-value for tables; '*' marks significance after Bonferroni (p < alpha / n_tests)."""
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    return s + ("*" if p < alpha / n_tests else "")



# %% [markdown]
# > **Revision workspace note.** The `%%writefile` / `%%sbatch` cells in the processing sections are the original pipeline. They were run in `$PROJECT_DIR` and use paths relative to that directory. **Don't run them from this workspace**: `preprocess_FS8*.py` deletes existing subject folders (`shutil.rmtree`), and `data/FS8` links to the original outputs. Re-runs go through `../scripts/` with outputs on `$SCRATCH`.

# %%
import livingpark_utils
import pandas as pd
import numpy as np

utils = livingpark_utils.LivingParkUtils()
random_seed = 2
utils.notebook_init()

import warnings

warnings.filterwarnings("ignore")

# %%
import livingpark_utils
from livingpark_utils.download import ppmi

utils = livingpark_utils.LivingParkUtils()
downloader = ppmi.Downloader(utils.study_files_dir)

required_files = [
    "Demographics.csv",
    "Age_at_visit.csv",
    "Primary_Clinical_Diagnosis.csv",
    "Cognitive_Categorization.csv",
    "Medical_Conditions_Log.csv",
    "Concomitant_Medication_Log.csv",
    "MDS-UPDRS_Part_III.csv",
    "Participant_Status.csv",
    "Socio-Economics.csv",
    "Montreal_Cognitive_Assessment__MoCA_.csv",
    "PD_Diagnosis_History.csv",
    "LEDD_Concomitant_Medication_Log.csv",
]

utils.notebook_init()
utils.get_study_files(required_files, default=downloader)

# %%
from livingpark_utils.scripts import run
from livingpark_utils.scripts import mri_metadata

run.mri_metadata()

# %%
import os
import pandas as pd

# Read data files


# Demographics
dem = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Demographics.csv"))[
    ["PATNO", "SEX", "BIRTHDT"]
]

# Age at visit
age = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Age_at_visit.csv"))[
    ["PATNO", "EVENT_ID", "AGE_AT_VISIT"]
]

# Education
edu = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Socio-Economics.csv"))[
    ["PATNO", "EDUCYRS"]
]

# Diagnosis
diag = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "Primary_Clinical_Diagnosis.csv")
)[["PATNO", "EVENT_ID", "PRIMDIAG", "OTHNEURO"]]


# Dx status
dx_status = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Participant_Status.csv"))[
    ["PATNO", "COHORT"]
]

# Disease duration / year of diagnosis
# med_cond = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Medical_Conditions_Log.csv"))[
#    ["PATNO", "EVENT_ID", "MHDIAGYR", "MHCAT"]
# ]


# PD dx history / disease duration calc
pd_hist = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "PD_Diagnosis_History.csv"))[
    ["PATNO", "EVENT_ID", "PDDXDT"]
]


# Cognitive Categorization
cog_cat = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "Cognitive_Categorization.csv")
)[["PATNO", "EVENT_ID", "COGSTATE"]]



# L-Dopa
ldopa = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "LEDD_Concomitant_Medication_Log.csv")
)[["PATNO", "EVENT_ID", "LEDD"]]

# meds = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "Concomitant_Medication_Log.csv"))[
#    ["PATNO", "EVENT_ID", "SEQNO", "CMTRT"]
# ]

# UPDRS and Hoehh Yahr
updrs = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "MDS-UPDRS_Part_III.csv"))[
    ["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY"]
]

# Clean UPDRS file. Impute missing ON/OFF values.
# It produces MDS_UPDRS_Part_III_clean.csv file
from livingpark_utils.scripts import pd_status

# updrs = pd.read_csv(
#     os.path.join(str(DATA / "inputs" / "study_files"), "MDS_UPDRS_Part_III_clean.csv")
# )[["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY", "PDTRTMNT"]]

# # MoCA
# moca = pd.read_csv(
#     os.path.join(str(DATA / "inputs" / "study_files"), "Montreal_Cognitive_Assessment__MoCA_.csv")
# )[["PATNO", "EVENT_ID", "MCATOT", "INFODT"]]
# # MoCA - Use screening instead of baseline.
# moca = moca[moca["EVENT_ID"] != "BL"]
# moca["EVENT_ID"].mask(moca["EVENT_ID"] == "SC", "BL", inplace=True)

# %%
import numpy as np

# Calculate disease duration

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

pdxhist = pd_hist[(pd_hist["EVENT_ID"] == "SC") & pd_hist["PDDXDT"].notna()]

pd_dur = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "MDS_UPDRS_Part_III_clean.csv"),
    low_memory=False,
)[["PATNO", "EVENT_ID", "INFODT"]]

PDDXDT_map = dict(zip(pdxhist["PATNO"].values, pdxhist["PDDXDT"].values))
pd_dur["PDDXDT"] = pd_dur["PATNO"].map(PDDXDT_map)

pd_dur["PDXDUR"] = pd_dur.apply(
    lambda row: relativedelta(parse(row["INFODT"]), parse(row["PDDXDT"])).months
    if row["PDDXDT"] is not np.nan
    else np.nan,
    axis=1,
)
pd_dur.drop(labels=["INFODT", "PDDXDT"], inplace=True, axis=1);

# %%
# MRI availability

mri = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "MRI_info.csv"))
mri["EVENT_ID"] = mri["Visit code"]
mri["PATNO"] = mri["Subject ID"]
mri["Sex"] = mri["Sex"].map({"F": 0, "M": 1})
mri = mri.drop(["Subject ID", "Visit code", "Visit", "Age", "Sex"], axis=1)
mri.groupby("EVENT_ID").size().sort_values(ascending=False).head(5)

# %% [markdown]
# # Pair visits

# %%
# Find visit pairs

from collections import defaultdict

visit2month = {
    "BL": 0,
    "V01": 3,
    "V02": 6,
    "V03": 9,
    "V04": 12,
    "V05": 18,
    "V06": 24,
    "V07": 30,
    "V08": 36,
    "V09": 42,
    "V10": 48,
    "V11": 54,
    "V12": 60,
    "V13": 72,
    "V14": 84,
    "V15": 96,
    "V16": 108,
    "V17": 120,
    "V18": 132,
    "V19": 144,
    "V20": 156,
}


def find_visit_pairs(months: int) -> int:
    """Return the pairs of visits closest to each other, given a target time difference in months."""

    diff = float("inf")
    diff_hist = defaultdict(dict)

    for (k, v), (k_, v_) in combinations(visit2month.items(), 2):
        if (diff_ := abs(abs(v - v_) - months)) <= diff:
            diff = diff_
            diff_hist[diff][k] = k_

    return diff_hist[diff]


# %%
def sample_cohort(df, /, *, n):
    _df = df.drop_duplicates(subset=["PATNO"])
    n = min(_df.index.size, n)
    return _df.sample(n=n, replace=False, random_state=1)
    return _df[_df.index.isin(sample)]


# %% [markdown]
# # Select HC

# %%
# diagnosis - use screening instead of baseline when PRIMDIAG is missing at baseline

diag_bl = diag[diag["EVENT_ID"] == "BL"]
diag_other = diag[diag["EVENT_ID"] != "BL"]
diag_other["EVENT_ID"].mask(diag_other["EVENT_ID"] == "SC", "BL", inplace=True)

diag_hc = pd.concat([diag_bl, diag_other])
diag_hc = diag_hc.drop_duplicates()

# %%
# merge into a single df

df_hc = (
    mri.merge(diag_hc, on=["PATNO", "EVENT_ID"])
    .merge(age, on=["PATNO", "EVENT_ID"], how="left")
    .merge(dem, on=["PATNO"], how="left")
    .merge(dx_status, on=["PATNO"], how='left') # check
    .merge(edu, on=["PATNO"], how="left")
    .merge(cog_cat, on=["PATNO", "EVENT_ID"], how="left")
    .merge(pd_hist, on=["PATNO", "EVENT_ID"], how="left")
    .drop_duplicates()
    .groupby("PATNO")
    .filter(lambda g: g["EVENT_ID"].nunique() > 1)
)

# %%
# find how many visit pairs are available for HC group

from itertools import combinations

events = ["BL", "V04", "V06", "V08", "V10"]

print("Unique HC subjects per visit pairs:")
for c in combinations(events, 2):
    v0 = set(
        df_hc[
            (df_hc["EVENT_ID"] == c[0])
            & (df_hc["PRIMDIAG"] == 17)
            #& (df_hc["COHORT"] == 2)
        ]["PATNO"].values
    )
    v1 = set(
        df_hc[
            (df_hc["EVENT_ID"] == c[1])
            & (df_hc["PRIMDIAG"] == 17)
            #& (df_hc["COHORT"] == 2)
        ]["PATNO"].values
    )
    if len(v0 & v1):
        print(
            f"{c[0]:3} & {c[1]:3} = {len(v0 & v1):>3}"
            f" | Month difference: {visit2month[c[1]] - visit2month[c[0]]}"
        )
#       print(v0 & v1)

# %%
def pairs_hc(arg):

    visit_pairs = find_visit_pairs(arg)
    visit_df = df_hc.copy()
    visit_df["NEXT_VISIT"] = visit_df["EVENT_ID"].map(visit_pairs)

    visit_df = visit_df.merge(
        visit_df.drop(
            ["AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS"],
            axis=1,
        ),
        left_on=[
            "PATNO",
            "NEXT_VISIT",
        ],
        right_on=[
            "PATNO",
            "EVENT_ID",
        ],
        suffixes=(None, "_NX"),
    ).drop_duplicates()

    return visit_df.loc[
        (visit_df["PRIMDIAG"] == 17)
        #& (visit_df["COHORT"] == 2)
        & (visit_df["PRIMDIAG_NX"] == 17)
        #& (visit_df["COHORT_NX"] == 2)
    ]


# %%
# build database of all available HC
hc_12 = pairs_hc(12)
hc_24 = pairs_hc(24)
hc_36 = pairs_hc(36)
hc = pd.concat([hc_12, hc_24, hc_36], ignore_index=True)
hc = hc.loc[hc['COHORT'].isin([2,4])]
hc = hc.drop_duplicates(subset=["PATNO"])
hc["dx_group"] = "HC"
print("Unique HC number before selection: ", hc["PATNO"].unique().size)


# %%
hc[hc['PRIMDIAG']==17].groupby('COHORT').count()

# %% [markdown]
# # Data aggregation for PD

# %%
# Merge into a single df for PD

df = (
    mri.merge(diag, on=["PATNO", "EVENT_ID"])
    .merge(age, on=["PATNO", "EVENT_ID"], how="left")
    .merge(dem, on=["PATNO"])
    .merge(edu, on=["PATNO"], how="left")
    .merge(dx_status, on=["PATNO"])
    .merge(pd_hist, on=["PATNO", "EVENT_ID"], how="left")
    .merge(cog_cat, on=["PATNO", "EVENT_ID"])
    .drop_duplicates()
    .groupby("PATNO")
    .filter(lambda g: g["EVENT_ID"].nunique() > 1)
)

# %%
# Pair PD-non-MCI
# Include Primdiag 1 & Cogstate 1 & Cohort 1 & [NHY 1 | 2] & MoCA & no comorb & UPDRS OFF

from itertools import combinations

events = ["BL", "V04", "V06", "V08", "V10"]

print("Unique PD-non-MCI subjects per visit pairs:")
for c in combinations(events, 2):
    v0 = set(
        df[
            (df["EVENT_ID"] == c[0])
            & (df["PRIMDIAG"] == 1)
            & (df["COGSTATE"] == 1)
            & (df["COHORT"] == 1)
            & (df["OTHNEURO"].isnull())
        ]["PATNO"].values
    )
    v1 = set(
        df[
            (df["EVENT_ID"] == c[1])
            & (df["PRIMDIAG"] == 1)
            & (df["COGSTATE"] == 1)
            & (df["COHORT"] == 1)
            & (df["OTHNEURO"].isnull())
        ]["PATNO"].values
    )
    if len(v0 & v1):
        print(
            f"{c[0]:3} & {c[1]:3} = {len(v0 & v1):>3}"
            f" | Month difference: {visit2month[c[1]] - visit2month[c[0]]}"
        )

# %%
# Pair PD-MCI
# Include Primdiag 1 & Cogstate 1 & Cohort 1 & [NHY 1 | 2] & MoCA & no comorb & UPDRS OFF

from itertools import combinations

events = ["BL", "V04", "V06", "V08", "V10"]

print("Unique PD-MCI subjects per visit pairs:")
for c in combinations(events, 2):
    v0 = set(
        df[
            (df["EVENT_ID"] == c[0])
            & (df["PRIMDIAG"] == 1)
            & (df["COGSTATE"] == 2)
            & (df["COHORT"] == 1)
            & (df["OTHNEURO"].isnull())
        ]["PATNO"].values
    )
    v1 = set(
        df[
            (df["EVENT_ID"] == c[1])
            & (df["PRIMDIAG"] == 1)
            & (df["COGSTATE"] == 2)
            & (df["COHORT"] == 1)
            & (df["OTHNEURO"].isnull())
        ]["PATNO"].values
    )
    if len(v0 & v1):
        print(
            f"{c[0]:3} & {c[1]:3} = {len(v0 & v1):>3}"
            f" | Month difference: {visit2month[c[1]] - visit2month[c[0]]}"
        )
        # print(v0 & v1)


# %% [markdown]
# # Select PD-MCI patients

# %%
def pairs_mci(arg):

    visit_pairs = find_visit_pairs(arg)
    visit_df = df.copy()
    visit_df["NEXT_VISIT"] = visit_df["EVENT_ID"].map(visit_pairs)

    visit_df = visit_df.merge(
        visit_df.drop(
            ["AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS"],
            axis=1,
        ),
        left_on=[
            "PATNO",
            "NEXT_VISIT",
        ],
        right_on=[
            "PATNO",
            "EVENT_ID",
        ],
        suffixes=(None, "_NX"),
    ).drop_duplicates()

    return visit_df.loc[
        (visit_df["COGSTATE"] == 2)
        & (visit_df["PRIMDIAG"] == 1)
        & (visit_df["COHORT"] == 1)
        & (visit_df["OTHNEURO"].isnull())
        & (visit_df["COGSTATE_NX"] == 2)
        & (visit_df["PRIMDIAG_NX"] == 1)
        & (visit_df["COHORT_NX"] == 1)
        & (visit_df["OTHNEURO_NX"].isnull())
    ]


# %%
mci = pairs_mci(12)
mci = mci.drop_duplicates(subset=["PATNO"])

mci_24 = pairs_mci(24)
mci = pd.concat([mci, mci_24], ignore_index=True)
mci = mci.drop_duplicates(subset=["PATNO"])

mci_36 = pairs_mci(36)
mci = pd.concat([mci, mci_36], ignore_index=True)
mci = mci.drop_duplicates(subset=["PATNO"])

mci["dx_group"] = "PD-MCI"
print("The final PD-MCI sample size is ", len(mci), " patients.")


# %% [markdown]
# # Select PD-non-MCI patients

# %%
def pairs_nonmci(arg):

    visit_pairs = find_visit_pairs(arg)
    visit_df = df.copy()
    visit_df["NEXT_VISIT"] = visit_df["EVENT_ID"].map(visit_pairs)

    visit_df = visit_df.merge(
        visit_df.drop(
            ["AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS"],
            axis=1,
        ),
        left_on=[
            "PATNO",
            "NEXT_VISIT",
        ],
        right_on=[
            "PATNO",
            "EVENT_ID",
        ],
        suffixes=(None, "_NX"),
    ).drop_duplicates()

    return visit_df.loc[
        (visit_df["COGSTATE"] == 1)
        & (visit_df["PRIMDIAG"] == 1)
        & (visit_df["COHORT"] == 1)
        & (visit_df["OTHNEURO"].isnull())
        & (visit_df["COGSTATE_NX"] == 1)
        & (visit_df["PRIMDIAG_NX"] == 1)
        & (visit_df["COHORT_NX"] == 1)
        & (visit_df["OTHNEURO_NX"].isnull())
    ]


# %%
wo_mci_12 = pairs_nonmci(12)
wo_mci_24 = pairs_nonmci(24)
wo_mci_36 = pairs_nonmci(36)
wo_mci_all = pd.concat([wo_mci_12, wo_mci_24, wo_mci_36], ignore_index=True)
wo_mci_all = wo_mci_all.drop_duplicates(subset=["PATNO"])
wo_mci_all["dx_group"] = "PD-non-MCI"

print("There are ", len(wo_mci_all), " PD-non-MCI patients.")

# %% [markdown]
# ## cohort to download

# %%
cohort = pd.concat([mci, wo_mci_all, hc], ignore_index=True)
cohort = cohort.drop_duplicates(subset=["PATNO"])
print("There are ", len(cohort), " unique subjects.")

# %%
cohort.columns

# %%
# calculate time diff between the visits 

# calculate Duration T2-T1 (months)
# based on MoCA assessment

cohort["Study Date"] = pd.to_datetime(cohort["Study Date"])
cohort["Study Date_NX"] = pd.to_datetime(cohort["Study Date_NX"])


cohort["durationT2_T1"] = (
    cohort["Study Date_NX"] - cohort["Study Date"]
) / np.timedelta64(1, "M")

cohort["durationT2_T1_y"] = cohort["durationT2_T1"] / 12

# %% [markdown]
# # Define PD cohorts with UPDRS

# %%
# Merge into a single df for PD

df_clinical = (
    mri.merge(diag, on=["PATNO", "EVENT_ID"])
    .merge(age, on=["PATNO", "EVENT_ID"], how="left")
    .merge(dem, on=["PATNO"])
    .merge(edu, on=["PATNO"], how="left")
    .merge(dx_status, on=["PATNO"])
    .merge(pd_hist, on=["PATNO", "EVENT_ID"], how="left")
    .merge(cog_cat, on=["PATNO", "EVENT_ID"])
    .merge(updrs, on=["PATNO", "EVENT_ID"]) #
    .drop_duplicates()
    .groupby("PATNO")
    .filter(lambda g: g["EVENT_ID"].nunique() > 1)
)


# %%
# find PD-MCI with UPDRS score

def pairs_mci(arg):

    visit_pairs = find_visit_pairs(arg)
    visit_df = df_clinical.copy()
    visit_df["NEXT_VISIT"] = visit_df["EVENT_ID"].map(visit_pairs)

    visit_df = visit_df.merge(
        visit_df.drop(
            ["AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS"],
            axis=1,
        ),
        left_on=[
            "PATNO",
            "NEXT_VISIT",
        ],
        right_on=[
            "PATNO",
            "EVENT_ID",
        ],
        suffixes=(None, "_NX"),
    ).drop_duplicates()

    return visit_df.loc[
        (visit_df["COGSTATE"] == 2)
        & (visit_df["PRIMDIAG"] == 1)
        & (visit_df["COHORT"] == 1)
        & (visit_df["OTHNEURO"].isnull())
        & (visit_df["PDSTATE"] == "OFF")
        & (visit_df["COGSTATE_NX"] == 2)
        & (visit_df["PRIMDIAG_NX"] == 1)
        & (visit_df["COHORT_NX"] == 1)
        & (visit_df["OTHNEURO_NX"].isnull())
        & (visit_df["PDSTATE_NX"] == "OFF")
    ]


# %%
mci = pairs_mci(12)
mci = mci.drop_duplicates(subset=["PATNO"])

mci_24 = pairs_mci(24)
mci = pd.concat([mci, mci_24], ignore_index=True)
mci = mci.drop_duplicates(subset=["PATNO"])

mci_36 = pairs_mci(36)
mci = pd.concat([mci, mci_36], ignore_index=True)
mci = mci.drop_duplicates(subset=["PATNO"])

mci = mci[~mci['NP3TOT'].isna()]
mci = mci[~mci['NP3TOT_NX'].isna()]

mci["dx_group"] = "PD-MCI"
print("The final PD-MCI sample size is ", len(mci), " patients.")


# %%
# find PD-non-MCI with UPDRS score

def pairs_nonmci(arg):

    visit_pairs = find_visit_pairs(arg)
    visit_df = df_clinical.copy()
    visit_df["NEXT_VISIT"] = visit_df["EVENT_ID"].map(visit_pairs)

    visit_df = visit_df.merge(
        visit_df.drop(
            ["AGE_AT_VISIT", "SEX", "NEXT_VISIT", "EDUCYRS"],
            axis=1,
        ),
        left_on=[
            "PATNO",
            "NEXT_VISIT",
        ],
        right_on=[
            "PATNO",
            "EVENT_ID",
        ],
        suffixes=(None, "_NX"),
    ).drop_duplicates()

    return visit_df.loc[
        (visit_df["COGSTATE"] == 1)
        & (visit_df["PRIMDIAG"] == 1)
        & (visit_df["COHORT"] == 1)
        & (visit_df["OTHNEURO"].isnull())
        & (visit_df["PDSTATE"] == "OFF")
        & (visit_df["COGSTATE_NX"] == 1)
        & (visit_df["PRIMDIAG_NX"] == 1)
        & (visit_df["COHORT_NX"] == 1)
        & (visit_df["OTHNEURO_NX"].isnull())
        & (visit_df["PDSTATE_NX"] == "OFF")
    ]


# %%
wo_mci_12 = pairs_nonmci(12)
wo_mci_24 = pairs_nonmci(24)
wo_mci_all = pd.concat([wo_mci_12, wo_mci_24], ignore_index=True)
wo_mci_all = wo_mci_all.drop_duplicates(subset=["PATNO"])

wo_mci_all = wo_mci_all[~wo_mci_all['NP3TOT'].isna()]
wo_mci_all = wo_mci_all[~wo_mci_all['NP3TOT_NX'].isna()]

print("There are ", len(wo_mci_all), " PD-non-MCI patients to match PD-MCI group.")

# %% [markdown]
# # Download data

# %%
first_visit = cohort

second_visit = first_visit
second_visit = second_visit.drop(["Description", "EVENT_ID"], axis=1)
second_visit.rename(
    columns={"Description_NX": "Description", "EVENT_ID_NX": "EVENT_ID"}, inplace=True)

download_data = first_visit.append(second_visit)

download_data.to_csv(out('download_data.csv'))

# %%
download_data = download_data.loc[download_data["dx_group"] == "PD-non-MCI"]
download_data = download_data[0:100]
download_data.shape

# %%
download_data = download_data.loc[download_data["dx_group"] == "PD-non-MCI"]
download_data.shape

# %%
utils.get_T1_nifti_files(download_data, default=downloader, symlink=True)

# %%
utils.get_T1_nifti_files(download_data, default=downloader, symlink=True)

# %%
utils.download_missing_nifti_files(download_data, link_in_outputs=True)

# %%
download_data.to_csv(out('download_data.csv'))

# %%

# %%

# %%

# %% [markdown]
# # Preprocessing

# %%
# %load_ext slurm_magic

# %%
# save df with all timepoints as json

import json

download_data["PATNO_id"] = "sub-" + download_data["PATNO"].astype(str) + "_ses-" + download_data["EVENT_ID"]

data_to_process = download_data.reset_index()
small_df = data_to_process[["PATNO_id", "File name"]]
json_data = small_df.to_json()
meta = json.loads(json_data)
with open(out("json_data.json"), "w") as fout:
    json.dump(meta, fout, indent=4)

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
# %%sbatch --array=0-285
# #!/bin/bash
#SBATCH --job-name=FS5_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc.%a.out
#SBATCH --error=logs/FS5_preproc.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5.py

# %%

# %%

# %%

# %% [markdown]
# ## Freesurfer 5 - preprocessing step 2 - base template

# %%
# save df with a single input for each subject as json
# double check if the visit pairs are correct!

import json

df_first["PATNO"] = df_first["PATNO"].astype(str)
df_second["PATNO"] = df_second["PATNO"].astype(str)

df_first = df_first.reset_index()
df_second = df_second.reset_index()

data_to_process = df_first
data_to_process["first_visit"] = "sub-" + df_first["PATNO"] + "_ses-" + df_first["EVENT_ID"]
data_to_process["second_visit"] = "sub-" + df_second["PATNO"] + "_ses-" + df_second["EVENT_ID"]
data_to_process["PATNO_base"] = "sub-" + df_first["PATNO"] + "_base"
small_df = data_to_process[["first_visit","second_visit","PATNO_base"]]
json_data = small_df.to_json()
meta = json.loads(json_data)
with open(out("json_data_base.json"), "w") as fout:
    json.dump(meta, fout, indent=4)

# %%
raise RuntimeError("Disabled in the revision workspace: this original pipeline step copies FreeSurfer "
                   "outputs or creates one folder per subject (file quota). The data is already linked in ../data.")

# # copy FS5 data from both visits to the working directory

import shutil

df["PATNO"] = df["PATNO"].astype(str)
df["PATNO_id"] = "sub-" + df["PATNO"] + "_ses-" + df["EVENT_ID"]

for subj in first_visit["PATNO_id"]:
    src='../freesurfer/data_FS5/{subj}'.format(subj=subj)
    #dst=os.getcwd()
    dst=os.path.join(os.getcwd(), subj)
    shutil.copytree(src,dst)

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
# %%sbatch --array=0-126
# #!/bin/bash
#SBATCH --job-name=FS5_base
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_base.%a.out
#SBATCH --error=logs/FS5_preproc_base.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

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

df_long1 = pd.DataFrame(small_df).rename(columns={"first_visit":"visit"})
df_long2 = pd.DataFrame(small_df).rename(columns={"second_visit":"visit"})
df_both = [df_long1,df_long2]
df_long = pd.concat(df_both, ignore_index=True)
df_long = df_long[["visit","PATNO_base"]]
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
# %%sbatch --array=0-253
# #!/bin/bash
#SBATCH --job-name=FS5_long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_long.%a.out
#SBATCH --error=logs/FS5_preproc_long.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

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
# %%sbatch --array=0-253
# #!/bin/bash
#SBATCH --job-name=FS5_longQcache
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_preproc_longQcache.%a.out
#SBATCH --error=logs/FS5_preproc_longQcache.%a.err
#SBATCH --time=2:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_long_qcache.py

# %% [markdown]
# ## FS5 - Prepare the vertex data with long_mris_slopes for longitudinal two stage model

# %%

# %%
# define df based on dx

df["visit_id"] = "sub-" + df["PATNO"] + "_ses-" + df["EVENT_ID"]


df_pdmci = df.loc[df['dx_group'] == "PD-MCI"]
#df_pdmci = df_pdmci.loc[df['visit_no'] == "first"]
df_pdnonmci = df.loc[df['dx_group'] == "PD-non-MCI"]
#df_pdnonmci = df_pdnonmci.loc[df['visit_no'] == "first"]
df_hc = df.loc[df['dx_group'] == "HC"]
#df_hc = df_hc.loc[df['visit_no'] == "first"]

# %%
# calculate Duration T2-T1 
# based on MoCA assessment

df_pdmci["INFODT"] = pd.to_datetime(df_pdmci["INFODT"])
df_pdmci["INFODT_NX"] = pd.to_datetime(df_pdmci["INFODT_NX"])
df_pdnonmci["INFODT"] = pd.to_datetime(df_pdnonmci["INFODT"])
df_pdnonmci["INFODT_NX"] = pd.to_datetime(df_pdnonmci["INFODT_NX"])
df_hc["INFODT"] = pd.to_datetime(df_hc["INFODT"])
df_hc["INFODT_NX"] = pd.to_datetime(df_hc["INFODT_NX"])

df_pdmci["durationT2_T1_y"] = ((
    df_pdmci["INFODT_NX"] - df_pdmci["INFODT"]
) / np.timedelta64(1, "M") / 12 )

df_pdnonmci["durationT2_T1_y"] = ((
    df_pdnonmci["INFODT_NX"] - df_pdnonmci["INFODT"]
) / np.timedelta64(1, "M") / 12 )

df_hc["durationT2_T1_y"] = ((df_hc["INFODT_NX"] - df_hc["INFODT"]) / np.timedelta64(1, "M") / 12 )

# calculate MoCA T1 - T2 difference
df_pdmci["MCATOT_diff"] = df_pdmci["MCATOT"] - df_pdmci["MCATOT_NX"]
df_pdnonmci["MCATOT_diff"] = df_pdnonmci["MCATOT"] - df_pdnonmci["MCATOT_NX"]

# %%
# Create longitudinal QDEC table

df_pdmci_qdec = df_pdmci[["PATNO","EVENT_ID", "durationT2_T1_y", "dx_group"]]
df_pdmci_qdec.sort_values(by=['PATNO',"EVENT_ID"], inplace=True)
df_pdnonmci_qdec = df_pdnonmci[["PATNO","EVENT_ID", "durationT2_T1_y", "dx_group"]]
df_pdnonmci_qdec.sort_values(by=['PATNO',"EVENT_ID"], inplace=True)
df_hc_qdec = df_hc[["PATNO","EVENT_ID", "durationT2_T1_y", "dx_group"]]
df_hc_qdec.sort_values(by=['PATNO',"EVENT_ID"], inplace=True)

df_all = [df_pdmci_qdec,df_pdnonmci_qdec,df_hc_qdec]
df_all = pd.concat(df_all, ignore_index=True)

df_all["PATNO_id"] = "sub-" + df_all["PATNO"] + "_ses-" + df_all["EVENT_ID"]
qdec_table = df_all

qdec_table["fsid"] = qdec_table["PATNO_id"]
qdec_table["fsid-base"] = "sub-" + qdec_table["PATNO"].astype(str) + "_base"
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_long_mris_slopes.py

# %% [markdown]
# ## FS 5 - Create FSGD file for the group analysis - baseline - t-test

# %%

df_pdmci_first = df_pdmci.loc[df['visit_no'] == "first"]
df_pdnonmci_first = df_pdnonmci.loc[df['visit_no'] == "first"]
df_hc_first = df_hc.loc[df['visit_no'] == "first"]

cohort_df_PDall = pd.concat([df_pdnonmci_first, df_pdmci_first])
cohort_df_PDMCI_HC = pd.concat([df_hc_first, df_pdmci_first])
cohort_df_PDnonMCI_HC = pd.concat([df_hc_first, df_pdnonmci_first])

# %%
# PD all group

fsgd_cortThick_diff = cohort_df_PDall[["visit_id","AGE_AT_VISIT","SEX","EDUCYRS","LEDD","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "visit_id", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS","LEDD"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDall_baseline.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDall_baseline.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass PD-non-MCI\nClass PD-MCI\nVariables age sex education LDopa\n"
    )


with open(out("fsgd_cort_group_PDnonMCI_PDMCI_baseline.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDall_baseline.txt", "fsgd_group_cort_PDall_baseline.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())

# PD-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDMCI_HC[["visit_id","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "visit_id", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDMCI_HC_baseline.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDMCI_HC_baseline.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_HC_PDMCI_baseline.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDMCI_HC_baseline.txt", "fsgd_group_cort_PDMCI_HC_baseline.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())
            
# PD-non-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDnonMCI_HC[["visit_id","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "visit_id", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDnonMCI_HC_baseline.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDnonMCI_HC_baseline.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_HC_PDnonMCI_baseline.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDnonMCI_HC_baseline.txt", "fsgd_group_cort_PDnonMCI_HC_baseline.txt"]:
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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.15.baseline.mgh'.format(hemi=hemi,group=group),
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
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_baseline_ttest.py

# %% [markdown]
# ## FS 5 - Create FSGD file for the group analysis - longitudinal - t-test

# %%
# PD all group

fsgd_cortThick_diff = cohort_df_PDall[["PATNO","AGE_AT_VISIT","SEX","EDUCYRS","LEDD","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff["PATNO"] = "sub-" + fsgd_cortThick_diff["PATNO"].astype(str) + "_base"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "PATNO", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS","LEDD"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDall_long.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDall_long.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass PD-non-MCI\nClass PD-MCI\nVariables age sex education LDopa\n"
    )


with open(out("fsgd_cort_group_PDnonMCI_PDMCI_long.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDall_long.txt", "fsgd_group_cort_PDall_long.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())

# PD-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDMCI_HC[["PATNO","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff["PATNO"] = "sub-" + fsgd_cortThick_diff["PATNO"].astype(str) + "_base"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "PATNO", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDMCI_HC_long.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDMCI_HC_long.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_HC_PDMCI_long.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDMCI_HC_long.txt", "fsgd_group_cort_PDMCI_HC_long.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())
            
# PD-non-MCI and HC group

fsgd_cortThick_diff = cohort_df_PDnonMCI_HC[["PATNO","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff["PATNO"] = "sub-" + fsgd_cortThick_diff["PATNO"].astype(str) + "_base"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "PATNO", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_PDnonMCI_HC_long.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_PDnonMCI_HC_long.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_HC_PDnonMCI_long.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_PDnonMCI_HC_long.txt", "fsgd_group_cort_PDnonMCI_HC_long.txt"]:
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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.15.long.mgh'.format(hemi=hemi,group=group),
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
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_long_ttest.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - baseline - t-test

# %%
# create group contrasts

with open(out("con_group_PDnonMCI_PDMCI.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0 0 0")
with open(out("con_group_HC_PDMCI.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0")
with open(out("con_group_HC_PDnonMCI.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0")

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_baseline_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.15.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}.mtx'.format(group=group) 
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_base_ttest.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - longitudinal - t-test

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.15.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}.mtx'.format(group=group) 
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_long_ttest.py

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm_FS5_group_sim_ttest.py

# %% [markdown]
# ## Create FSGD file for the group analysis - baseline - ANCOVA

# %%
cohort_df = pd.concat([df_hc_first, df_pdnonmci_first, df_pdmci_first])

# %%

fsgd_cortThick_diff = cohort_df[["visit_id","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "visit_id", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_baseline.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_baseline.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nClass PD-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_baseline.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_baseline.txt", "fsgd_group_cort_baseline.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())



# %% [markdown]
# ## FS 5 - Stack images for group analyses - baseline - ANCOVA

# %%
# %%writefile preprocess_FS5_mris_preproc_group_baseline.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                                '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_baseline.fsgd',
                                        out='stack.{hemi}.group.thickness.15.baseline.mgh'.format(hemi=hemi),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_preproc_group_baseline.out
#SBATCH --error=logs/FS5_mris_preproc_group_baseline.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_baseline.py

# %% [markdown]
# ## Create FSGD file for the group analysis - longitudinal - ANCOVA

# %%

fsgd_cortThick_diff = cohort_df[["PATNO","AGE_AT_VISIT","SEX","EDUCYRS","dx_group"]]
fsgd_cortThick_diff["Input"] = "Input"
fsgd_cortThick_diff["PATNO"] = "sub-" + fsgd_cortThick_diff["PATNO"].astype(str) + "_base"
fsgd_cortThick_diff = fsgd_cortThick_diff[
    ["Input", "PATNO", "dx_group", "AGE_AT_VISIT","SEX","EDUCYRS"]
]

# generate sample matrix
fsgd_cortThick_diff.to_csv(out("fsgd_group_cort_long.txt"), sep="\t", index=False, header=None)

# generate file header
with open(out("fsgd_group_desc_long.txt"), "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nClass PD-MCI\nVariables age sex education\n"
    )


with open(out("fsgd_cort_group_long.fsgd"), "wb") as outfile:
    for f in ["fsgd_group_desc_long.txt", "fsgd_group_cort_long.txt"]:
        with open(inp(f), "rb") as infile:
            outfile.write(infile.read())

# %% [markdown]
# ## FS 5 - Stack images for group analyses - longitudinal - ANCOVA

# %%
# %%writefile preprocess_FS5_mris_preproc_group_long.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893729"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                        '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_long.fsgd',
                                        out='stack.{hemi}.group.thickness.rate.15.long.mgh'.format(hemi=hemi),
                                       )


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS5_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_mris_preproc_group_long.out
#SBATCH --error=logs/FS5_mris_preproc_group_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_group_long.py

# %% [markdown]
# ## Create FSGD file for the correlational analysis - baseline

# %%
cohort_df_PD = pd.concat([df_pdnonmci_first, df_pdmci_first])

fsgd_cortThick_mcat = cohort_df_PD[["visit_id","AGE_AT_VISIT","SEX","EDUCYRS","MCATOT"]]
fsgd_cortThick_mcat["Input"] = "Input"
fsgd_cortThick_mcat["dx_group"] = "PDall"
fsgd_cortThick_mcat = fsgd_cortThick_mcat[
    ["Input", "visit_id", "dx_group", "MCATOT","AGE_AT_VISIT","SEX","EDUCYRS"]
]


# %%
# generate fsgd files for correlational analysis
# make sure that the order of subjects in each FSGD file is the same as in long.qdec table

for group in ["PDall"]:

    # generate sample matrix
    exec(
        "fsgd_cortThick_mcat.to_csv(out('fsgd_corr_%s_cort.txt'), sep='\t', index=False, header=None)"
        % (group)
    )

    # generate file header
    with open(out("fsgd_corr_{group}_desc.txt".format(group=group)), "w") as f:
        f.write(
            "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName MoCA\nClass {group}\nVariables MoCA age sex education\n".format(
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
    for group in ["PDall"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.15.mgh'.format(hemi=hemi, group=group),
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
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python preprocess_FS5_mris_preproc_corr.py

# %% [markdown]
# ## FS 5 - Run GLM model for the correlational analysis - baseline

# %%
# create contrast for MoCA scores

with open(out("con_corr_MoCA.mtx"), "w") as f:
    f.write("0 1 0 0 0")

# %%
# %%writefile glm5_corr.py

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
    for group in ["PDall"]:   
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_corr_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.15.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_MoCA.mtx'
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_corr.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - baseline - ANCOVA

# %%
# create group contrasts

with open(out("con_group_PDMCI_PDnonMCI.mtx"), "w") as f:
    f.write("0 -1 1 0 0 0 0 0 0 0 0 0")
with open(out("con_group_PDMCI_HC.mtx"), "w") as f:
    f.write("-1 0 1 0 0 0 0 0 0 0 0 0")
with open(out("con_group_PDnonMCI_HC.mtx"), "w") as f:
    f.write("-1 1 0 0 0 0 0 0 0 0 0 0")

# %%
# %%writefile glm5_group_base.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893798"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:  
    out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                        '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_baseline_{hemi}'.format(hemi=hemi),
                                        inputdata='stack.{hemi}.group.thickness.15.baseline.mgh'.format(hemi=hemi),
                                        fsgd='fsgd_cort_group_baseline.fsgd',
                                        con1="con_group_PDMCI_PDnonMCI.mtx", 
                                        con2="con_group_PDMCI_HC.mtx", 
                                        con3="con_group_PDnonMCI_HC.mtx"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_group_bl
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_group_base.out
#SBATCH --error=logs/FS5_glm_group_base.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_group_base.py

# %% [markdown]
# ## FS 5 - Run GLM model for the group analysis - longitudinal - ANCOVA

# %%
# %%writefile glm5_group_long.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893798"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:  
    out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                        '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        hemi=hemi,
                                        outdir='results_group_long_{hemi}'.format(hemi=hemi),
                                        inputdata='stack.{hemi}.group.thickness.rate.15.long.mgh'.format(hemi=hemi),
                                        fsgd='fsgd_cort_group_long.fsgd',
                                        con1="con_group_PDMCI_PDnonMCI.mtx", 
                                        con2="con_group_PDMCI_HC.mtx", 
                                        con3="con_group_PDnonMCI_HC.mtx"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm5_group_lg
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS5_glm_group_long.out
#SBATCH --error=logs/FS5_glm_group_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_group_long.py

# %% [markdown]
# ## FS 5 - Correction for multiple comparison (with mri_glmfit-sim)

# %%
# %%writefile glm5_group_sim.py

# GLM model for the group analysis - ANCOVA

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7893807"
from boutiques.descriptor2func import function
freesurfer = function(zid)

license = str(os.environ["FS_LICENSE"]) 

for hemi in ["lh", "rh"]:
    for time in ["baseline", "long"]:
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                        '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        dir='results_group_{time}_{hemi}'.format(time=time,hemi=hemi),
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
#SBATCH --output=logs/FS5_glm_sim_group.out
#SBATCH --error=logs/FS5_glm_sim_group.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_group_sim.py

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
    for group in ["PDall"]: 
        out_fs = freesurfer('-v', f'{license}:/usr/local/freesurfer/license.txt',
                            '--imagepath', 'ansokol-freesurfer_5.3.simg',
                                        dir='results_corr_{group}_{hemi}'.format(group=group, hemi=hemi),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
export FS_LICENSE=$(realpath FS_licence/license.txt)
python glm5_corr_sim.py

# %% [markdown]
# # Freesurfer 6

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
# %%sbatch --array=0-285
# #!/bin/bash
#SBATCH --job-name=FS6_preproc
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc.%a.out
#SBATCH --error=logs/FS6_preproc.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS6.py

# %%

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
# %%sbatch --array=0-126
# #!/bin/bash
#SBATCH --job-name=FS6_base
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_base.%a.out
#SBATCH --error=logs/FS6_preproc_base.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

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
# %%sbatch --array=0-253
# #!/bin/bash
#SBATCH --job-name=FS6_long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_long.%a.out
#SBATCH --error=logs/FS6_preproc_long.%a.err
#SBATCH --time=10:0:0
#SBATCH --account=<account>

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
# %%sbatch --array=0-253
# #!/bin/bash
#SBATCH --job-name=FS6_longQcache
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_preproc_longQcache.%a.out
#SBATCH --error=logs/FS6_preproc_longQcache.%a.err
#SBATCH --time=2:0:0
#SBATCH --account=<account>

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
#SBATCH --account=<account>

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.15.baseline.mgh'.format(hemi=hemi,group=group),
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
#SBATCH --account=<account>

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        out='stack.{hemi}.group.{group}.thickness.rate.15.long.mgh'.format(hemi=hemi,group=group),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_group_long_ttest.py

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_baseline_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.15.baseline.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_baseline.fsgd'.format(group=group),
                                        con='con_group_{group}.mtx'.format(group=group) 
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
#SBATCH --account=<account>

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:  
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_long_{group}_{hemi}'.format(group=group,hemi=hemi),
                                        inputdata='stack.{hemi}.group.{group}.thickness.rate.15.long.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_cort_group_{group}_long.fsgd'.format(group=group),
                                        con='con_group_{group}.mtx'.format(group=group) 
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS6_group_long_ttest.py

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

for group in ["PDnonMCI_PDMCI", "HC_PDMCI", "HC_PDnonMCI"]:
    for hemi in ["lh", "rh"]:
        for time in ["baseline", "long"]:
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS6_group_sim_ttest.py

# %% [markdown]
# ## FS 6 - Stack images for group analyses - baseline - ANCOVA

# %%
# %%writefile preprocess_FS6_mris_preproc_group_baseline.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                        license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_baseline.fsgd',
                                        out='stack.{hemi}.group.thickness.15.baseline.mgh'.format(hemi=hemi),
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_group_baseline.out
#SBATCH --error=logs/FS6_mris_preproc_group_baseline.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_group_baseline.py

# %% [markdown]
# ## FS 6 - Stack images for group analyses - longitudinal - ANCOVA

# %%
# %%writefile preprocess_FS6_mris_preproc_group_long.py

# Concatenate images from group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900745"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                        license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="long.thickness-rate.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_cort_group_long.fsgd',
                                        out='stack.{hemi}.group.thickness.rate.15.long.mgh'.format(hemi=hemi),
                                       )


# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=FS6_mris_preproc_group
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_mris_preproc_group_long.out
#SBATCH --error=logs/FS6_mris_preproc_group_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_group_long.py

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
    for group in ["PDall"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        cachein="thickness.fwhm15.fsaverage",
                                        target="fsaverage",
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        out='stack.{hemi}.corr.{group}.thickness.rate.15.mgh'.format(hemi=hemi, group=group),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS6_mris_preproc_corr.py

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
    for group in ["PDall"]:   
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_corr_{group}_{hemi}'.format(group=group, hemi=hemi),
                                        inputdata='stack.{hemi}.corr.{group}.thickness.rate.15.mgh'.format(hemi=hemi,group=group),
                                        fsgd='fsgd_corr_{group}_group.fsgd'.format(group=group),
                                        con='con_corr_MoCA.mtx'
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_corr.py

# %% [markdown]
# ## FS 6 - Run GLM model for the group analysis - baseline - ANCOVA

# %%
# %%writefile glm6_group_base.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900731"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:  
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                        license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_baseline_{hemi}'.format(hemi=hemi),
                                        inputdata='stack.{hemi}.group.thickness.15.baseline.mgh'.format(hemi=hemi),
                                        fsgd='fsgd_cort_group_baseline.fsgd',
                                        con1="con_group_PDMCI_PDnonMCI.mtx", 
                                        con2="con_group_PDMCI_HC.mtx", 
                                        con3="con_group_PDnonMCI_HC.mtx"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_group_bl
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_group_base.out
#SBATCH --error=logs/FS6_glm_group_base.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_group_base.py

# %% [markdown]
# ## FS 6 - Run GLM model for the group analysis - longitudinal - ANCOVA

# %%
# %%writefile glm6_group_long.py

# GLM model for the group analysis

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900731"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:  
    out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                        license="FS_license/license.txt",
                                        hemi=hemi,
                                        outdir='results_group_long_{hemi}'.format(hemi=hemi),
                                        inputdata='stack.{hemi}.group.thickness.rate.15.long.mgh'.format(hemi=hemi),
                                        fsgd='fsgd_cort_group_long.fsgd',
                                        con1="con_group_PDMCI_PDnonMCI.mtx", 
                                        con2="con_group_PDMCI_HC.mtx", 
                                        con3="con_group_PDnonMCI_HC.mtx"
                                       )

# %%
# %%sbatch
# #!/bin/bash
#SBATCH --job-name=glm6_group_lg
#SBATCH --mem=1G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS6_glm_group_long.out
#SBATCH --error=logs/FS6_glm_group_long.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_group_long.py

# %% [markdown]
# ## FS 6 - Correction for multiple comparison (with mri_glmfit-sim)

# %%
# %%writefile glm6_group_sim.py

# GLM model for the group analysis - ANCOVA

import os
import json

import boutiques
from boutiques import bosh
zid = "zenodo.7900735"
from boutiques.descriptor2func import function
freesurfer = function(zid)

for hemi in ["lh", "rh"]:
    for time in ["baseline", "long"]:
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        dir='results_group_{time}_{hemi}'.format(time=time,hemi=hemi),
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
#SBATCH --output=logs/FS6_glm_sim_group.out
#SBATCH --error=logs/FS6_glm_sim_group.err
#SBATCH --time=0:10:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_group_sim.py

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
    for group in ["PDall"]: 
        out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_6.0.1.simg',
                            license="FS_license/license.txt",
                                        dir='results_corr_{group}_{hemi}'.format(group=group, hemi=hemi),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm6_corr_sim.py

# %%

# %% [markdown]
# # Freesurfer 7.3.2

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

with open('json_data_err7_cross.json') as fin:
    subject_map = json.load(fin)

    
out_fs = freesurfer('--imagepath', 'ansokol-freesurfer_7.3.2.simg',
                                        input=subject_map["File name"][task_id], qcache_flag=True,
                                        license="FS_license/license.txt",
                                        subjid=str(subject_map["PATNO_id"][task_id]),
                                       )

# %%
# %%sbatch --array=0-2
# #!/bin/bash
#SBATCH --job-name=FS7_preproc
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=logs/FS7_preproc_err.%a.out
#SBATCH --error=logs/FS7_preproc_err.%a.err
#SBATCH --time=96:0:0
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_err.py

# %% [markdown]
# # 7 long

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_long.py

# %% [markdown]
# # Freesurfer 7 - preprocessing step 4 - Qcache

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity

python preprocess_FS7_long_qcache.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_long_mris_slopes.py

# %%

# %%
# create group contrasts

with open(out("con_group_HC_PDnonMCI_base.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0")
    
# create group contrasts

with open(out("con_group_HC_PDnonMCI_long.mtx"), "w") as f:
    f.write("1 -1 0 0 0 0 0 0")
    
# create contrast for MoCA scores

with open(out("con_corr_UPDRS_base.mtx"), "w") as f:
    f.write("0 1 0 0")
    
# create contrast for UPDRS scores

with open(out("con_corr_UPDRS_long.mtx"), "w") as f:
    f.write("0 1 0 0 0")

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_group_baseline_ttest.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_group_long_ttest.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_corr.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python preprocess_FS7_mris_preproc_corr_long.py

# %%

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
                                        outdir='results_group_baseline_{group}_{hemi}'.format(group=group,hemi=hemi),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_base_ttest.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_long_ttest.py

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
                                        outdir='results_corr_baseline_{group}_{hemi}'.format(group=group, hemi=hemi),
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr.py

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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr_long.py

# %%

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
        for time in ["baseline", "long"]:
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm_FS7_group_sim_ttest.py

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
        for time in ["baseline", "long"]:
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
#SBATCH --account=<account>

. venv/bin/activate # opens virtual environment. change depending where you proprocess the data  

module load singularity
python glm7_corr_sim.py

# %%

# %%

# %%

# %% [markdown]
# # Copy data from the server

# %%
raise RuntimeError("Disabled in the revision workspace: this original pipeline step copies FreeSurfer "
                   "outputs or creates one folder per subject (file quota). The data is already linked in ../data.")

import os

for pat in df["visit_id"]:
    os.makedirs('stats/FS5/{pat}/stats'.format(pat=pat), exist_ok=True)
    os.makedirs('stats/FS6/{pat}/stats'.format(pat=pat), exist_ok=True)
    os.makedirs('stats/FS7/{pat}/stats'.format(pat=pat), exist_ok=True)

# %%
long_visits = df["visit_id"]
long_visits.to_csv(out("long_visits.csv"))

# %% language="bash"
#
# for i in $(cut -d ',' -f2 long_visits.csv);
#
# do 
# scp $REMOTE:$PROJECT_DIR/mak/$i/stats/aseg.stats $LOCAL_DIR/mak/stats/FS5/$i/stats/aseg.stats
# scp $REMOTE:$PROJECT_DIR/mak/$i/stats/lh.aparc.a2009s.stats $LOCAL_DIR/mak/stats/FS5/$i/stats/lh.aparc.a2009s.stats
# scp $REMOTE:$PROJECT_DIR/mak/$i/stats/rh.aparc.a2009s.stats $LOCAL_DIR/mak/stats/FS5/$i/stats/rh.aparc.a2009s.stats
#
# done

# %%
raise RuntimeError("Disabled in the revision workspace: this original pipeline step copies FreeSurfer "
                   "outputs or creates one folder per subject (file quota). The data is already linked in ../data.")


for pat in df["visit_id"]:
    os.makedirs('data/FS5/{pat}/mri'.format(pat=pat), exist_ok=True)
    os.makedirs('data/FS6/{pat}/mri'.format(pat=pat), exist_ok=True)
    os.makedirs('data/FS7/{pat}/mri'.format(pat=pat), exist_ok=True)
    os.makedirs('data/FS5/{pat}/surf'.format(pat=pat), exist_ok=True)
    os.makedirs('data/FS6/{pat}/surf'.format(pat=pat), exist_ok=True)
    os.makedirs('data/FS7/{pat}/surf'.format(pat=pat), exist_ok=True)

# %% language="bash"
#
# for i in $(cut -d ',' -f2 long_visits.csv);
#
# do 
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/mri/T1.mgz $LOCAL_DIR/mak/data/FS5/$i/mri/T1.mgz
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/mri/brainmask.mgz $LOCAL_DIR/mak/data/FS5/$i/mri/brainmask.mgz
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/mri/aseg.mgz $LOCAL_DIR/mak/data/FS5/$i/mri/aseg.mgz
#
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/surf/lh.pial $LOCAL_DIR/mak/data/FS5/$i/surf/lh.pial
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/surf/lh.white $LOCAL_DIR/mak/data/FS5/$i/surf/lh.white
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/surf/rh.pial $LOCAL_DIR/mak/data/FS5/$i/surf/rh.pial 
# scp $REMOTE:$PROJECT_DIR/mak/output/FS5/$i/surf/rh.white $LOCAL_DIR/mak/data/FS5/$i/surf/rh.white
# done

# %% language="bash"
#
# scp -r $REMOTE:$PROJECT_DIR/mak/output/FS5/results* $LOCAL_DIR/mak/stats/results_FS5/
#

# %%
