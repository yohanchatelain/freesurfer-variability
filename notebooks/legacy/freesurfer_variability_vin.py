# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv (3.11.4)
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
INPUT_SUBDIR = "tables"  # default ../data subdirectory for files written by the original pipeline


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
# # Initial setup

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

# %% [markdown]
# # Cohort preparation
#
# We will build a cohort using data from Parkinson's Progression Markers Initiative (PPMI). We will select patients with Parkinson's disease (PD) with (PD-MCI) and without Mild Cognitive Impairment (PD-non-MCI) as well as healthy controls (HC). The cohort will be built directly from PPMI Study Data so that it can be replicated and updated whenever necessary.
#
# We will use the LivingPark utils library to download files from the notebook. If files are already present in the notebook cache, they won't be downloaded again. Otherwise, you will need to enter your PPMI username and password. **In case you don't have a PPMI account, you can request one [here](http://ppmi-info.org).**

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

# PD dx history / disease duration calc
pd_hist = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "PD_Diagnosis_History.csv"))[
    ["PATNO", "EVENT_ID", "PDDXDT"]
]

# Cognitive Categorization
cog_cat = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "Cognitive_Categorization.csv")
)[["PATNO", "EVENT_ID", "COGSTATE"]]

# UPDRS and Hoehh Yahr
# updrs = pd.read_csv(os.path.join(str(DATA / "inputs" / "study_files"), "MDS-UPDRS_Part_III.csv"))[
#     ["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY"]
# ]

# Clean UPDRS file. Impute missing ON/OFF values.
# It produces MDS_UPDRS_Part_III_clean.csv file
# from livingpark_utils.scripts import pd_status

updrs = pd.read_csv(
    os.path.join(str(DATA / "inputs" / "study_files"), "MDS_UPDRS_Part_III_clean.csv")
)[["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY", "PDTRTMNT"]]

# %% [markdown]
# We will also need file `MRI_info.csv` produced by another LivingPark notebook available at https://github.com/LivingPark-MRI/livingpark-utils/blob/main/livingpark_utils/notebooks/mri_metadata.ipynb. This file contains a list of T1-weighted MRI images. 
#     

# %%
from livingpark_utils.scripts import run
from livingpark_utils.scripts import mri_metadata

run.mri_metadata()

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
# ## Pair visits

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
    .merge(dx_status, on=["PATNO"], how="left")  # check
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
        df_hc[(df_hc["EVENT_ID"] == c[0]) & (df_hc["PRIMDIAG"] == 17)]["PATNO"].values
    )
    v1 = set(
        df_hc[(df_hc["EVENT_ID"] == c[1]) & (df_hc["PRIMDIAG"] == 17)]["PATNO"].values
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

    return visit_df.loc[(visit_df["PRIMDIAG"] == 17) & (visit_df["PRIMDIAG_NX"] == 17)]


# %%
# build database of all available HC
hc_12 = pairs_hc(12)
hc_24 = pairs_hc(24)
hc_36 = pairs_hc(36)
hc = pd.concat([hc_12, hc_24, hc_36], ignore_index=True)
hc = hc.loc[hc["COHORT"].isin([2, 4])]
hc = hc.drop_duplicates(subset=["PATNO"])
hc["dx_group"] = "HC"
print("Unique HC number before selection: ", hc["PATNO"].unique().size)

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
print("There are ", len(mci), " PD-MCI patients.")


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
from dateutil.relativedelta import relativedelta
import numpy as np

# Calculate time difference between the visits (Duration T2-T1)
cohort["Study Date"] = pd.to_datetime(cohort["Study Date"])
cohort["Study Date_NX"] = pd.to_datetime(cohort["Study Date_NX"])


# Function to calculate months between dates
def months_between(date1, date2):
    if pd.isna(date1) or pd.isna(date2):
        return np.nan
    rd = relativedelta(date2, date1)
    return rd.years * 12 + rd.months + rd.days / 30.44


cohort["durationT2_T1"] = cohort.apply(
    lambda row: months_between(row["Study Date"], row["Study Date_NX"]), axis=1
)

cohort["durationT2_T1_y"] = cohort["durationT2_T1"] / 12

# %% [markdown]
# ## Descriptive statistics (computational cohort)
#
# To calculate descriptive statistics we exclude images that failed preprocessing in the next steps. PPMI's Data Usage Agreement prevents us from publicaly sharing subjects' identifiers.

# %%
cohort_stat = cohort
cohort_stat["PATNO_id"] = (
    "sub-" + cohort_stat["PATNO"].astype(str) + "_ses-" + cohort_stat["EVENT_ID"]
)

# exclude images due to the preprocessing failure
# failed = pd.read_csv(inp("failed.csv"))

# for i in failed["PATNO_id"]:
#     cohort_stat = cohort_stat[cohort_stat["PATNO_id"].str.contains(f"{i}") == False]

# %%
from collections.abc import Iterable

import rich
from rich.console import Console
from rich.table import Table


def cohort_summary(*, hc, nc, mci, title):
    def gen_row(D, *, agg, col, f="4.1f", sep=" ± "):
        if not isinstance(agg, str) and isinstance(agg, Iterable):
            return [f"{sep}".join([f"{d.loc[a][col]:{f}}" for a in agg]) for d in D]
        else:
            return [f"{d.loc[agg][col]:{f}}" for d in D]

    def gender_ratio(df):
        male_count = df[df["SEX"] == 1]["PATNO"].nunique()
        return f"{male_count:.0f}, {male_count / df['PATNO'].nunique() * 100:.1f}%"

    D = [hc.describe(), nc.describe(), mci.describe()]

    table = Table(title=title, box=rich.box.SIMPLE_HEAVY, show_footer=True)

    table.add_column("Subject groups", footer="Values expressed as mean ± SD.")
    table.add_column("HC")
    table.add_column("PD-non-MCI")
    table.add_column("PD-MCI")

    table.add_row("n", *gen_row(D, agg="count", col="PATNO", f=".0f"))
    table.add_row("Age (y)", *gen_row(D, agg=["mean", "std"], col="AGE_AT_VISIT"))
    table.add_row(
        "Age range", *gen_row(D, agg=["min", "max"], col="AGE_AT_VISIT", sep=" - ")
    )
    table.add_row(
        "Gender (male, %)", gender_ratio(hc), gender_ratio(nc), gender_ratio(mci)
    )
    table.add_row("Education (y)", *gen_row(D, agg=["mean", "std"], col="EDUCYRS"))

    console = Console()
    console.print(table)


# %%
hc = cohort_stat[cohort_stat["dx_group"] == "HC"]
nc = cohort_stat[cohort_stat["dx_group"] == "PD-non-MCI"]
mci = cohort_stat[cohort_stat["dx_group"] == "PD-MCI"]

cohort_summary(
    hc=hc,
    nc=nc,
    mci=mci,
    title="Demographic and clinical characteristics",
)

# %%
from scipy.stats import ttest_ind, chi2_contingency

pd_both = pd.concat([nc, mci])

[t_age, p_age] = ttest_ind(pd_both["AGE_AT_VISIT"], hc["AGE_AT_VISIT"])
[t_edu, p_edu] = ttest_ind(pd_both["EDUCYRS"], hc["EDUCYRS"], nan_policy="omit")

print(
    "p-value for group difference (PD vs HC) in age is ",
    p_age,
    ", in education is ",
    p_edu,
)

# %%
# test difference in sex frequency between PD and HC groups

a1 = pd_both["SEX"].value_counts()[0]
a2 = pd_both["SEX"].value_counts()[1]
b1 = hc["SEX"].value_counts()[0]
b2 = hc["SEX"].value_counts()[1]

obs = np.array([[a1, a2], [b1, b2]])
chi2_contingency(obs)

# %% [markdown]
# ## Define PD cohort with UPDRS

# %%
# Merge into a single dataframe

df_clinical = (
    mri.merge(diag, on=["PATNO", "EVENT_ID"])
    .merge(age, on=["PATNO", "EVENT_ID"], how="left")
    .merge(dem, on=["PATNO"])
    .merge(edu, on=["PATNO"], how="left")
    .merge(dx_status, on=["PATNO"])
    .merge(pd_hist, on=["PATNO", "EVENT_ID"], how="left")
    .merge(cog_cat, on=["PATNO", "EVENT_ID"])
    .merge(updrs, on=["PATNO", "EVENT_ID"])  #
    .drop_duplicates()
    .groupby("PATNO")
    .filter(lambda g: g["EVENT_ID"].nunique() > 1)
)

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
wo_mci_36 = pairs_nonmci(36)
wo_mci_all = pd.concat([wo_mci_12, wo_mci_24, wo_mci_36], ignore_index=True)
wo_mci_all = wo_mci_all.drop_duplicates(subset=["PATNO"])
wo_mci_all["dx_group"] = "PD-non-MCI"

wo_mci_all = wo_mci_all[~wo_mci_all["NP3TOT"].isna()]
wo_mci_all = wo_mci_all[~wo_mci_all["NP3TOT_NX"].isna()]

print("There are ", len(wo_mci_all), " PD-non-MCI patients.")

# %%
# calculate the time difference between the visits
pd_clinical = wo_mci_all

pd_clinical["Study Date"] = pd.to_datetime(pd_clinical["Study Date"])
pd_clinical["Study Date_NX"] = pd.to_datetime(pd_clinical["Study Date_NX"])

# Calculate difference in years first, then convert to months
pd_clinical["durationT2_T1_y"] = (
    pd_clinical["Study Date_NX"] - pd_clinical["Study Date"]
).dt.days / 365.25  # Using 365.25 to account for leap years

# Convert years to months
pd_clinical["durationT2_T1"] = pd_clinical["durationT2_T1_y"] * 12

# visit ID
pd_clinical["PATNO_id"] = (
    "sub-" + pd_clinical["PATNO"].astype(str) + "_ses-" + pd_clinical["EVENT_ID"]
)

# calculate the change of UPDRS score
pd_clinical["NP3TOT_change"] = pd_clinical["NP3TOT"] - pd_clinical["NP3TOT_NX"]

# pd_clinical = pd_clinical[pd_clinical.PATNO_id.isin(success["subject"].values)]
# pd_clinical = pd_clinical[~pd_clinical["PATNO_id"].isin(failed["subject"].values)]
pd_clinical.to_csv(out("pd_clinical.csv"), index=False)
print(f"PD clinical data saved to {out('pd_clinical.csv')}")

# %% [markdown]
# ## Descriptive statistics (clinical cohort)

# %%
from collections.abc import Iterable

import rich
from rich.console import Console
from rich.table import Table


def cohort_summary(*, hc, nc, title):
    def gen_row(D, *, agg, col, f="4.1f", sep=" ± "):
        if not isinstance(agg, str) and isinstance(agg, Iterable):
            return [f"{sep}".join([f"{d.loc[a][col]:{f}}" for a in agg]) for d in D]
        else:
            return [f"{d.loc[agg][col]:{f}}" for d in D]

    def gender_ratio(df):
        male_count = df[df["SEX"] == 1]["PATNO"].nunique()
        return f"{male_count:.0f}, {male_count / df['PATNO'].nunique() * 100:.1f}%"

    D = [hc.describe(), nc.describe()]

    table = Table(title=title, box=rich.box.SIMPLE_HEAVY, show_footer=True)

    table.add_column("Subject groups", footer="Values expressed as mean ± SD.")
    table.add_column("HC")
    table.add_column("PD-non-MCI")
    # table.add_column("[italic]p")  # TODO

    table.add_row("n", *gen_row(D, agg="count", col="PATNO", f=".0f"))
    table.add_row("Age (y)", *gen_row(D, agg=["mean", "std"], col="AGE_AT_VISIT"))
    table.add_row(
        "Age range", *gen_row(D, agg=["min", "max"], col="AGE_AT_VISIT", sep=" - ")
    )
    table.add_row("Gender (male, %)", gender_ratio(hc), gender_ratio(nc))
    table.add_row("Education (y)", *gen_row(D, agg=["mean", "std"], col="EDUCYRS"))
    table.add_row(
        "UPDRS III OFF baseline", "", *gen_row(D[1:], agg=["mean", "std"], col="NP3TOT")
    )
    table.add_row(
        "UPDRS III OFF follow-up",
        "",
        *gen_row(D[1:], agg=["mean", "std"], col="NP3TOT_NX"),
    )
    table.add_row(
        "Duration T2 - T1 (y)", *gen_row(D, agg=["mean", "std"], col="durationT2_T1_y")
    )

    console = Console()
    console.print(table)


# %%
pd_nonmci_clinical = pd_clinical[pd_clinical["dx_group"] == "PD-non-MCI"]

cohort_summary(
    hc=hc,
    nc=pd_nonmci_clinical,
    title="Demographic and clinical characteristics",
)

# %%
[t_age, p_age] = ttest_ind(pd_nonmci_clinical["AGE_AT_VISIT"], hc["AGE_AT_VISIT"])
[t_edu, p_edu] = ttest_ind(
    pd_nonmci_clinical["EDUCYRS"], hc["EDUCYRS"], nan_policy="omit"
)
[t_dur, p_dur] = ttest_ind(pd_nonmci_clinical["durationT2_T1_y"], hc["durationT2_T1_y"])

print(
    "p-value for group difference (PD vs HC) in age is ",
    p_age,
    ", in education is ",
    p_edu,
    " in time difference between the two visits ",
    p_dur,
)

# %%
# test difference in sex frequency between PD and HC groups

a1 = pd_nonmci_clinical["SEX"].value_counts()[0]
a2 = pd_nonmci_clinical["SEX"].value_counts()[1]
b1 = hc["SEX"].value_counts()[0]
b2 = hc["SEX"].value_counts()[1]

obs = np.array([[a1, a2], [b1, b2]])
chi2_contingency(obs)

# %% [markdown]
# ## Download data

# %%
first_visit = cohort

second_visit = first_visit
second_visit = second_visit.drop(["Description", "EVENT_ID"], axis=1)
second_visit.rename(
    columns={"Description_NX": "Description", "EVENT_ID_NX": "EVENT_ID"}, inplace=True
)

# download_data = first_visit.append(second_visit)
download_data = pd.concat([first_visit, second_visit])

download_data["PATNO_id"] = (
    "sub-" + download_data["PATNO"].astype(str) + "_ses-" + download_data["EVENT_ID"]
)

download_data.to_csv(out('download_data.csv'))

# %%
# utils.download_missing_nifti_files(download_data, link_in_outputs=True)

# %% [markdown]
# We will compare PD cohorts from the computational stage and clinical (longitudinal) stage of analyses. Clinical sample is a subsample of the computational stample (i.e., no new subjects are added). However, there might be a difference which visit is considered as baseline or follow-up.

# %%
first_visit_clin = pd_clinical

second_visit_clin = first_visit_clin
second_visit_clin = second_visit_clin.drop(["Description", "EVENT_ID"], axis=1)
second_visit_clin.rename(
    columns={"Description_NX": "Description", "EVENT_ID_NX": "EVENT_ID"}, inplace=True
)

# visit_clin = first_visit_clin.append(second_visit_clin)
visit_clin = pd.concat([first_visit_clin, second_visit_clin])

visit_clin["PATNO_id"] = (
    "sub-" + visit_clin["PATNO"].astype(str) + "_ses-" + visit_clin["EVENT_ID"]
)

# %%
# test if there are any new images in the clinical sample

visit_clin_pat = visit_clin[["PATNO", "PATNO_id"]]
download_data_pat = download_data[["PATNO", "PATNO_id"]]

test = visit_clin_pat.merge(download_data_pat)
df_diff = pd.concat([visit_clin_pat, test]).drop_duplicates(keep=False)

len(visit_clin_pat.merge(download_data_pat)) == len(visit_clin_pat)

# %%
# download missing images in clinical stage

# visit_clin_download = pd.read_csv(inp("clinical_diff.csv"))

# utils.download_missing_nifti_files(visit_clin_download, link_in_outputs=True)

# %%
Throw new error

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

# %% [markdown]
# # MRIQC
#
# Quality control of the input images performed with MRIQC 22.0.1.

# %%
cohort = pd.read_csv(inp("download_data.csv"))
df_data = cohort
df_data["PATNO_id"] = (
    "sub-" + df_data["PATNO"].astype(str) + "_ses-" + df_data["EVENT_ID"]
)

# exclude images due to the preprocessing failure
failed = pd.read_csv(inp("failed.csv"))

for i in failed["PATNO_id"]:
    df_data = df_data[df_data["PATNO_id"].str.contains(f"{i}") == False]

df_qc = df_data
df_data.to_csv(out("subjects.csv"), index=False)

# %%
metrics = [
    "cjv",
    "cnr",
    "fber",
    "qi_1",
    "qi_2",
    "rpve_gm",
    "snr_gm",
    "snr_total",
    "snrd_gm",
    "snrd_total",
]

for subj in df_qc["PATNO"]:
    for metric in metrics:
        event = df_qc.loc[df_qc["PATNO"] == subj, "EVENT_ID"].iloc[0]
        file = f"{DATA}/source_files/clinical/mriqc/output/sub-{subj}/ses-{event}/anat/sub-{subj}_T1w.json"

        with open(file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if line.find(f"{metric}") != -1:
                    # print(float(line.split()[1].rstrip(',')))
                    df_qc.loc[
                        df_qc["PATNO"] == subj,
                        metric,
                    ] = float(line.split()[1].rstrip(","))

# %%
from scipy.stats import ttest_ind

group1 = df_qc.loc[df_qc["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]
group2 = df_qc[df_qc["dx_group"] == "HC"]

group_qc = {}

for metric in metrics:
    group_qc[f"t_{metric}"], group_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )
    group_qc[f"t_{metric}"], group_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )

# %%
group_qc

# %% [markdown]
# # Postprocessing
#
# ## Computational analyses

# %%
import pandas as pd
import re

patnos = []

with open(inp("orig.txt")) as f:
    for line in f:
        line = line.strip()

        # m = re.match(r"sub-(\d+)_ses-", line)
        # if m:
        patnos.append(line)

aseg_table = pd.DataFrame({"PATNO_id": patnos})

print(aseg_table.head())

# %%
# TODO remove later
# Images excluded after QC; the list holds participant IDs, so it stays in data/ (not versioned)
to_drop = [s.strip() for s in open(inp("excluded_images.txt")) if s.strip()]

aseg_table = aseg_table[~aseg_table["PATNO_id"].isin(to_drop)]

# %%
# extract structural measures
# aseg_table = df_data

ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

# Canonical (FS7/FS8-style) Destrieux ROI names, using "and"/hyphen conventions
ROIs_aparc = [
    "G_and_S_frontomargin",
    "G_and_S_occipital_inf",
    "G_and_S_paracentral",
    "G_and_S_subcentral",
    "G_and_S_transv_frontopol",
    "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral",
    "G_cuneus",
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
    "G_Ins_lg_and_S_cent_ins",
    "G_insular_short",
    "G_occipital_middle",
    "G_occipital_sup",
    "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual",
    "G_oc-temp_med-Parahip",
    "G_orbital",
    "G_pariet_inf-Angular",
    "G_pariet_inf-Supramar",
    "G_parietal_sup",
    "G_postcentral",
    "G_precentral",
    "G_precuneus",
    "G_rectus",
    "G_subcallosal",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    "G_temporal_inf",
    "G_temporal_middle",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Pole_occipital",
    "Pole_temporal",
    "S_calcarine",
    "S_central",
    "S_cingul-Marginalis",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
    "S_collat_transv_ant",
    "S_collat_transv_post",
    "S_front_inf",
    "S_front_middle",
    "S_front_sup",
    "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans",
    "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal",
    "S_occipital_ant",
    "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral",
    "S_orbital_med-olfact",
    "S_orbital-H_Shaped",
    "S_parieto_occipital",
    "S_pericallosal",
    "S_postcentral",
    "S_precentral-inf-part",
    "S_precentral-sup-part",
    "S_suborbital",
    "S_subparietal",
    "S_temporal_inf",
    "S_temporal_sup",
    "S_temporal_transverse",
]

# FS6 uses "&" instead of "and"/"_and_" in a subset of names. Map canonical -> FS6 spelling
# for exactly the ROIs that differ; everything else is spelled the same in FS6.
FS6_NAME_OVERRIDES = {
    "G_and_S_frontomargin": "G&S_frontomargin",
    "G_and_S_occipital_inf": "G&S_occipital_inf",
    "G_and_S_paracentral": "G&S_paracentral",
    "G_and_S_subcentral": "G&S_subcentral",
    "G_and_S_transv_frontopol": "G&S_transv_frontopol",
    "G_and_S_cingul-Ant": "G&S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant": "G&S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post": "G&S_cingul-Mid-Post",
    "G_Ins_lg_and_S_cent_ins": "G_Ins_lg&S_cent_ins",
    "S_intrapariet_and_P_trans": "S_intrapariet&P_trans",
    "S_oc_middle_and_Lunatus": "S_oc_middle&Lunatus",
    "S_oc_sup_and_transversal": "S_oc_sup&transversal",
    "S_oc-temp_med_and_Lingual": "S_oc-temp_med&Lingual",
}


def roi_label_for_version(roi, version):
    """Return the on-disk Destrieux label for this ROI under a given FS version."""
    if version == "FS6":
        return FS6_NAME_OVERRIDES.get(roi, roi)
    return roi


def parse_aparc_stats(path):
    """Read a lh/rh.aparc.a2009s.stats file once, returning {roi_label: (surf, vol, ct)}."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            roi = parts[0]
            try:
                surf = float(parts[2])
                vol = float(parts[3])
                ct = float(parts[4])
            except ValueError:
                continue
            data[roi] = (surf, vol, ct)
    return data


def parse_aseg_stats(path):
    """Read an aseg.stats file once, returning (tiv, {roi_label: volume})."""
    tiv = None
    vols = {}
    with open(path, "r") as f:
        for line in f:
            if "Estimated Total Intracranial Volume" in line:
                # Header line format: "# Measure EstimatedTotalIntraCranialVol, ..., <value>, mm^3"
                tiv = float(line.split(",")[3])
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # Standard aseg.stats data rows: Index SegId NVoxels Volume_mm3 StructName ...
            # Volume is field index 3, ROI name is field index 4.
            try:
                roi_name = parts[4]
                roi_vol = float(parts[3])
            except (IndexError, ValueError):
                continue
            vols[roi_name] = roi_vol
    return tiv, vols


for subj in aseg_table["PATNO_id"]:
    for version in ["FS5", "FS6", "FS7", "FS8"]:

        # --- aseg.stats: TIV + subcortical ROI volumes (one read) ---
        aseg_file = str(DATA) + "/{version}/{subidd}/stats/aseg.stats".format(
            subidd=subj, version=version
        )
        try:
            tiv, aseg_vols = parse_aseg_stats(aseg_file)
        except FileNotFoundError:
            tiv, aseg_vols = None, {}

        if tiv is not None:
            aseg_table.loc[
                aseg_table["PATNO_id"] == subj, "TIV_{version}".format(version=version)
            ] = tiv

        for roi in ROIs:
            if roi in aseg_vols:
                aseg_table.loc[
                    aseg_table["PATNO_id"] == subj, roi + "_" + version
                ] = aseg_vols[roi]

        # --- lh/rh aparc.a2009s.stats: cortical volume, surface area, thickness (one read each) ---
        for hemi in ["lh", "rh"]:
            aparc_file = str(DATA) + "/{version}/{subidd}/stats/{hemi}.aparc.a2009s.stats".format(
                subidd=subj, version=version, hemi=hemi
            )
            try:
                aparc_data = parse_aparc_stats(aparc_file)
            except FileNotFoundError:
                aparc_data = {}

            for roi in ROIs_aparc:
                on_disk_label = roi_label_for_version(roi, version)
                if on_disk_label not in aparc_data:
                    continue
                surf, vol, ct = aparc_data[on_disk_label]

                aseg_table.loc[
                    aseg_table["PATNO_id"] == subj,
                    roi + "_" + hemi + "_vol_" + version,
                ] = vol
                aseg_table.loc[
                    aseg_table["PATNO_id"] == subj,
                    roi + "_" + hemi + "_surf_" + version,
                ] = surf
                aseg_table.loc[
                    aseg_table["PATNO_id"] == subj,
                    roi + "_" + hemi + "_ct_" + version,
                ] = ct

# %%
# calculate V2-V1/V1 and take an absolute value
# Percentage change between versions for each ROI, for each subject

ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

ROIs_bil = [
    "G_and_S_frontomargin",
    "G_and_S_occipital_inf",
    "G_and_S_paracentral",
    "G_and_S_subcentral",
    "G_and_S_transv_frontopol",
    "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral",
    "G_cuneus",
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
    "G_Ins_lg_and_S_cent_ins",
    "G_insular_short",
    "G_occipital_middle",
    "G_occipital_sup",
    "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual",
    "G_oc-temp_med-Parahip",
    "G_orbital",
    "G_pariet_inf-Angular",
    "G_pariet_inf-Supramar",
    "G_parietal_sup",
    "G_postcentral",
    "G_precentral",
    "G_precuneus",
    "G_rectus",
    "G_subcallosal",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    "G_temporal_inf",
    "G_temporal_middle",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Pole_occipital",
    "Pole_temporal",
    "S_calcarine",
    "S_central",
    "S_cingul-Marginalis",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
    "S_collat_transv_ant",
    "S_collat_transv_post",
    "S_front_inf",
    "S_front_middle",
    "S_front_sup",
    "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans",
    "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal",
    "S_occipital_ant",
    "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral",
    "S_orbital_med-olfact",
    "S_orbital-H_Shaped",
    "S_parieto_occipital",
    "S_pericallosal",
    "S_postcentral",
    "S_precentral-inf-part",
    "S_precentral-sup-part",
    "S_suborbital",
    "S_subparietal",
    "S_temporal_inf",
    "S_temporal_sup",
    "S_temporal_transverse",
]


for roi in ROIs:
    # calculate (Vol 2 - Vol 1) / Vol 1
    aseg_table["FS7_6_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS7".format(roi=roi)]
            - aseg_table["{roi}_FS6".format(roi=roi)]
        )
        / aseg_table["{roi}_FS6".format(roi=roi)]
    ).abs()
    aseg_table["FS7_5_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS7".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
        / aseg_table["{roi}_FS5".format(roi=roi)]
    ).abs()
    aseg_table["FS6_5_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS6".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
        / aseg_table["{roi}_FS5".format(roi=roi)]
    ).abs()
    
    aseg_table["FS8_7_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS7".format(roi=roi)]
        )
        / aseg_table["{roi}_FS7".format(roi=roi)]
    ).abs()
    aseg_table["FS8_6_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS6".format(roi=roi)]
        )
        / aseg_table["{roi}_FS6".format(roi=roi)]
    ).abs()
    aseg_table["FS8_5_pct_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
        / aseg_table["{roi}_FS5".format(roi=roi)]
    ).abs()

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        for i in ["vol", "surf", "ct"]:
            # calculate (Vol 2 - Vol 1) / Vol 1
            aseg_table[f"FS7_6_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS6"]
            ).abs()
            aseg_table[f"FS7_5_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS5"]
            ).abs()
            aseg_table[f"FS6_5_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS5"]
            ).abs()
            
            aseg_table[f"FS8_7_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS7"]
            ).abs()
            aseg_table[f"FS8_6_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS6"]
            ).abs()
            aseg_table[f"FS8_5_pct_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
                / aseg_table[f"{roi}_{hemi}_{i}_FS5"]
            ).abs()

# %%
# calculate V1 - V2 diff and take an absolute value
# Raw differences between versions for each ROI, for each subject

for roi in ROIs:
    aseg_table["FS7_6_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS7".format(roi=roi)]
            - aseg_table["{roi}_FS6".format(roi=roi)]
        )
    ).abs()
    aseg_table["FS7_5_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS7".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
    ).abs()
    aseg_table["FS6_5_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS6".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
    ).abs()
    aseg_table["FS8_7_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS7".format(roi=roi)]
        )
    ).abs()
    aseg_table["FS8_6_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS6".format(roi=roi)]
        )
    ).abs()
    aseg_table["FS8_5_diff_{roi}".format(roi=roi)] = (
        (
            aseg_table["{roi}_FS8".format(roi=roi)]
            - aseg_table["{roi}_FS5".format(roi=roi)]
        )
    ).abs()

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        for i in ["vol", "surf", "ct"]:
            aseg_table[f"FS7_6_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                )
            ).abs()
            aseg_table[f"FS7_5_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
            ).abs()
            aseg_table[f"FS6_5_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
            ).abs()
            aseg_table[f"FS8_7_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS7"]
                )
            ).abs()
            aseg_table[f"FS8_6_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS6"]
                )
            ).abs()
            aseg_table[f"FS8_5_diff_{hemi}_{roi}_{i}"] = (
                (
                    aseg_table[f"{roi}_{hemi}_{i}_FS8"]
                    - aseg_table[f"{roi}_{hemi}_{i}_FS5"]
                )
            ).abs()

# %%
# aseg_table.to_csv(out("aseg_table3.csv"), index=False)
aseg_table = pd.read_csv(inp("aseg_table3.csv"))

# %% [markdown]
# ## stats - between version differences (software variability)

# %%
# paired t-test

import numpy as np
import scipy
from scipy import stats


t_paired_76 = {}
t_paired_75 = {}
t_paired_65 = {}
t_paired_87 = {}
t_paired_86 = {}
t_paired_85 = {}

for roi in ROIs:
    (
        t_paired_76["t_{roi}".format(roi=roi)],
        t_paired_76["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS7".format(roi=roi)],
        aseg_table["{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_75["t_{roi}".format(roi=roi)],
        t_paired_75["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS7".format(roi=roi)],
        aseg_table["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_65["t_{roi}".format(roi=roi)],
        t_paired_65["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS6".format(roi=roi)],
        aseg_table["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_87["t_{roi}".format(roi=roi)],
        t_paired_87["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS8".format(roi=roi)],
        aseg_table["{roi}_FS7".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_86["t_{roi}".format(roi=roi)],
        t_paired_86["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS8".format(roi=roi)],
        aseg_table["{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_85["t_{roi}".format(roi=roi)],
        t_paired_85["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        aseg_table["{roi}_FS8".format(roi=roi)],
        aseg_table["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_paired_76["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_76["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_75["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_75["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_65["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_65["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_87["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_87["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_86["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_86["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_85["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_85["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                aseg_table["{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                aseg_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )

# temp = pd.DataFrame.from_dict(t_paired_76, orient = 'index')
# temp.to_csv(out("results_ttest_paired_76.csv"))
# temp = pd.DataFrame.from_dict(t_paired_75, orient = 'index')
# temp.to_csv(out("results_ttest_paired_75.csv"))
# temp = pd.DataFrame.from_dict(t_paired_65, orient = 'index')
# temp.to_csv(out("results_ttest_paired_65.csv"))
# temp = pd.DataFrame.from_dict(t_paired_87, orient = 'index')
# temp.to_csv(out("results_ttest_paired_87.csv"))
# temp = pd.DataFrame.from_dict(t_paired_86, orient = 'index')
# temp.to_csv(out("results_ttest_paired_86.csv"))
# temp = pd.DataFrame.from_dict(t_paired_85, orient = 'index')
# temp.to_csv(out("results_ttest_paired_85.csv"))

# %% [markdown]
# ## stats - between group differences (PD vs HC) in software variability

# %%
download = pd.read_csv(inp("download_data.csv"))
download["PATNO_id"] = (
    "sub-" + download["PATNO"].astype(str)
    + "_ses-" + download["EVENT_ID"]
)
aseg_table = aseg_table.merge(
    download[["PATNO_id", "dx_group"]],
    on="PATNO_id",
    how="left",
)
print(aseg_table["dx_group"].value_counts(dropna=False))


# %%
from scipy.stats import ttest_ind

group1 = aseg_table.loc[aseg_table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]
group2 = aseg_table[aseg_table["dx_group"] == "HC"]

t_groups_76 = {}
t_groups_75 = {}
t_groups_65 = {}
t_groups_87 = {}
t_groups_86 = {}
t_groups_85 = {}

for roi in ROIs:
    (
        t_groups_76["t_{roi}".format(roi=roi)],
        t_groups_76["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS7_6_pct_{roi}".format(roi=roi)],
        group2["FS7_6_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_75["t_{roi}".format(roi=roi)],
        t_groups_75["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS7_5_pct_{roi}".format(roi=roi)],
        group2["FS7_5_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_65["t_{roi}".format(roi=roi)],
        t_groups_65["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS6_5_pct_{roi}".format(roi=roi)],
        group2["FS6_5_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_87["t_{roi}".format(roi=roi)],
        t_groups_87["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS8_7_pct_{roi}".format(roi=roi)],
        group2["FS8_7_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_86["t_{roi}".format(roi=roi)],
        t_groups_86["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS8_6_pct_{roi}".format(roi=roi)],
        group2["FS8_6_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_85["t_{roi}".format(roi=roi)],
        t_groups_85["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS8_5_pct_{roi}".format(roi=roi)],
        group2["FS8_5_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_groups_76["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_76["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS7_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS7_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_75["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_75["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS7_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS7_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_65["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_65["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS6_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS6_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_87["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_87["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS8_7_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS8_7_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_86["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_86["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS8_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS8_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_85["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_85["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS8_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS8_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )


# temp = pd.DataFrame.from_dict(t_groups_76, orient = 'index')
# temp.to_csv(out("results_ttest_groups_76.csv"))
# temp = pd.DataFrame.from_dict(t_groups_75, orient = 'index')
# temp.to_csv(out("results_ttest_groups_75.csv"))
# temp = pd.DataFrame.from_dict(t_groups_65, orient = 'index')
# temp.to_csv(out("results_ttest_groups_65.csv"))
# temp = pd.DataFrame.from_dict(t_groups_87, orient = 'index')
# temp.to_csv(out("results_ttest_groups_87.csv"))
# temp = pd.DataFrame.from_dict(t_groups_86, orient = 'index')
# temp.to_csv(out("results_ttest_groups_86.csv"))
# temp = pd.DataFrame.from_dict(t_groups_85, orient = 'index')
# temp.to_csv(out("results_ttest_groups_85.csv"))

# %% [markdown]
# ## stats - correlation between MRIQC and software variability

# %%
from scipy.stats import pearsonr


def correlation(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    [corr_r, corr_p] = pearsonr(Y, X)
    return [corr_r, corr_p]


# %%
metrics = [
    "cjv",
    "cnr",
    "fber",
    "qi_1",
    "qi_2",
    "rpve_gm",
    "snr_gm",
    "snr_total",
    "snrd_gm",
    "snrd_total",
]

qc_table = df_qc
qc_corr_87 = {}
qc_corr_86 = {}
qc_corr_85 = {}
qc_corr_76 = {}
qc_corr_75 = {}
qc_corr_65 = {}

fs_pairs = [
    ("FS8_7", qc_corr_87),
    ("FS8_6", qc_corr_86),
    ("FS8_5", qc_corr_85),
    ("FS7_6", qc_corr_76),
    ("FS7_5", qc_corr_75),
    ("FS6_5", qc_corr_65),
]

# Bilateral ROIs with hemisphere
for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            for m in metrics:

                for fs_name, corr_dict in fs_pairs:
                    col = f"{fs_name}_pct_{hemi}_{roi}_{i}"

                    qc_table_stat = qc_table.dropna(subset=[col, m])

                    (
                        corr_dict[f"r_{hemi}_{roi}_{i}_{m}"],
                        corr_dict[f"p_{hemi}_{roi}_{i}_{m}"],
                    ) = correlation(
                        qc_table_stat,
                        col,
                        m,
                    )


# Non-hemispheric ROIs
for roi in ROIs:
    for m in metrics:

        for fs_name, corr_dict in fs_pairs:
            col = f"{fs_name}_pct_{roi}"

            qc_table_stat = qc_table.dropna(subset=[col, m])

            (
                corr_dict[f"r_{roi}_{m}"],
                corr_dict[f"p_{roi}_{m}"],
            ) = correlation(
                qc_table_stat,
                col,
                m,
            )
# temp = pd.DataFrame.from_dict(qc_corr_76, orient = 'index')
# temp.to_csv(out("qc_corr_76.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv(out("qc_corr_75.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv(out("qc_corr_65.csv"))

# %%
# correlation for PD PPMI cohort only

metrics = [
    "cjv",
    "cnr",
    "fber",
    "qi_1",
    "qi_2",
    "rpve_gm",
    "snr_gm",
    "snr_total",
    "snrd_gm",
    "snrd_total",
]

qc_table = df_qc.loc[df_qc["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]

qc_corr_76 = {}
qc_corr_75 = {}
qc_corr_65 = {}

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            for m in metrics:
                qc_table_stat = qc_table.dropna(subset=f"FS7_6_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_76[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_76[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS7_6_pct_{hemi}_{roi}_{i}", f"{m}")

                qc_table_stat = qc_table.dropna(subset=f"FS7_5_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_75[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_75[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS7_5_pct_{hemi}_{roi}_{i}", f"{m}")

                qc_table_stat = qc_table.dropna(subset=f"FS6_5_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_65[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_65[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS6_5_pct_{hemi}_{roi}_{i}", f"{m}")

for roi in ROIs:
    for m in metrics:
        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS7_6_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_76[f"r_{roi}_{m}"], qc_corr_76[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS7_6_pct_{roi}", f"{m}"
        )

        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS7_5_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_75[f"r_{roi}_{m}"], qc_corr_75[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS7_5_pct_{roi}", f"{m}"
        )

        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS6_5_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_65[f"r_{roi}_{m}"], qc_corr_65[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS6_5_pct_{roi}", f"{m}"
        )

# temp = pd.DataFrame.from_dict(qc_corr_76, orient = 'index')
# temp.to_csv(out("qc_PD_corr_76.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv(out("qc_PD_corr_75.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv(out("qc_PD_corr_65.csv"))

# %% [markdown]
# ## stats - Sørensen–Dice coefficient variability 

# %%
# build sets of significant results

set_76_vol = {}
set_75_vol = {}
set_65_vol = {}
set_87_vol = {}
set_86_vol = {}
set_85_vol = {}

set_76_surf = {}
set_75_surf = {}
set_65_surf = {}
set_87_surf = {}
set_86_surf = {}
set_85_surf = {}

set_76_ct = {}
set_75_ct = {}
set_65_ct = {}
set_87_ct = {}
set_86_ct = {}
set_85_ct = {}


for roi in ROIs:
    for s, t in [
        ("76", t_paired_76),
        ("75", t_paired_75),
        ("65", t_paired_65),
        ("87", t_paired_87),
        ("86", t_paired_86),
        ("85", t_paired_85),
    ]:
        globals()[f"set_{s}_vol"][f"p_{roi}"] = t[f"p_{roi}"] < (0.05 / 187)


for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:

        for s, t in [
            ("76", t_paired_76),
            ("75", t_paired_75),
            ("65", t_paired_65),
            ("87", t_paired_87),
            ("86", t_paired_86),
            ("85", t_paired_85),
        ]:
            # volume
            globals()[f"set_{s}_vol"][f"{hemi}_{roi}_vol"] = (
                t[f"p_{hemi}_{roi}_vol"] < (0.05 / 187)
            )

            # surface
            globals()[f"set_{s}_surf"][f"{hemi}_{roi}_surf"] = (
                t[f"p_{hemi}_{roi}_surf"] < (0.05 / 148)
            )

            # cortical thickness
            globals()[f"set_{s}_ct"][f"{hemi}_{roi}_ct"] = (
                t[f"p_{hemi}_{roi}_ct"] < (0.05 / 148)
            )


# convert dictionaries to dataframes
for s in ["76", "75", "65", "87", "86", "85"]:
    for metric in ["vol", "surf", "ct"]:
        globals()[f"set_{s}_{metric}"] = pd.DataFrame.from_dict(
            globals()[f"set_{s}_{metric}"],
            orient="index",
        )

        globals()[f"set_{s}_{metric}"] = globals()[f"set_{s}_{metric}"].loc[
            globals()[f"set_{s}_{metric}"][0] == True
        ]

# %%
# calculate intersection between the sets

# FS7 vs FS6, FS7 vs FS5, FS6 vs FS5
set_76_75_vol_inter = set_76_vol.index.intersection(set_75_vol.index)
set_76_65_vol_inter = set_76_vol.index.intersection(set_65_vol.index)
set_75_65_vol_inter = set_75_vol.index.intersection(set_65_vol.index)

set_76_75_surf_inter = set_76_surf.index.intersection(set_75_surf.index)
set_76_65_surf_inter = set_76_surf.index.intersection(set_65_surf.index)
set_75_65_surf_inter = set_75_surf.index.intersection(set_65_surf.index)

set_76_75_ct_inter = set_76_ct.index.intersection(set_75_ct.index)
set_76_65_ct_inter = set_76_ct.index.intersection(set_65_ct.index)
set_75_65_ct_inter = set_75_ct.index.intersection(set_65_ct.index)


# FS8 comparisons

# Volume
set_87_76_vol_inter = set_87_vol.index.intersection(set_76_vol.index)
set_87_75_vol_inter = set_87_vol.index.intersection(set_75_vol.index)
set_87_65_vol_inter = set_87_vol.index.intersection(set_65_vol.index)

set_86_76_vol_inter = set_86_vol.index.intersection(set_76_vol.index)
set_86_75_vol_inter = set_86_vol.index.intersection(set_75_vol.index)
set_86_65_vol_inter = set_86_vol.index.intersection(set_65_vol.index)

set_85_76_vol_inter = set_85_vol.index.intersection(set_76_vol.index)
set_85_75_vol_inter = set_85_vol.index.intersection(set_75_vol.index)
set_85_65_vol_inter = set_85_vol.index.intersection(set_65_vol.index)

set_87_86_vol_inter = set_87_vol.index.intersection(set_86_vol.index)
set_87_85_vol_inter = set_87_vol.index.intersection(set_85_vol.index)
set_86_85_vol_inter = set_86_vol.index.intersection(set_85_vol.index)


# Surface
set_87_76_surf_inter = set_87_surf.index.intersection(set_76_surf.index)
set_87_75_surf_inter = set_87_surf.index.intersection(set_75_surf.index)
set_87_65_surf_inter = set_87_surf.index.intersection(set_65_surf.index)

set_86_76_surf_inter = set_86_surf.index.intersection(set_76_surf.index)
set_86_75_surf_inter = set_86_surf.index.intersection(set_75_surf.index)
set_86_65_surf_inter = set_86_surf.index.intersection(set_65_surf.index)

set_85_76_surf_inter = set_85_surf.index.intersection(set_76_surf.index)
set_85_75_surf_inter = set_85_surf.index.intersection(set_75_surf.index)
set_85_65_surf_inter = set_85_surf.index.intersection(set_65_surf.index)

set_87_86_surf_inter = set_87_surf.index.intersection(set_86_surf.index)
set_87_85_surf_inter = set_87_surf.index.intersection(set_85_surf.index)
set_86_85_surf_inter = set_86_surf.index.intersection(set_85_surf.index)


# Cortical thickness
set_87_76_ct_inter = set_87_ct.index.intersection(set_76_ct.index)
set_87_75_ct_inter = set_87_ct.index.intersection(set_75_ct.index)
set_87_65_ct_inter = set_87_ct.index.intersection(set_65_ct.index)

set_86_76_ct_inter = set_86_ct.index.intersection(set_76_ct.index)
set_86_75_ct_inter = set_86_ct.index.intersection(set_75_ct.index)
set_86_65_ct_inter = set_86_ct.index.intersection(set_65_ct.index)

set_85_76_ct_inter = set_85_ct.index.intersection(set_76_ct.index)
set_85_75_ct_inter = set_85_ct.index.intersection(set_75_ct.index)
set_85_65_ct_inter = set_85_ct.index.intersection(set_65_ct.index)

set_87_86_ct_inter = set_87_ct.index.intersection(set_86_ct.index)
set_87_85_ct_inter = set_87_ct.index.intersection(set_85_ct.index)
set_86_85_ct_inter = set_86_ct.index.intersection(set_85_ct.index)

# %%
# calculate coefficients

# Existing FS7/FS6/FS5 comparisons
dice_76_75_vol = (2 * len(set_76_75_vol_inter)) / (len(set_76_vol) + len(set_75_vol))
dice_76_65_vol = (2 * len(set_76_65_vol_inter)) / (len(set_76_vol) + len(set_65_vol))
dice_75_65_vol = (2 * len(set_75_65_vol_inter)) / (len(set_75_vol) + len(set_65_vol))

dice_76_75_surf = (2 * len(set_76_75_surf_inter)) / (
    len(set_76_surf) + len(set_75_surf)
)
dice_76_65_surf = (2 * len(set_76_65_surf_inter)) / (
    len(set_76_surf) + len(set_65_surf)
)
dice_75_65_surf = (2 * len(set_75_65_surf_inter)) / (
    len(set_75_surf) + len(set_65_surf)
)

dice_76_75_ct = (2 * len(set_76_75_ct_inter)) / (len(set_76_ct) + len(set_75_ct))
dice_76_65_ct = (2 * len(set_76_65_ct_inter)) / (len(set_76_ct) + len(set_65_ct))
dice_75_65_ct = (2 * len(set_75_65_ct_inter)) / (len(set_75_ct) + len(set_65_ct))


# FS8 comparisons

# Volume
dice_87_76_vol = (2 * len(set_87_76_vol_inter)) / (
    len(set_87_vol) + len(set_76_vol)
)
dice_87_75_vol = (2 * len(set_87_75_vol_inter)) / (
    len(set_87_vol) + len(set_75_vol)
)
dice_87_65_vol = (2 * len(set_87_65_vol_inter)) / (
    len(set_87_vol) + len(set_65_vol)
)

dice_86_76_vol = (2 * len(set_86_76_vol_inter)) / (
    len(set_86_vol) + len(set_76_vol)
)
dice_86_75_vol = (2 * len(set_86_75_vol_inter)) / (
    len(set_86_vol) + len(set_75_vol)
)
dice_86_65_vol = (2 * len(set_86_65_vol_inter)) / (
    len(set_86_vol) + len(set_65_vol)
)

dice_85_76_vol = (2 * len(set_85_76_vol_inter)) / (
    len(set_85_vol) + len(set_76_vol)
)
dice_85_75_vol = (2 * len(set_85_75_vol_inter)) / (
    len(set_85_vol) + len(set_75_vol)
)
dice_85_65_vol = (2 * len(set_85_65_vol_inter)) / (
    len(set_85_vol) + len(set_65_vol)
)

dice_87_86_vol = (2 * len(set_87_86_vol_inter)) / (
    len(set_87_vol) + len(set_86_vol)
)
dice_87_85_vol = (2 * len(set_87_85_vol_inter)) / (
    len(set_87_vol) + len(set_85_vol)
)
dice_86_85_vol = (2 * len(set_86_85_vol_inter)) / (
    len(set_86_vol) + len(set_85_vol)
)


# Surface
dice_87_76_surf = (2 * len(set_87_76_surf_inter)) / (
    len(set_87_surf) + len(set_76_surf)
)
dice_87_75_surf = (2 * len(set_87_75_surf_inter)) / (
    len(set_87_surf) + len(set_75_surf)
)
dice_87_65_surf = (2 * len(set_87_65_surf_inter)) / (
    len(set_87_surf) + len(set_65_surf)
)

dice_86_76_surf = (2 * len(set_86_76_surf_inter)) / (
    len(set_86_surf) + len(set_76_surf)
)
dice_86_75_surf = (2 * len(set_86_75_surf_inter)) / (
    len(set_86_surf) + len(set_75_surf)
)
dice_86_65_surf = (2 * len(set_86_65_surf_inter)) / (
    len(set_86_surf) + len(set_65_surf)
)

dice_85_76_surf = (2 * len(set_85_76_surf_inter)) / (
    len(set_85_surf) + len(set_76_surf)
)
dice_85_75_surf = (2 * len(set_85_75_surf_inter)) / (
    len(set_85_surf) + len(set_75_surf)
)
dice_85_65_surf = (2 * len(set_85_65_surf_inter)) / (
    len(set_85_surf) + len(set_65_surf)
)

dice_87_86_surf = (2 * len(set_87_86_surf_inter)) / (
    len(set_87_surf) + len(set_86_surf)
)
dice_87_85_surf = (2 * len(set_87_85_surf_inter)) / (
    len(set_87_surf) + len(set_85_surf)
)
dice_86_85_surf = (2 * len(set_86_85_surf_inter)) / (
    len(set_86_surf) + len(set_85_surf)
)


# Cortical thickness
dice_87_76_ct = (2 * len(set_87_76_ct_inter)) / (
    len(set_87_ct) + len(set_76_ct)
)
dice_87_75_ct = (2 * len(set_87_75_ct_inter)) / (
    len(set_87_ct) + len(set_75_ct)
)
dice_87_65_ct = (2 * len(set_87_65_ct_inter)) / (
    len(set_87_ct) + len(set_65_ct)
)

dice_86_76_ct = (2 * len(set_86_76_ct_inter)) / (
    len(set_86_ct) + len(set_76_ct)
)
dice_86_75_ct = (2 * len(set_86_75_ct_inter)) / (
    len(set_86_ct) + len(set_75_ct)
)
dice_86_65_ct = (2 * len(set_86_65_ct_inter)) / (
    len(set_86_ct) + len(set_65_ct)
)

dice_85_76_ct = (2 * len(set_85_76_ct_inter)) / (
    len(set_85_ct) + len(set_76_ct)
)
dice_85_75_ct = (2 * len(set_85_75_ct_inter)) / (
    len(set_85_ct) + len(set_75_ct)
)
dice_85_65_ct = (2 * len(set_85_65_ct_inter)) / (
    len(set_85_ct) + len(set_65_ct)
)

dice_87_86_ct = (2 * len(set_87_86_ct_inter)) / (
    len(set_87_ct) + len(set_86_ct)
)
dice_87_85_ct = (2 * len(set_87_85_ct_inter)) / (
    len(set_87_ct) + len(set_85_ct)
)
dice_86_85_ct = (2 * len(set_86_85_ct_inter)) / (
    len(set_86_ct) + len(set_85_ct)
)

# %%
print(
    str(len(set_76_vol) / 187)
    + " % regions differed in volume between FS7 and FS6\n"
    + str(len(set_75_vol) / 187)
    + " % regions differed in volume between FS7 and FS5\n"
    + str(len(set_65_vol) / 187)
    + " % regions differed in volume between FS6 and FS5\n"
    + str(len(set_87_vol) / 187)
    + " % regions differed in volume between FS8 and FS7\n"
    + str(len(set_86_vol) / 187)
    + " % regions differed in volume between FS8 and FS6\n"
    + str(len(set_85_vol) / 187)
    + " % regions differed in volume between FS8 and FS5\n"
    + str(len(set_76_surf) / 148)
    + " % regions differed in surface between FS7 and FS6\n"
    + str(len(set_75_surf) / 148)
    + " % regions differed in surface between FS7 and FS5\n"
    + str(len(set_65_surf) / 148)
    + " % regions differed in surface between FS6 and FS5\n"
    + str(len(set_87_surf) / 148)
    + " % regions differed in surface between FS8 and FS7\n"
    + str(len(set_86_surf) / 148)
    + " % regions differed in surface between FS8 and FS6\n"
    + str(len(set_85_surf) / 148)
    + " % regions differed in surface between FS8 and FS5\n"
    + str(len(set_76_ct) / 148)
    + " % regions differed in thickness between FS7 and FS6\n"
    + str(len(set_75_ct) / 148)
    + " % regions differed in thickness between FS7 and FS5\n"
    + str(len(set_65_ct) / 148)
    + " % regions differed in thickness between FS6 and FS5\n"
    + str(len(set_87_ct) / 148)
    + " % regions differed in thickness between FS8 and FS7\n"
    + str(len(set_86_ct) / 148)
    + " % regions differed in thickness between FS8 and FS6\n"
    + str(len(set_85_ct) / 148)
    + " % regions differed in thickness between FS8 and FS5\n"
)

# %%
print(
    "The Sørensen–Dice coefficients for software variability are:\n"
    "Volume 76 vs 75 = "
    + str(dice_76_75_vol)
    + "\n Volume 76 vs 65 = "
    + str(dice_76_65_vol)
    + "\n Volume 75 vs 65 = "
    + str(dice_75_65_vol)
    + "\n Volume 87 vs 76 = "
    + str(dice_87_76_vol)
    + "\n Volume 87 vs 75 = "
    + str(dice_87_75_vol)
    + "\n Volume 87 vs 65 = "
    + str(dice_87_65_vol)
    + "\n Volume 86 vs 76 = "
    + str(dice_86_76_vol)
    + "\n Volume 86 vs 75 = "
    + str(dice_86_75_vol)
    + "\n Volume 86 vs 65 = "
    + str(dice_86_65_vol)
    + "\n Volume 85 vs 76 = "
    + str(dice_85_76_vol)
    + "\n Volume 85 vs 75 = "
    + str(dice_85_75_vol)
    + "\n Volume 85 vs 65 = "
    + str(dice_85_65_vol)
    + "\n Surface 76 vs 75 = "
    + str(dice_76_75_surf)
    + "\n Surface 76 vs 65 = "
    + str(dice_76_65_surf)
    + "\n Surface 75 vs 65 = "
    + str(dice_75_65_surf)
    + "\n Surface 87 vs 76 = "
    + str(dice_87_76_surf)
    + "\n Surface 87 vs 75 = "
    + str(dice_87_75_surf)
    + "\n Surface 87 vs 65 = "
    + str(dice_87_65_surf)
    + "\n Surface 86 vs 76 = "
    + str(dice_86_76_surf)
    + "\n Surface 86 vs 75 = "
    + str(dice_86_75_surf)
    + "\n Surface 86 vs 65 = "
    + str(dice_86_65_surf)
    + "\n Surface 85 vs 76 = "
    + str(dice_85_76_surf)
    + "\n Surface 85 vs 75 = "
    + str(dice_85_75_surf)
    + "\n Surface 85 vs 65 = "
    + str(dice_85_65_surf)
    + "\n Thickness 76 vs 75 = "
    + str(dice_76_75_ct)
    + "\n Thickness 76 vs 65 = "
    + str(dice_76_65_ct)
    + "\n Thickness 75 vs 65 = "
    + str(dice_75_65_ct)
    + "\n Thickness 87 vs 76 = "
    + str(dice_87_76_ct)
    + "\n Thickness 87 vs 75 = "
    + str(dice_87_75_ct)
    + "\n Thickness 87 vs 65 = "
    + str(dice_87_65_ct)
    + "\n Thickness 86 vs 76 = "
    + str(dice_86_76_ct)
    + "\n Thickness 86 vs 75 = "
    + str(dice_86_75_ct)
    + "\n Thickness 86 vs 65 = "
    + str(dice_86_65_ct)
    + "\n Thickness 85 vs 76 = "
    + str(dice_85_76_ct)
    + "\n Thickness 85 vs 75 = "
    + str(dice_85_75_ct)
    + "\n Thickness 85 vs 65 = "
    + str(dice_85_65_ct)
)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comparisons = [
    "FS7-6",
    "FS7-5",
    "FS6-5",
    "FS8-7",
    "FS8-6",
    "FS8-5",
]

# Volume matrix
dice_volume = pd.DataFrame(
    [
        [1, dice_76_75_vol, dice_76_65_vol, dice_87_76_vol, dice_86_76_vol, dice_85_76_vol],
        [dice_76_75_vol, 1, dice_75_65_vol, dice_87_75_vol, dice_86_75_vol, dice_85_75_vol],
        [dice_76_65_vol, dice_75_65_vol, 1, dice_87_65_vol, dice_86_65_vol, dice_85_65_vol],
        [dice_87_76_vol, dice_87_75_vol, dice_87_65_vol, 1, dice_87_86_vol, dice_87_85_vol],
        [dice_86_76_vol, dice_86_75_vol, dice_86_65_vol, dice_87_86_vol, 1, dice_86_85_vol],
        [dice_85_76_vol, dice_85_75_vol, dice_85_65_vol, dice_87_85_vol, dice_86_85_vol, 1],
    ],
    index=comparisons,
    columns=comparisons,
)


# Surface matrix
dice_surface = pd.DataFrame(
    [
        [1, dice_76_75_surf, dice_76_65_surf, dice_87_76_surf, dice_86_76_surf, dice_85_76_surf],
        [dice_76_75_surf, 1, dice_75_65_surf, dice_87_75_surf, dice_86_75_surf, dice_85_75_surf],
        [dice_76_65_surf, dice_75_65_surf, 1, dice_87_65_surf, dice_86_65_surf, dice_85_65_surf],
        [dice_87_76_surf, dice_87_75_surf, dice_87_65_surf, 1, dice_87_86_surf, dice_87_85_surf],
        [dice_86_76_surf, dice_86_75_surf, dice_86_65_surf, dice_87_86_surf, 1, dice_86_85_surf],
        [dice_85_76_surf, dice_85_75_surf, dice_85_65_surf, dice_87_85_surf, dice_86_85_surf, 1],
    ],
    index=comparisons,
    columns=comparisons,
)


# Cortical thickness matrix
dice_ct = pd.DataFrame(
    [
        [1, dice_76_75_ct, dice_76_65_ct, dice_87_76_ct, dice_86_76_ct, dice_85_76_ct],
        [dice_76_75_ct, 1, dice_75_65_ct, dice_87_75_ct, dice_86_75_ct, dice_85_75_ct],
        [dice_76_65_ct, dice_75_65_ct, 1, dice_87_65_ct, dice_86_65_ct, dice_85_65_ct],
        [dice_87_76_ct, dice_87_75_ct, dice_87_65_ct, 1, dice_87_86_ct, dice_87_85_ct],
        [dice_86_76_ct, dice_86_75_ct, dice_86_65_ct, dice_87_86_ct, 1, dice_86_85_ct],
        [dice_85_76_ct, dice_85_75_ct, dice_85_65_ct, dice_87_85_ct, dice_86_85_ct, 1],
    ],
    index=comparisons,
    columns=comparisons,
)


# Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, data, title in zip(
    axes,
    [dice_volume, dice_surface, dice_ct],
    ["Volume", "Surface Area", "Cortical Thickness"],
):
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cmap="viridis",
        cbar=True,
    )
    ax.set_title(f"Sørensen–Dice Similarity: {title}")
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()

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
# ## stats - software variability separately in HC and PD cohorts

# %%
selection_pd = aseg_table.loc[aseg_table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]
selection_hc = aseg_table[aseg_table["dx_group"] == "HC"]

# %%
t_paired_76_hc = {}
t_paired_75_hc = {}
t_paired_65_hc = {}
t_paired_76_pd = {}
t_paired_75_pd = {}
t_paired_65_pd = {}
t_paired_87_hc = {}
t_paired_86_hc = {}
t_paired_85_hc = {}
t_paired_87_pd = {}
t_paired_86_pd = {}
t_paired_85_pd = {}

# PD
for roi in ROIs:
    (
        t_paired_76_pd["t_{roi}".format(roi=roi)],
        t_paired_76_pd["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd["{roi}_FS7".format(roi=roi)],
        selection_pd["{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_75_pd["t_{roi}".format(roi=roi)],
        t_paired_75_pd["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd["{roi}_FS7".format(roi=roi)],
        selection_pd["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_65_pd["t_{roi}".format(roi=roi)],
        t_paired_65_pd["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd["{roi}_FS6".format(roi=roi)],
        selection_pd["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    
    (
        t_paired_87_pd[f"t_{roi}".format(roi=roi)],
        t_paired_87_pd[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd[f"{roi}_FS8".format(roi=roi)],
        selection_pd[f"{roi}_FS7".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_86_pd[f"t_{roi}".format(roi=roi)],
        t_paired_86_pd[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd[f"{roi}_FS8".format(roi=roi)],
        selection_pd[f"{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_85_pd[f"t_{roi}".format(roi=roi)],
        t_paired_85_pd[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_pd[f"{roi}_FS8".format(roi=roi)],
        selection_pd[f"{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_paired_76_pd["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_76_pd["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                selection_pd["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_75_pd["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_75_pd["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                selection_pd["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_65_pd["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_65_pd["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                selection_pd["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            
            (
                t_paired_87_pd[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_87_pd[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_pd[f"{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_86_pd[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_86_pd[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_pd[f"{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_85_pd[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_85_pd[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_pd[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_pd[f"{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )

# HC
for roi in ROIs:
    (
        t_paired_76_hc["t_{roi}".format(roi=roi)],
        t_paired_76_hc["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc["{roi}_FS7".format(roi=roi)],
        selection_hc["{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_75_hc["t_{roi}".format(roi=roi)],
        t_paired_75_hc["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc["{roi}_FS7".format(roi=roi)],
        selection_hc["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_65_hc["t_{roi}".format(roi=roi)],
        t_paired_65_hc["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc["{roi}_FS6".format(roi=roi)],
        selection_hc["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_87_hc[f"t_{roi}".format(roi=roi)],
        t_paired_87_hc[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc[f"{roi}_FS8".format(roi=roi)],
        selection_hc[f"{roi}_FS7".format(roi=roi)],
        nan_policy="omit",
    )

    (
        t_paired_86_hc[f"t_{roi}".format(roi=roi)],
        t_paired_86_hc[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc[f"{roi}_FS8".format(roi=roi)],
        selection_hc[f"{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )

    (
        t_paired_85_hc[f"t_{roi}".format(roi=roi)],
        t_paired_85_hc[f"p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        selection_hc[f"{roi}_FS8".format(roi=roi)],
        selection_hc[f"{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_paired_76_hc["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_76_hc["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                selection_hc["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_75_hc["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_75_hc["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                selection_hc["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_65_hc["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_65_hc["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                selection_hc["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            
            (
                t_paired_87_hc[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_87_hc[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_hc[f"{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )

            (
                t_paired_86_hc[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_86_hc[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_hc[f"{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )

            (
                t_paired_85_hc[f"t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_85_hc[f"p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                selection_hc[f"{roi}_{hemi}_{i}_FS8".format(roi=roi, hemi=hemi, i=i)],
                selection_hc[f"{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )



# temp = pd.DataFrame.from_dict(t_paired_76_pd, orient = 'index')
# temp.to_csv(out("results_ttest_paired_76_pd.csv"))
# temp = pd.DataFrame.from_dict(t_paired_75_pd, orient = 'index')
# temp.to_csv(out("results_ttest_paired_75_pd.csv"))
# temp = pd.DataFrame.from_dict(t_paired_65_pd, orient = 'index')
# temp.to_csv(out("results_ttest_paired_65_pd.csv"))

# temp = pd.DataFrame.from_dict(t_paired_76_hc, orient = 'index')
# temp.to_csv(out("results_ttest_paired_76_hc.csv"))
# temp = pd.DataFrame.from_dict(t_paired_75_hc, orient = 'index')
# temp.to_csv(out("results_ttest_paired_75_hc.csv"))
# temp = pd.DataFrame.from_dict(t_paired_65_hc, orient = 'index')
# temp.to_csv(out("results_ttest_paired_65_hc.csv"))

# %% [markdown]
# # Clinical analyses
# ## subcortical volume analysis

# %%
stage_two  = pd.read_csv(inp("stage_two_clinical.csv"))
aseg_table2 = stage_two
# aseg_table2 = pd.read_csv(inp("download_data.csv")) 

aseg_table2["first_visit"] = (
    "sub-" + aseg_table2["PATNO"].astype(str) + "_ses-" + aseg_table2["EVENT_ID"]
)
aseg_table2["second_visit"] = (
    "sub-" + aseg_table2["PATNO"].astype(str) + "_ses-" + aseg_table2["NEXT_VISIT"]
)


aseg_table2["PATNO_base"] = "sub-" + aseg_table2["PATNO"].astype(str) + "_base"
aseg_table2["first_id"] = (
    aseg_table2["first_visit"] + ".long." + aseg_table2["PATNO_base"]
)
aseg_table2["second_id"] = (
    aseg_table2["second_visit"] + ".long." + aseg_table2["PATNO_base"]
)

# select only PD-non-MCI patients and HC
aseg_table2 = aseg_table2.loc[aseg_table2["dx_group"].isin(["HC", "PD-non-MCI"])]



# aseg_table2 = aseg_table2[~aseg_table2["first_id"].isin(to_drop)]
# aseg_table2 = aseg_table2[~aseg_table2["second_id"].isin(to_drop)]

# %%
import os

valid_subjects = []
missing_files = {}

for _, row in aseg_table2.iterrows():

    first_id = row["first_id"]
    second_id = row["second_id"]

    missing = []

    # Check both first_id and second_id
    for subj in [first_id, second_id]:

        for version in ["FS5", "FS6", "FS7", "FS8"]:

            file = f"{DATA}/{version}/{subj}/stats/aseg.stats"

            if not os.path.exists(file):
                missing.append(file)

    # Keep row only if ALL files for BOTH visits exist
    if len(missing) == 0:
        valid_subjects.append(True)
    else:
        valid_subjects.append(False)
        missing_files[first_id] = {
            "second_id": second_id,
            "missing": missing
        }

# Keep only rows where both first_id and second_id have all 4 versions
aseg_table2 = aseg_table2.loc[valid_subjects].copy()

print(f"Valid rows: {len(aseg_table2)}")
print(f"Rows dropped: {len(missing_files)}")

print("Dropped rows:")
for first_id, info in missing_files.items():
    print(f"  {first_id} -> {info['second_id']}: {info['missing']}")

# Verify that every remaining row has all files
for _, row in aseg_table2.iterrows():

    for subj in [row["first_id"], row["second_id"]]:

        for version in ["FS5", "FS6", "FS7", "FS8"]:

            file = f"{DATA}/{version}/{subj}/stats/aseg.stats"

            assert os.path.exists(file), f"Missing: {file}"

print("✓ All remaining rows have FS5, FS6, FS7, and FS8 for both first_id and second_id.")

# %%
# extract subcortical volumes

ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]


# extract first visit
for subj in aseg_table2["first_id"]:
    for version in ["FS5", "FS6", "FS7", "FS8"]:
        # extract TIV
        file = str(DATA) + "/{version}/{subj}/stats/aseg.stats".format(
            subj=subj, version=version
        )
        with open(file, "r") as fp:
            # read all lines in a list
            lines = fp.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find("Estimated Total Intracranial Volume") != -1:
                    aseg_table2.loc[
                        aseg_table2["first_id"] == subj,
                        "TIV_{version}_first".format(version=version),
                    ] = float(line.split(",")[3])

                    # aseg_table["TIV{ses}".format(ses = session)] = float(out)

        # extract ROIs volume
        for roi in ROIs:

            file = str(DATA) + "/{version}/{subj}/stats/aseg.stats".format(
                subj=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table2.loc[
                            aseg_table2["first_id"] == subj,
                            roi + "_" + version + "_first",
                        ] = float(line.split()[3])

# extract second visit
for subj in aseg_table2["second_id"]:
    for version in ["FS5", "FS6", "FS7", "FS8"]:
        # extract TIV
        file = str(DATA) + "/{version}/{subj}/stats/aseg.stats".format(
            subj=subj, version=version
        )
        with open(file, "r") as fp:
            # read all lines in a list
            lines = fp.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find("Estimated Total Intracranial Volume") != -1:
                    aseg_table2.loc[
                        aseg_table2["second_id"] == subj,
                        "TIV_{version}_second".format(version=version),
                    ] = float(line.split(",")[3])

                    # aseg_table["TIV{ses}".format(ses = session)] = float(out)

        # extract ROIs volume
        for roi in ROIs:

            file = str(DATA) + "/{version}/{subj}/stats/aseg.stats".format(
                subj=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table2.loc[
                            aseg_table2["second_id"] == subj,
                            roi + "_" + version + "_second",
                        ] = float(line.split()[3])

# %%
# this calculates the change in subcortical volume

for roi_bil in ROIs:
    for version in ["FS5", "FS6", "FS7", "FS8"]:
        #         # calculate change in subcortical volume
        #         aseg_table2[roi_bil + version + "_change"] = (
        #             aseg_table2[roi_bil + version + "_second"] - aseg_table2[roi_bil + version + "_first"]
        #         )

        # calculate percentage of change
        aseg_table2[roi_bil + "_" + version + "_change_pct"] = (
            (
                aseg_table2[roi_bil + "_" + version + "_second"]
                - aseg_table2[roi_bil + "_" + version + "_first"]
            )
            / aseg_table2[roi_bil + "_" + version + "_first"]
        ) * 100

# %%
ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-choroid-plexus",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

# %% [markdown]
# ## stats - clinical analysis (subcortical)

# %%
from pingouin import partial_corr

ROI_PDnonMCI = aseg_table2.loc[aseg_table2["dx_group"] == "PD-non-MCI"]

corr_cov = {}

# baseline
for roi_bil in ROIs:
    for version in ["FS5", "FS6", "FS7", "FS8"]:

        x_col = "{roi}_{ver}_first".format(
            roi=roi_bil,
            ver=version
        )

        corr_temp = partial_corr(
            data=ROI_PDnonMCI,
            x=x_col,
            y="NP3TOT",
            covar=["AGE_AT_VISIT", "SEX"],
            method="pearson",
        )

        corr_cov["corr_n_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["n"]["pearson"]
        corr_cov["corr_r_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["r"]["pearson"]
        corr_cov["corr_ci_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["CI95"]["pearson"]
        corr_cov["corr_p_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["p_val"]["pearson"]

excluded_longitudinal = {
    ("FS8", "Left-vessel"),
    ("FS8", "Left-choroid-plexus"),
    ("FS8", "Right-choroid-plexus"),
    ("FS8", "Optic-Chiasm"),
}

# longitudinal
for roi_bil in ROIs:
    for version in ["FS5", "FS6", "FS7", "FS8"]:

        if (version, roi_bil) in excluded_longitudinal:
            continue
        
        x_col = "{roi}_{ver}_change_pct".format(
            roi=roi_bil,
            ver=version
        )

        corr_temp = partial_corr(
            data=ROI_PDnonMCI,
            x=x_col,
            y="NP3TOT_change",
            covar=["AGE_AT_VISIT", "SEX", "durationT2_T1_y"],
            method="pearson",
        )

        corr_cov["corr_n_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["n"]["pearson"]
        corr_cov["corr_r_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["r"]["pearson"]
        corr_cov["corr_ci_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["CI95"]["pearson"]
        corr_cov["corr_p_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["p_val"]["pearson"]


df_corr_cov = pd.DataFrame.from_dict(corr_cov, orient="index")
df_corr_cov.to_csv(out("results_corr_cov_PDnonMCI.csv"))

# %%
## group analysis with covariates

from pingouin import ancova as pg_ancova

ROI_PDnonMCI = aseg_table2.loc[aseg_table2["dx_group"] == "PD-non-MCI"]
ROI_HC = aseg_table2.loc[aseg_table2["dx_group"] == "HC"]
HC_PDMCI = pd.concat([ROI_HC, ROI_PDnonMCI], ignore_index=True)

ancova = {}

# baseline
for roi in ROIs:
    for ver in ["FS5", "FS6", "FS7", "FS8"]:
        temp = pg_ancova(
            data=HC_PDMCI,
            dv="{roi}_{ver}_first".format(roi=roi, ver=ver),
            between="dx_group",
            covar=["AGE_AT_VISIT", "SEX"],
        )
        ancova["ancova_F_" + ver + "_" + roi + "_base"] = temp["F"][0]
        ancova["ancova_p_" + ver + "_" + roi + "_base"] = temp["p_unc"][0]

# longitudinal
for roi in ROIs:
    for ver in ["FS5", "FS6", "FS7", "FS8"]:

        if (ver, roi) in excluded_longitudinal:
            continue

        temp = pg_ancova(
            data=HC_PDMCI,
            dv="{roi}_{ver}_change_pct".format(roi=roi, ver=ver),
            between="dx_group",
            covar=["AGE_AT_VISIT", "SEX", "durationT2_T1_y"],
        )
        ancova["ancova_F_" + ver + "_" + roi + "_long"] = temp["F"][0]
        ancova["ancova_p_" + ver + "_" + roi + "_long"] = temp["p_unc"][0]

df_ancova = pd.DataFrame.from_dict(ancova, orient="index")
df_ancova.to_csv(out("results_ancova_subcortical_cov.csv"))

# %% [markdown]
# ## Clinical results (subcortical)

# %%
## partial correlation

from rich.console import Console
from rich.table import Table

table = Table(
    title="Partial correlation between the UPDRS score and subcortical volumes at baseline"
)

table.add_column("Subcortical segmentations", justify="Left")
table.add_column("FS 5")
table.add_column("")
table.add_column("FS 6")
table.add_column("")
table.add_column("FS 7")
table.add_column("")
table.add_column("FS 8")
table.add_column("")

table.add_row("", "r", "p", "r", "p", "r", "p", "r", "p")

rois_sub = [
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Left-Accumbens-area",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
]

for rois in rois_sub:

    table.add_row(
        rois,
        str(
            (corr_cov["corr_r_FS5_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS5_{rois}_UPDRS_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS6_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS6_{rois}_UPDRS_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS7_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS7_{rois}_UPDRS_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS8_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS8_{rois}_UPDRS_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
    )


console = Console()
console.print(table)

# %%
from rich.console import Console
from rich.table import Table

table = Table(
    title="Longitudinal partial correlation between the change in UPDRS score and rate of change in subcortical volumes"
)

table.add_column("Subcortical segmentations", justify="Left")
table.add_column("FS 5")
table.add_column("")
table.add_column("FS 6")
table.add_column("")
table.add_column("FS 7")
table.add_column("")
table.add_column("FS 8")
table.add_column("")

table.add_row("", "r", "p", "r", "p", "r", "p", "r", "p")

rois_sub = [
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Left-Accumbens-area",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
]

for rois in rois_sub:

    table.add_row(
        rois,
        str(
            (corr_cov["corr_r_FS5_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS5_{rois}_UPDRS_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS6_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS6_{rois}_UPDRS_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS7_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS7_{rois}_UPDRS_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str(
            (corr_cov["corr_r_FS8_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        fmt_p(corr_cov["corr_p_FS8_{rois}_UPDRS_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
    )


console = Console()
console.print(table)

# %%
## ancova

from rich.console import Console
from rich.table import Table

table = Table(title="Group differences (HC vs PD) in subcortical volumes at baseline")

table.add_column("Subcortical segmentations", justify="Left")
table.add_column("FS 5")
table.add_column("")
table.add_column("FS 6")
table.add_column("")
table.add_column("FS 7")
table.add_column("")
table.add_column("FS 8")
table.add_column("")

table.add_row("", "F", "p", "F", "p", "F", "p", "F", "p")

rois_sub = [
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Left-Accumbens-area",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
]

for rois in rois_sub:

    table.add_row(
        rois,
        str((ancova["ancova_F_FS5_{rois}_base".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS5_{rois}_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS6_{rois}_base".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS6_{rois}_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS7_{rois}_base".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS7_{rois}_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS8_{rois}_base".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS8_{rois}_base".format(rois=rois)], n_tests=N_SUBCORTICAL),
    )


console = Console()
console.print(table)

# %%
from rich.console import Console
from rich.table import Table

table = Table(
    title="Longtudinal group differences (HC vs PD) in the rate of change in subcortical volumes"
)

table.add_column("Subcortical segmentations", justify="Left")
table.add_column("FS 5")
table.add_column("")
table.add_column("FS 6")
table.add_column("")
table.add_column("FS 7")
table.add_column("")
table.add_column("FS 8")
table.add_column("")

table.add_row("", "F", "p", "F", "p", "F", "p", "F", "p")

rois_sub = [
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "Left-Hippocampus",
    "Left-Amygdala",
    "Left-Accumbens-area",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
]

for rois in rois_sub:

    table.add_row(
        rois,
        str((ancova["ancova_F_FS5_{rois}_long".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS5_{rois}_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS6_{rois}_long".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS6_{rois}_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS7_{rois}_long".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS7_{rois}_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
        str((ancova["ancova_F_FS8_{rois}_long".format(rois=rois)]).round(decimals=3)),
        fmt_p(ancova["ancova_p_FS8_{rois}_long".format(rois=rois)], n_tests=N_SUBCORTICAL),
    )


console = Console()
console.print(table)

# %% [markdown]
# # Data visualisation
#
# ## Computational analyses
# Dice coefficients

# %%
aseg_table = pd.read_csv(inp("aseg_table4.csv"))
dice_table = aseg_table.filter(like="dice_")
mean = dice_table.mean()
sd = dice_table.std(numeric_only=True)

dice_results = {}
dice_results["mean"] = mean
dice_results["sd"] = sd
dice_results = pd.DataFrame.from_dict(dice_results, orient="index")
dice_results = dice_results.transpose()
# dice_results.to_csv(out("dice_results.csv"))

# %%
dice_table_76_subR = dice_table.filter(like="dice_FS76_Right")
dice_table_76_subL = dice_table.filter(like="dice_FS76_Left")
dice_table_76_ctx_lh = dice_table.filter(like="dice_FS76_ctx_lh")
dice_table_76_ctx_rh = dice_table.filter(like="dice_FS76_ctx_rh")

dice_table_75_subR = dice_table.filter(like="dice_FS75_Right")
dice_table_75_subL = dice_table.filter(like="dice_FS75_Left")
dice_table_75_ctx_lh = dice_table.filter(like="dice_FS75_ctx_lh")
dice_table_75_ctx_rh = dice_table.filter(like="dice_FS75_ctx_rh")

dice_table_65_subR = dice_table.filter(like="dice_FS65_Right")
dice_table_65_subL = dice_table.filter(like="dice_FS65_Left")
dice_table_65_ctx_rh = dice_table.filter(like="dice_FS65_ctx_rh")
dice_table_65_ctx_lh = dice_table.filter(like="dice_FS65_ctx_lh")

dice_table_87_subR = dice_table.filter(like="dice_FS87_Right")
dice_table_87_subL = dice_table.filter(like="dice_FS87_Left")
dice_table_87_ctx_lh = dice_table.filter(like="dice_FS87_ctx_lh")
dice_table_87_ctx_rh = dice_table.filter(like="dice_FS87_ctx_rh")

dice_table_86_subR = dice_table.filter(like="dice_FS86_Right")
dice_table_86_subL = dice_table.filter(like="dice_FS86_Left")
dice_table_86_ctx_lh = dice_table.filter(like="dice_FS86_ctx_lh")
dice_table_86_ctx_rh = dice_table.filter(like="dice_FS86_ctx_rh")

dice_table_85_subR = dice_table.filter(like="dice_FS85_Right")
dice_table_85_subL = dice_table.filter(like="dice_FS85_Left")
dice_table_85_ctx_lh = dice_table.filter(like="dice_FS85_ctx_lh")
dice_table_85_ctx_rh = dice_table.filter(like="dice_FS85_ctx_rh")


dice_table_65_subR.columns = dice_table_65_subR.columns.str.replace(
    "^dice_FS65_Right-", ""
)
dice_table_65_subL.columns = dice_table_65_subL.columns.str.replace(
    "^dice_FS65_Left-", ""
)
dice_table_65_ctx_rh.columns = dice_table_65_ctx_rh.columns.str.replace(
    "^dice_FS65_ctx_rh_", ""
)
dice_table_65_ctx_lh.columns = dice_table_65_ctx_lh.columns.str.replace(
    "^dice_FS65_ctx_lh_", ""
)

dice_table_75_subR.columns = dice_table_75_subR.columns.str.replace(
    "^dice_FS75_Right-", ""
)
dice_table_75_subL.columns = dice_table_75_subL.columns.str.replace(
    "^dice_FS75_Left-", ""
)
dice_table_75_ctx_rh.columns = dice_table_75_ctx_rh.columns.str.replace(
    "^dice_FS75_ctx_rh_", ""
)
dice_table_75_ctx_lh.columns = dice_table_75_ctx_lh.columns.str.replace(
    "^dice_FS75_ctx_lh_", ""
)

dice_table_76_subR.columns = dice_table_76_subR.columns.str.replace(
    "^dice_FS76_Right-", ""
)
dice_table_76_subL.columns = dice_table_76_subL.columns.str.replace(
    "^dice_FS76_Left-", ""
)
dice_table_76_ctx_rh.columns = dice_table_76_ctx_rh.columns.str.replace(
    "^dice_FS76_ctx_rh_", ""
)
dice_table_76_ctx_lh.columns = dice_table_76_ctx_lh.columns.str.replace(
    "^dice_FS76_ctx_lh_", ""
)

for df, prefix in [
    (dice_table_87_subR, "^dice_FS87_Right-"),
    (dice_table_87_subL, "^dice_FS87_Left-"),
    (dice_table_87_ctx_rh, "^dice_FS87_ctx_rh_"),
    (dice_table_87_ctx_lh, "^dice_FS87_ctx_lh_"),

    (dice_table_86_subR, "^dice_FS86_Right-"),
    (dice_table_86_subL, "^dice_FS86_Left-"),
    (dice_table_86_ctx_rh, "^dice_FS86_ctx_rh_"),
    (dice_table_86_ctx_lh, "^dice_FS86_ctx_lh_"),

    (dice_table_85_subR, "^dice_FS85_Right-"),
    (dice_table_85_subL, "^dice_FS85_Left-"),
    (dice_table_85_ctx_rh, "^dice_FS85_ctx_rh_"),
    (dice_table_85_ctx_lh, "^dice_FS85_ctx_lh_"),
]:
    df.columns = df.columns.str.replace(prefix, "")



dice_table_76_rh = pd.concat([dice_table_76_subR, dice_table_76_ctx_rh], axis=1)
dice_table_76_lh = pd.concat([dice_table_76_subL, dice_table_76_ctx_lh], axis=1)
dice_table_76_rh["hemi"] = "rh"
dice_table_76_lh["hemi"] = "lh"
dice_table_76_rh["group"] = "FS 7 vs 6"
dice_table_76_lh["group"] = "FS 7 vs 6"

dice_table_75_rh = pd.concat([dice_table_75_subR, dice_table_75_ctx_rh], axis=1)
dice_table_75_lh = pd.concat([dice_table_75_subL, dice_table_75_ctx_lh], axis=1)
dice_table_75_rh["hemi"] = "rh"
dice_table_75_lh["hemi"] = "lh"
dice_table_75_rh["group"] = "FS 7 vs 5"
dice_table_75_lh["group"] = "FS 7 vs 5"

dice_table_65_rh = pd.concat([dice_table_65_subR, dice_table_65_ctx_rh], axis=1)
dice_table_65_lh = pd.concat([dice_table_65_subL, dice_table_65_ctx_lh], axis=1)
dice_table_65_rh["hemi"] = "rh"
dice_table_65_lh["hemi"] = "lh"
dice_table_65_rh["group"] = "FS 6 vs 5"
dice_table_65_lh["group"] = "FS 6 vs 5"

dice_table_87_rh = pd.concat(
    [dice_table_87_subR, dice_table_87_ctx_rh],
    axis=1
)

dice_table_87_lh = pd.concat(
    [dice_table_87_subL, dice_table_87_ctx_lh],
    axis=1
)

dice_table_86_rh = pd.concat(
    [dice_table_86_subR, dice_table_86_ctx_rh],
    axis=1
)

dice_table_86_lh = pd.concat(
    [dice_table_86_subL, dice_table_86_ctx_lh],
    axis=1
)

dice_table_85_rh = pd.concat(
    [dice_table_85_subR, dice_table_85_ctx_rh],
    axis=1
)

dice_table_85_lh = pd.concat(
    [dice_table_85_subL, dice_table_85_ctx_lh],
    axis=1
)

dice_table_87_rh["hemi"] = "rh"
dice_table_87_lh["hemi"] = "lh"
dice_table_87_rh["group"] = "FS 8 vs 7"
dice_table_87_lh["group"] = "FS 8 vs 7"

dice_table_86_rh["hemi"] = "rh"
dice_table_86_lh["hemi"] = "lh"
dice_table_86_rh["group"] = "FS 8 vs 6"
dice_table_86_lh["group"] = "FS 8 vs 6"

dice_table_85_rh["hemi"] = "rh"
dice_table_85_lh["hemi"] = "lh"
dice_table_85_rh["group"] = "FS 8 vs 5"
dice_table_85_lh["group"] = "FS 8 vs 5"

dice_table_all = pd.concat(
    [
        dice_table_76_rh,
        dice_table_76_lh,
        dice_table_75_rh,
        dice_table_75_lh,
        dice_table_65_rh,
        dice_table_65_lh,
        dice_table_87_rh,
        dice_table_87_lh,
        dice_table_86_rh,
        dice_table_86_lh,
        dice_table_85_rh,
        dice_table_85_lh,
    ]
)

dice_table_lh = pd.concat(
    [
        dice_table_76_lh,
        dice_table_75_lh,
        dice_table_65_lh,
        dice_table_87_lh,
        dice_table_86_lh,
        dice_table_85_lh,
    ]
)

dice_table_rh = pd.concat(
    [
        dice_table_76_rh,
        dice_table_75_rh,
        dice_table_65_rh,
        dice_table_87_rh,
        dice_table_86_rh,
        dice_table_85_rh,
    ]
)

dice_table_all.to_csv(out('dice_table_all.csv'))

# %%
# Start from the original Dice dataframe
dice_long = dice_table.copy()

# Keep only Dice columns
dice_cols = [c for c in dice_long.columns if c.startswith("dice_FS")]

dice_long = dice_long[dice_cols]

# Convert wide -> long
dice_long = dice_table.melt(
    value_vars=dice_cols,
    var_name="comparison_roi",
    value_name="dice"
)

# Extract FS comparison and ROI
extracted = dice_long["comparison_roi"].str.extract(
    r"^dice_FS(\d)(\d)_(.*)$"
)

extracted.columns = ["fs1", "fs2", "roi_raw"]

dice_long = pd.concat(
    [dice_long, extracted],
    axis=1
)

# Create comparison label
dice_long["comparison"] = (
    "FS "
    + dice_long["fs1"]
    + " vs "
    + dice_long["fs2"]
)

# Determine hemisphere
dice_long["hemi"] = dice_long["roi_raw"].str.extract(
    r"^(Left|Right|ctx_lh|ctx_rh)"
)[0]

dice_long["hemi"] = dice_long["hemi"].replace({
    "Left": "lh",
    "ctx_lh": "lh",
    "Right": "rh",
    "ctx_rh": "rh",
})

# Remove hemisphere prefix from ROI
dice_long["roi"] = dice_long["roi_raw"].str.replace(
    r"^(Left-|Right-|ctx_lh_|ctx_rh_)",
    "",
    regex=True
)

# Remove missing Dice values
dice_long = dice_long.dropna(subset=["dice"])

# %%
import plotly
import plotly.express as px
import plotly.graph_objects as go
fig = px.box(
    dice_long,
    x="roi",
    y="dice",
    color="comparison",
    facet_row="hemi",
)

fig.update_xaxes(
    tickangle=90,
    gridwidth=1,
    griddash="dot"
)

fig.update_yaxes(
    title="Sørensen–Dice coefficient",
    range=[0, 1]
)

fig.update_layout(
    legend_title_text="FreeSurfer comparison",
    height=1000,
    width=2800,
)

fig.show()

# %%
import plotly
import plotly.express as px
import plotly.graph_objects as go

x = dice_table_lh.copy()
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="group",
    orientation="h",
)

fig.update_yaxes(title="Regions (left hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Sørensen–Dice coefficients")
fig.update_layout(legend_title_text="", font_size=20, width=1600, height=2700)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_dice_lh.png"), width=1080, height=2700)
fig.show()

# %%
import plotly
import plotly.express as px
import plotly.graph_objects as go

x = dice_table_rh.copy()
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="group",
    orientation="h",
)

fig.update_yaxes(title="Regions (right hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Sørensen–Dice coefficients")
fig.update_layout(legend_title_text="", font_size=20, width=1600, height=2700)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_dice_lh.png"), width=1080, height=2700)
fig.show()

# %% [markdown]
# ### subcortical volumes

# %%
sub_table_5_a = aseg_table[
    [
        "Left-Lateral-Ventricle_FS5",
        "Left-Inf-Lat-Vent_FS5",
        "Left-Cerebellum-White-Matter_FS5",
        "Left-Cerebellum-Cortex_FS5",
        "Left-Thalamus_FS5",
        "Left-Caudate_FS5",
        "Left-Putamen_FS5",
        "Left-Pallidum_FS5",
        "3rd-Ventricle_FS5",
        "4th-Ventricle_FS5",
        "Brain-Stem_FS5",
        "Left-Hippocampus_FS5",
        "Left-Amygdala_FS5",
        "CSF_FS5",
        "Left-Accumbens-area_FS5",
        "Left-VentralDC_FS5",
        "Left-vessel_FS5",
        "Left-choroid-plexus_FS5",
        "Right-Lateral-Ventricle_FS5",
        "Right-Inf-Lat-Vent_FS5",
        "Right-Cerebellum-White-Matter_FS5",
        "Right-Cerebellum-Cortex_FS5",
        "Right-Thalamus_FS5",
        "Right-Caudate_FS5",
        "Right-Putamen_FS5",
        "Right-Pallidum_FS5",
        "Right-Hippocampus_FS5",
        "Right-Amygdala_FS5",
        "Right-Accumbens-area_FS5",
        "Right-VentralDC_FS5",
        "Right-vessel_FS5",
        "Right-choroid-plexus_FS5",
        "5th-Ventricle_FS5",
        "Optic-Chiasm_FS5",
        "CC_Posterior_FS5",
        "CC_Mid_Posterior_FS5",
        "CC_Central_FS5",
        "CC_Mid_Anterior_FS5",
        "CC_Anterior_FS5",
    ]
]

sub_table_6_a = aseg_table[
    [
        "Left-Lateral-Ventricle_FS6",
        "Left-Inf-Lat-Vent_FS6",
        "Left-Cerebellum-White-Matter_FS6",
        "Left-Cerebellum-Cortex_FS6",
        "Left-Thalamus_FS6",
        "Left-Caudate_FS6",
        "Left-Putamen_FS6",
        "Left-Pallidum_FS6",
        "3rd-Ventricle_FS6",
        "4th-Ventricle_FS6",
        "Brain-Stem_FS6",
        "Left-Hippocampus_FS6",
        "Left-Amygdala_FS6",
        "CSF_FS6",
        "Left-Accumbens-area_FS6",
        "Left-VentralDC_FS6",
        "Left-vessel_FS6",
        "Left-choroid-plexus_FS6",
        "Right-Lateral-Ventricle_FS6",
        "Right-Inf-Lat-Vent_FS6",
        "Right-Cerebellum-White-Matter_FS6",
        "Right-Cerebellum-Cortex_FS6",
        "Right-Thalamus_FS6",
        "Right-Caudate_FS6",
        "Right-Putamen_FS6",
        "Right-Pallidum_FS6",
        "Right-Hippocampus_FS6",
        "Right-Amygdala_FS6",
        "Right-Accumbens-area_FS6",
        "Right-VentralDC_FS6",
        "Right-vessel_FS6",
        "Right-choroid-plexus_FS6",
        "5th-Ventricle_FS6",
        "Optic-Chiasm_FS6",
        "CC_Posterior_FS6",
        "CC_Mid_Posterior_FS6",
        "CC_Central_FS6",
        "CC_Mid_Anterior_FS6",
        "CC_Anterior_FS6",
    ]
]

sub_table_7_a = aseg_table[
    [
        "Left-Lateral-Ventricle_FS7",
        "Left-Inf-Lat-Vent_FS7",
        "Left-Cerebellum-White-Matter_FS7",
        "Left-Cerebellum-Cortex_FS7",
        "Left-Thalamus_FS7",
        "Left-Caudate_FS7",
        "Left-Putamen_FS7",
        "Left-Pallidum_FS7",
        "3rd-Ventricle_FS7",
        "4th-Ventricle_FS7",
        "Brain-Stem_FS7",
        "Left-Hippocampus_FS7",
        "Left-Amygdala_FS7",
        "CSF_FS7",
        "Left-Accumbens-area_FS7",
        "Left-VentralDC_FS7",
        "Left-vessel_FS7",
        "Left-choroid-plexus_FS7",
        "Right-Lateral-Ventricle_FS7",
        "Right-Inf-Lat-Vent_FS7",
        "Right-Cerebellum-White-Matter_FS7",
        "Right-Cerebellum-Cortex_FS7",
        "Right-Thalamus_FS7",
        "Right-Caudate_FS7",
        "Right-Putamen_FS7",
        "Right-Pallidum_FS7",
        "Right-Hippocampus_FS7",
        "Right-Amygdala_FS7",
        "Right-Accumbens-area_FS7",
        "Right-VentralDC_FS7",
        "Right-vessel_FS7",
        "Right-choroid-plexus_FS7",
        "5th-Ventricle_FS7",
        "Optic-Chiasm_FS7",
        "CC_Posterior_FS7",
        "CC_Mid_Posterior_FS7",
        "CC_Central_FS7",
        "CC_Mid_Anterior_FS7",
        "CC_Anterior_FS7",
    ]
]

sub_table_8_a = aseg_table[
    [
        "Left-Lateral-Ventricle_FS8",
        "Left-Inf-Lat-Vent_FS8",
        "Left-Cerebellum-White-Matter_FS8",
        "Left-Cerebellum-Cortex_FS8",
        "Left-Thalamus_FS8",
        "Left-Caudate_FS8",
        "Left-Putamen_FS8",
        "Left-Pallidum_FS8",
        "3rd-Ventricle_FS8",
        "4th-Ventricle_FS8",
        "Brain-Stem_FS8",
        "Left-Hippocampus_FS8",
        "Left-Amygdala_FS8",
        "CSF_FS8",
        "Left-Accumbens-area_FS8",
        "Left-VentralDC_FS8",
        "Left-vessel_FS8",
        "Left-choroid-plexus_FS8",
        "Right-Lateral-Ventricle_FS8",
        "Right-Inf-Lat-Vent_FS8",
        "Right-Cerebellum-White-Matter_FS8",
        "Right-Cerebellum-Cortex_FS8",
        "Right-Thalamus_FS8",
        "Right-Caudate_FS8",
        "Right-Putamen_FS8",
        "Right-Pallidum_FS8",
        "Right-Hippocampus_FS8",
        "Right-Amygdala_FS8",
        "Right-Accumbens-area_FS8",
        "Right-VentralDC_FS8",
        "Right-vessel_FS8",
        "Right-choroid-plexus_FS8",
        "5th-Ventricle_FS8",
        "Optic-Chiasm_FS8",
        "CC_Posterior_FS8",
        "CC_Mid_Posterior_FS8",
        "CC_Central_FS8",
        "CC_Mid_Anterior_FS8",
        "CC_Anterior_FS8",
    ]
]

# %%
sub_table_coh = aseg_table.filter(like="dx_group")

sub_table_5 = pd.concat([sub_table_5_a, sub_table_coh], axis=1)
sub_table_6 = pd.concat([sub_table_6_a, sub_table_coh], axis=1)
sub_table_7 = pd.concat([sub_table_7_a, sub_table_coh], axis=1)
sub_table_8 = pd.concat([sub_table_8_a, sub_table_coh], axis=1)

sub_table_5["version"] = "FS5"
sub_table_6["version"] = "FS6"
sub_table_7["version"] = "FS7"
sub_table_8["version"] = "FS8"

# %%
sub_table_5.columns = sub_table_5.columns.str.replace("_FS5", "")
sub_table_6.columns = sub_table_6.columns.str.replace("_FS6", "")
sub_table_7.columns = sub_table_7.columns.str.replace("_FS7", "")
sub_table_8.columns = sub_table_8.columns.str.replace("_FS8", "")

sub_all = pd.concat([sub_table_5, sub_table_6, sub_table_7, sub_table_8])

sub_all.to_csv(out('sub_all.csv'))

# %% [markdown]
# ### volumes

# %%
vol_table_5_a = aseg_table.filter(like="_vol_FS5")
vol_table_6_a = aseg_table.filter(like="_vol_FS6")
vol_table_7_a = aseg_table.filter(like="_vol_FS7")
vol_table_8_a = aseg_table.filter(like="_vol_FS8")

vol_table_coh = aseg_table.filter(like="dx_group")

vol_table_5 = pd.concat([vol_table_5_a, vol_table_coh], axis=1)
vol_table_6 = pd.concat([vol_table_6_a, vol_table_coh], axis=1)
vol_table_7 = pd.concat([vol_table_7_a, vol_table_coh], axis=1)
vol_table_8 = pd.concat([vol_table_8_a, vol_table_coh], axis=1)

vol_table_5_lh = vol_table_5.filter(like="lh_vol_FS5")
vol_table_5_rh = vol_table_5.filter(like="rh_vol_FS5")

vol_table_6_lh = vol_table_6.filter(like="lh_vol_FS6")
vol_table_6_rh = vol_table_6.filter(like="rh_vol_FS6")

vol_table_7_lh = vol_table_7.filter(like="lh_vol_FS7")
vol_table_7_rh = vol_table_7.filter(like="rh_vol_FS7")

vol_table_8_lh = vol_table_8.filter(like="lh_vol_FS8")
vol_table_8_rh = vol_table_8.filter(like="rh_vol_FS8")

vol_table_5_coh = vol_table_5.filter(like="dx_group")
vol_table_6_coh = vol_table_6.filter(like="dx_group")
vol_table_7_coh = vol_table_7.filter(like="dx_group")
vol_table_8_coh = vol_table_8.filter(like="dx_group")

vol_table_5_lh = pd.concat([vol_table_5_lh, vol_table_5_coh], axis=1)
vol_table_6_lh = pd.concat([vol_table_6_lh, vol_table_6_coh], axis=1)
vol_table_7_lh = pd.concat([vol_table_7_lh, vol_table_7_coh], axis=1)
vol_table_8_lh = pd.concat([vol_table_8_lh, vol_table_8_coh], axis=1)

vol_table_5_rh = pd.concat([vol_table_5_rh, vol_table_5_coh], axis=1)
vol_table_6_rh = pd.concat([vol_table_6_rh, vol_table_6_coh], axis=1)
vol_table_7_rh = pd.concat([vol_table_7_rh, vol_table_7_coh], axis=1)
vol_table_8_rh = pd.concat([vol_table_8_rh, vol_table_8_coh], axis=1)

vol_table_5_lh["hemi"] = "lh"
vol_table_5_rh["hemi"] = "rh"

vol_table_6_lh["hemi"] = "lh"
vol_table_6_rh["hemi"] = "rh"

vol_table_7_lh["hemi"] = "lh"
vol_table_7_rh["hemi"] = "rh"

vol_table_8_lh["hemi"] = "lh"
vol_table_8_rh["hemi"] = "rh"

vol_table_5_lh["version"] = "FS5"
vol_table_5_rh["version"] = "FS5"

vol_table_6_lh["version"] = "FS6"
vol_table_6_rh["version"] = "FS6"

vol_table_7_lh["version"] = "FS7"
vol_table_7_rh["version"] = "FS7"

vol_table_8_lh["version"] = "FS8"
vol_table_8_rh["version"] = "FS8"

# %%
# rename columns
vol_table_5_lh.columns = vol_table_5_lh.columns.str.replace("_lh_vol_FS5", "")
vol_table_5_rh.columns = vol_table_5_rh.columns.str.replace("_rh_vol_FS5", "")
vol_table_6_lh.columns = vol_table_6_lh.columns.str.replace("_lh_vol_FS6", "")
vol_table_6_rh.columns = vol_table_6_rh.columns.str.replace("_rh_vol_FS6", "")
vol_table_7_lh.columns = vol_table_7_lh.columns.str.replace("_lh_vol_FS7", "")
vol_table_7_rh.columns = vol_table_7_rh.columns.str.replace("_rh_vol_FS7", "")
vol_table_8_lh.columns = vol_table_8_lh.columns.str.replace("_lh_vol_FS8", "")
vol_table_8_rh.columns = vol_table_8_rh.columns.str.replace("_rh_vol_FS8", "")

vol_all = pd.concat(
    [
        vol_table_5_lh,
        vol_table_5_rh,
        vol_table_6_lh,
        vol_table_6_rh,
        vol_table_7_lh,
        vol_table_7_rh,
        vol_table_8_lh,
        vol_table_8_rh,
    ]
)

# vol_all.to_csv(out('vol_all.csv'))

# %% [markdown]
# ### cortical thickness

# %%
ct_table_5_a = aseg_table.filter(like="_ct_FS5")
ct_table_6_a = aseg_table.filter(like="_ct_FS6")
ct_table_7_a = aseg_table.filter(like="_ct_FS7")
ct_table_8_a = aseg_table.filter(like="_ct_FS8")

ct_table_coh = aseg_table.filter(like="dx_group")

ct_table_5 = pd.concat([ct_table_5_a, ct_table_coh], axis=1)
ct_table_6 = pd.concat([ct_table_6_a, ct_table_coh], axis=1)
ct_table_7 = pd.concat([ct_table_7_a, ct_table_coh], axis=1)
ct_table_8 = pd.concat([ct_table_8_a, ct_table_coh], axis=1)

ct_table_5_lh = ct_table_5.filter(like="lh_ct_FS5")
ct_table_5_rh = ct_table_5.filter(like="rh_ct_FS5")
ct_table_6_lh = ct_table_6.filter(like="lh_ct_FS6")
ct_table_6_rh = ct_table_6.filter(like="rh_ct_FS6")
ct_table_7_lh = ct_table_7.filter(like="lh_ct_FS7")
ct_table_7_rh = ct_table_7.filter(like="rh_ct_FS7")
ct_table_8_lh = ct_table_8.filter(like="lh_ct_FS8")
ct_table_8_rh = ct_table_8.filter(like="rh_ct_FS8")

ct_table_5_coh = ct_table_5.filter(like="dx_group")
ct_table_6_coh = ct_table_6.filter(like="dx_group")
ct_table_7_coh = ct_table_7.filter(like="dx_group")
ct_table_8_coh = ct_table_8.filter(like="dx_group")

ct_table_5_lh = pd.concat([ct_table_5_lh, ct_table_5_coh], axis=1)
ct_table_6_lh = pd.concat([ct_table_6_lh, ct_table_6_coh], axis=1)
ct_table_7_lh = pd.concat([ct_table_7_lh, ct_table_7_coh], axis=1)
ct_table_8_lh = pd.concat([ct_table_8_lh, ct_table_8_coh], axis=1)
ct_table_5_rh = pd.concat([ct_table_5_rh, ct_table_5_coh], axis=1)
ct_table_6_rh = pd.concat([ct_table_6_rh, ct_table_6_coh], axis=1)
ct_table_7_rh = pd.concat([ct_table_7_rh, ct_table_7_coh], axis=1)
ct_table_8_rh = pd.concat([ct_table_8_rh, ct_table_8_coh], axis=1)

ct_table_5_lh["hemi"] = "lh"
ct_table_5_rh["hemi"] = "rh"
ct_table_6_lh["hemi"] = "lh"
ct_table_6_rh["hemi"] = "rh"
ct_table_7_lh["hemi"] = "lh"
ct_table_7_rh["hemi"] = "rh"
ct_table_8_lh["hemi"] = "lh"
ct_table_8_rh["hemi"] = "rh"

ct_table_5_lh["version"] = "FS5"
ct_table_5_rh["version"] = "FS5"
ct_table_6_lh["version"] = "FS6"
ct_table_6_rh["version"] = "FS6"
ct_table_7_lh["version"] = "FS7"
ct_table_7_rh["version"] = "FS7"
ct_table_8_lh["version"] = "FS8"
ct_table_8_rh["version"] = "FS8"

# %%
# rename columns
ct_table_5_lh.columns = ct_table_5_lh.columns.str.replace("_lh_ct_FS5", "")
ct_table_5_rh.columns = ct_table_5_rh.columns.str.replace("_rh_ct_FS5", "")
ct_table_6_lh.columns = ct_table_6_lh.columns.str.replace("_lh_ct_FS6", "")
ct_table_6_rh.columns = ct_table_6_rh.columns.str.replace("_rh_ct_FS6", "")
ct_table_7_lh.columns = ct_table_7_lh.columns.str.replace("_lh_ct_FS7", "")
ct_table_7_rh.columns = ct_table_7_rh.columns.str.replace("_rh_ct_FS7", "")
ct_table_8_lh.columns = ct_table_8_lh.columns.str.replace("_lh_ct_FS8", "")
ct_table_8_rh.columns = ct_table_8_rh.columns.str.replace("_rh_ct_FS8", "")

ct_all = pd.concat(
    [
        ct_table_5_lh,
        ct_table_5_rh,
        ct_table_6_lh,
        ct_table_6_rh,
        ct_table_7_lh,
        ct_table_7_rh,
        ct_table_8_lh,
        ct_table_8_rh,
    ]
)


ct_table_rh = pd.concat([ct_table_5_rh, ct_table_6_rh, ct_table_7_rh, ct_table_8_rh])
ct_table_lh = pd.concat([ct_table_5_lh, ct_table_6_lh, ct_table_7_lh, ct_table_8_lh])


# ct_all.to_csv(out('ct_all.csv'))

# %% [markdown]
# ### surface area

# %%
surf_table_5_a = aseg_table.filter(like="_surf_FS5")
surf_table_6_a = aseg_table.filter(like="_surf_FS6")
surf_table_7_a = aseg_table.filter(like="_surf_FS7")
surf_table_8_a = aseg_table.filter(like="_surf_FS8")

surf_table_coh = aseg_table.filter(like="dx_group")

surf_table_5 = pd.concat([surf_table_5_a, surf_table_coh], axis=1)
surf_table_6 = pd.concat([surf_table_6_a, surf_table_coh], axis=1)
surf_table_7 = pd.concat([surf_table_7_a, surf_table_coh], axis=1)
surf_table_8 = pd.concat([surf_table_8_a, surf_table_coh], axis=1)

surf_table_5_lh = surf_table_5.filter(like="lh_surf_FS5")
surf_table_5_rh = surf_table_5.filter(like="rh_surf_FS5")
surf_table_6_lh = surf_table_6.filter(like="lh_surf_FS6")
surf_table_6_rh = surf_table_6.filter(like="rh_surf_FS6")
surf_table_7_lh = surf_table_7.filter(like="lh_surf_FS7")
surf_table_7_rh = surf_table_7.filter(like="rh_surf_FS7")
surf_table_8_lh = surf_table_8.filter(like="lh_surf_FS8")
surf_table_8_rh = surf_table_8.filter(like="rh_surf_FS8")


surf_table_5_coh = surf_table_5.filter(like="dx_group")
surf_table_6_coh = surf_table_6.filter(like="dx_group")
surf_table_7_coh = surf_table_7.filter(like="dx_group")
surf_table_8_coh = surf_table_8.filter(like="dx_group")

surf_table_5_lh = pd.concat([surf_table_5_lh, surf_table_5_coh], axis=1)
surf_table_6_lh = pd.concat([surf_table_6_lh, surf_table_6_coh], axis=1)
surf_table_7_lh = pd.concat([surf_table_7_lh, surf_table_7_coh], axis=1)
surf_table_8_lh = pd.concat([surf_table_8_lh, surf_table_8_coh], axis=1)

surf_table_5_rh = pd.concat([surf_table_5_rh, surf_table_5_coh], axis=1)
surf_table_6_rh = pd.concat([surf_table_6_rh, surf_table_6_coh], axis=1)
surf_table_7_rh = pd.concat([surf_table_7_rh, surf_table_7_coh], axis=1)
surf_table_8_rh = pd.concat([surf_table_8_rh, surf_table_8_coh], axis=1)


surf_table_5_lh["hemi"] = "lh"
surf_table_5_rh["hemi"] = "rh"
surf_table_6_lh["hemi"] = "lh"
surf_table_6_rh["hemi"] = "rh"
surf_table_7_lh["hemi"] = "lh"
surf_table_7_rh["hemi"] = "rh"
surf_table_8_lh["hemi"] = "lh"
surf_table_8_rh["hemi"] = "rh"

surf_table_5_lh["version"] = "FS5"
surf_table_5_rh["version"] = "FS5"
surf_table_6_lh["version"] = "FS6"
surf_table_6_rh["version"] = "FS6"
surf_table_7_lh["version"] = "FS7"
surf_table_7_rh["version"] = "FS7"
surf_table_8_lh["version"] = "FS8"
surf_table_8_rh["version"] = "FS8"

# %%
# rename columns
surf_table_5_lh.columns = surf_table_5_lh.columns.str.replace("_lh_surf_FS5", "")
surf_table_5_rh.columns = surf_table_5_rh.columns.str.replace("_rh_surf_FS5", "")
surf_table_6_lh.columns = surf_table_6_lh.columns.str.replace("_lh_surf_FS6", "")
surf_table_6_rh.columns = surf_table_6_rh.columns.str.replace("_rh_surf_FS6", "")
surf_table_7_lh.columns = surf_table_7_lh.columns.str.replace("_lh_surf_FS7", "")
surf_table_7_rh.columns = surf_table_7_rh.columns.str.replace("_rh_surf_FS7", "")
surf_table_8_lh.columns = surf_table_8_lh.columns.str.replace("_lh_surf_FS8", "")
surf_table_8_rh.columns = surf_table_8_rh.columns.str.replace("_rh_surf_FS8", "")

surf_all = pd.concat(
    [
        surf_table_5_lh,
        surf_table_5_rh,
        surf_table_6_lh,
        surf_table_6_rh,
        surf_table_7_lh,
        surf_table_7_rh,
        surf_table_8_lh,
        surf_table_8_rh,
    ]
)

surf_table_rh = pd.concat([surf_table_5_rh, surf_table_6_rh, surf_table_7_rh, surf_table_8_rh])
surf_table_lh = pd.concat([surf_table_5_lh, surf_table_6_lh, surf_table_7_lh, surf_table_8_lh])

# surf_all.to_csv(out('surf_all.csv'))

# %% [markdown]
# ## Plots - structural estimations

# %%
import plotly.express as px
import kaleido

y_sub = [
    "Left-Lateral-Ventricle",
    "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Right-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Right-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex",
    "Left-Thalamus",
    "Right-Thalamus",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Pallidum",
    "Right-Pallidum",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Accumbens-area",
    "Right-Accumbens-area",
    "Left-VentralDC",
    "Right-VentralDC",
    "Left-vessel",
    "Right-vessel",
    "Left-choroid-plexus",
    "Right-choroid-plexus",
    "3rd-Ventricle",
    "4th-Ventricle",
    "5th-Ventricle",
    "CSF",
    "Brain-Stem",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

y_cort = [
    "G_and_S_frontomargin",
    "G_and_S_occipital_inf",
    "G_and_S_paracentral",
    "G_and_S_subcentral",
    "G_and_S_transv_frontopol",
    "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral",
    "G_cuneus",
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
    "G_Ins_lg_and_S_cent_ins",
    "G_insular_short",
    "G_occipital_middle",
    "G_occipital_sup",
    "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual",
    "G_oc-temp_med-Parahip",
    "G_orbital",
    "G_pariet_inf-Angular",
    "G_pariet_inf-Supramar",
    "G_parietal_sup",
    "G_postcentral",
    "G_precentral",
    "G_precuneus",
    "G_rectus",
    "G_subcallosal",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    "G_temporal_inf",
    "G_temporal_middle",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Pole_occipital",
    "Pole_temporal",
    "S_calcarine",
    "S_central",
    "S_cingul-Marginalis",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
    "S_collat_transv_ant",
    "S_collat_transv_post",
    "S_front_inf",
    "S_front_middle",
    "S_front_sup",
    "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans",
    "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal",
    "S_occipital_ant",
    "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral",
    "S_orbital_med-olfact",
    "S_orbital-H_Shaped",
    "S_parieto_occipital",
    "S_pericallosal",
    "S_postcentral",
    "S_precentral-inf-part",
    "S_precentral-sup-part",
    "S_suborbital",
    "S_subparietal",
    "S_temporal_inf",
    "S_temporal_sup",
    "S_temporal_transverse",
]

# %%
fig = px.box(
    sub_all,
    y=y_sub,
    color="version",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume<br> (mm<sup>3</sup>)")
# fig.write_image(out("plot_sub.png"), width=1920, height=1080)
fig.show()

# %%
fig = px.box(
    vol_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume<br> (mm<sup>3</sup>)")
# fig.write_image(out("plot_vol.png"), width=1920, height=1080)
fig.show()

# %%
fig = px.box(
    surf_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Surface area<br> (mm<sup>2</sup>)")
# fig.write_image(out("plot_surf.png"), width=1920, height=1080)
fig.show()

# %%
fig = px.box(
    ct_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Thickness<br> (mm)")
# fig.write_image(out("plot_ct.png"), width=1920, height=1080)
fig.show()

# %% [markdown]
# # Plots - between-version and between-subject variability

# %%
# sd across subjects

sub5_between = sub_all.loc[(sub_all["version"] == "FS5")]
sub5_between = sub5_between.std(numeric_only=True)
sub5_between["version"] = "FS5"


sub6_between = sub_all.loc[(sub_all["version"] == "FS6")]
sub6_between = sub6_between.std(numeric_only=True)
sub6_between["version"] = "FS6"


sub7_between = sub_all.loc[(sub_all["version"] == "FS7")]
sub7_between = sub7_between.std(numeric_only=True)
sub7_between["version"] = "FS7"

sub8_between = sub_all.loc[(sub_all["version"] == "FS8")]
sub8_between = sub8_between.std(numeric_only=True)
sub8_between["version"] = "FS8"


sub5_between = pd.DataFrame(data=sub5_between).T
sub6_between = pd.DataFrame(data=sub6_between).T
sub7_between = pd.DataFrame(data=sub7_between).T
sub8_between = pd.DataFrame(data=sub8_between).T

sub_between = pd.concat([sub5_between, sub6_between, sub7_between, sub8_between])

sub_between = sub_between[
    [
        "Left-Lateral-Ventricle",
        "Right-Lateral-Ventricle",
        "Left-Inf-Lat-Vent",
        "Right-Inf-Lat-Vent",
        "Left-Cerebellum-White-Matter",
        "Right-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Right-Cerebellum-Cortex",
        "Left-Thalamus",
        "Right-Thalamus",
        "Left-Caudate",
        "Right-Caudate",
        "Left-Putamen",
        "Right-Putamen",
        "Left-Pallidum",
        "Right-Pallidum",
        "Left-Hippocampus",
        "Right-Hippocampus",
        "Left-Amygdala",
        "Right-Amygdala",
        "Left-Accumbens-area",
        "Right-Accumbens-area",
        "Left-VentralDC",
        "Right-VentralDC",
        "Left-vessel",
        "Right-vessel",
        "Left-choroid-plexus",
        "Right-choroid-plexus",
        "3rd-Ventricle",
        "4th-Ventricle",
        "5th-Ventricle",
        "CSF",
        "Brain-Stem",
        "Optic-Chiasm",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
    ]
]

# %%
# sd across subjects cort

# vol
vol5lh_between = vol_all.loc[(vol_all["version"] == "FS5") & (vol_all["hemi"] == "lh")]
vol5lh_between = vol5lh_between.std(numeric_only=True)
vol5lh_between["version"] = "FS5"
vol5lh_between["hemi"] = "lh"

vol5rh_between = vol_all.loc[(vol_all["version"] == "FS5") & (vol_all["hemi"] == "rh")]
vol5rh_between = vol5rh_between.std(numeric_only=True)
vol5rh_between["version"] = "FS5"
vol5rh_between["hemi"] = "rh"

vol6lh_between = vol_all.loc[(vol_all["version"] == "FS6") & (vol_all["hemi"] == "lh")]
vol6lh_between = vol6lh_between.std(numeric_only=True)
vol6lh_between["version"] = "FS6"
vol6lh_between["hemi"] = "lh"

vol6rh_between = vol_all.loc[(vol_all["version"] == "FS6") & (vol_all["hemi"] == "rh")]
vol6rh_between = vol6rh_between.std(numeric_only=True)
vol6rh_between["version"] = "FS6"
vol6rh_between["hemi"] = "rh"

vol7lh_between = vol_all.loc[(vol_all["version"] == "FS7") & (vol_all["hemi"] == "lh")]
vol7lh_between = vol7lh_between.std(numeric_only=True)
vol7lh_between["version"] = "FS7"
vol7lh_between["hemi"] = "lh"

vol7rh_between = vol_all.loc[(vol_all["version"] == "FS7") & (vol_all["hemi"] == "rh")]
vol7rh_between = vol7rh_between.std(numeric_only=True)
vol7rh_between["version"] = "FS7"
vol7rh_between["hemi"] = "rh"

vol8lh_between = vol_all.loc[(vol_all["version"] == "FS8") & (vol_all["hemi"] == "lh")]
vol8lh_between = vol8lh_between.std(numeric_only=True)
vol8lh_between["version"] = "FS8"
vol8lh_between["hemi"] = "lh"

vol8rh_between = vol_all.loc[(vol_all["version"] == "FS8") & (vol_all["hemi"] == "rh")]
vol8rh_between = vol8rh_between.std(numeric_only=True)
vol8rh_between["version"] = "FS8"
vol8rh_between["hemi"] = "rh"


vol5lh_between = pd.DataFrame(data=vol5lh_between).T
vol5rh_between = pd.DataFrame(data=vol5rh_between).T
vol6lh_between = pd.DataFrame(data=vol6lh_between).T
vol6rh_between = pd.DataFrame(data=vol6rh_between).T
vol7lh_between = pd.DataFrame(data=vol7lh_between).T
vol7rh_between = pd.DataFrame(data=vol7rh_between).T
vol8lh_between = pd.DataFrame(data=vol8lh_between).T
vol8rh_between = pd.DataFrame(data=vol8rh_between).T

vol_between = pd.concat(
    [
        vol5lh_between,
        vol5rh_between,
        vol6lh_between,
        vol6rh_between,
        vol7lh_between,
        vol7rh_between,
        vol8lh_between,
        vol8rh_between,
    ]
)

# surf
surf5lh_between = surf_all.loc[
    (surf_all["version"] == "FS5") & (surf_all["hemi"] == "lh")
]
surf5lh_between = surf5lh_between.std(numeric_only=True)
surf5lh_between["version"] = "FS5"
surf5lh_between["hemi"] = "lh"

surf5rh_between = surf_all.loc[
    (surf_all["version"] == "FS5") & (surf_all["hemi"] == "rh")
]
surf5rh_between = surf5rh_between.std(numeric_only=True)
surf5rh_between["version"] = "FS5"
surf5rh_between["hemi"] = "rh"

surf6lh_between = surf_all.loc[
    (surf_all["version"] == "FS6") & (surf_all["hemi"] == "lh")
]
surf6lh_between = surf6lh_between.std(numeric_only=True)
surf6lh_between["version"] = "FS6"
surf6lh_between["hemi"] = "lh"

surf6rh_between = surf_all.loc[
    (surf_all["version"] == "FS6") & (surf_all["hemi"] == "rh")
]
surf6rh_between = surf6rh_between.std(numeric_only=True)
surf6rh_between["version"] = "FS6"
surf6rh_between["hemi"] = "rh"

surf7lh_between = surf_all.loc[
    (surf_all["version"] == "FS7") & (surf_all["hemi"] == "lh")
]
surf7lh_between = surf7lh_between.std(numeric_only=True)
surf7lh_between["version"] = "FS7"
surf7lh_between["hemi"] = "lh"

surf7rh_between = surf_all.loc[
    (surf_all["version"] == "FS7") & (surf_all["hemi"] == "rh")
]
surf7rh_between = surf7rh_between.std(numeric_only=True)
surf7rh_between["version"] = "FS7"
surf7rh_between["hemi"] = "rh"

surf8lh_between = surf_all.loc[
    (surf_all["version"] == "FS8") & (surf_all["hemi"] == "lh")
]
surf8lh_between = surf8lh_between.std(numeric_only=True)
surf8lh_between["version"] = "FS8"
surf8lh_between["hemi"] = "lh"

surf8rh_between = surf_all.loc[
    (surf_all["version"] == "FS8") & (surf_all["hemi"] == "rh")
]
surf8rh_between = surf8rh_between.std(numeric_only=True)
surf8rh_between["version"] = "FS8"
surf8rh_between["hemi"] = "rh"


surf5lh_between = pd.DataFrame(data=surf5lh_between).T
surf5rh_between = pd.DataFrame(data=surf5rh_between).T
surf6lh_between = pd.DataFrame(data=surf6lh_between).T
surf6rh_between = pd.DataFrame(data=surf6rh_between).T
surf7lh_between = pd.DataFrame(data=surf7lh_between).T
surf7rh_between = pd.DataFrame(data=surf7rh_between).T
surf8lh_between = pd.DataFrame(data=surf8lh_between).T
surf8rh_between = pd.DataFrame(data=surf8rh_between).T

surf_between = pd.concat(
    [
        surf5lh_between,
        surf5rh_between,
        surf6lh_between,
        surf6rh_between,
        surf7lh_between,
        surf7rh_between,
        surf8lh_between,
        surf8rh_between,
    ]
)

# ct
ct5lh_between = ct_all.loc[(ct_all["version"] == "FS5") & (ct_all["hemi"] == "lh")]
ct5lh_between = ct5lh_between.std(numeric_only=True)
ct5lh_between["version"] = "FS5"
ct5lh_between["hemi"] = "lh"

ct5rh_between = ct_all.loc[(ct_all["version"] == "FS5") & (ct_all["hemi"] == "rh")]
ct5rh_between = ct5rh_between.std(numeric_only=True)
ct5rh_between["version"] = "FS5"
ct5rh_between["hemi"] = "rh"

ct6lh_between = ct_all.loc[(ct_all["version"] == "FS6") & (ct_all["hemi"] == "lh")]
ct6lh_between = ct6lh_between.std(numeric_only=True)
ct6lh_between["version"] = "FS6"
ct6lh_between["hemi"] = "lh"

ct6rh_between = ct_all.loc[(ct_all["version"] == "FS6") & (ct_all["hemi"] == "rh")]
ct6rh_between = ct6rh_between.std(numeric_only=True)
ct6rh_between["version"] = "FS6"
ct6rh_between["hemi"] = "rh"

ct7lh_between = ct_all.loc[(ct_all["version"] == "FS7") & (ct_all["hemi"] == "lh")]
ct7lh_between = ct7lh_between.std(numeric_only=True)
ct7lh_between["version"] = "FS7"
ct7lh_between["hemi"] = "lh"

ct7rh_between = ct_all.loc[(ct_all["version"] == "FS7") & (ct_all["hemi"] == "rh")]
ct7rh_between = ct7rh_between.std(numeric_only=True)
ct7rh_between["version"] = "FS7"
ct7rh_between["hemi"] = "rh"

ct8lh_between = ct_all.loc[(ct_all["version"] == "FS8") & (ct_all["hemi"] == "lh")]
ct8lh_between = ct8lh_between.std(numeric_only=True)
ct8lh_between["version"] = "FS8"
ct8lh_between["hemi"] = "lh"

ct8rh_between = ct_all.loc[(ct_all["version"] == "FS8") & (ct_all["hemi"] == "rh")]
ct8rh_between = ct8rh_between.std(numeric_only=True)
ct8rh_between["version"] = "FS8"
ct8rh_between["hemi"] = "rh"

ct5lh_between = pd.DataFrame(data=ct5lh_between).T
ct5rh_between = pd.DataFrame(data=ct5rh_between).T
ct6lh_between = pd.DataFrame(data=ct6lh_between).T
ct6rh_between = pd.DataFrame(data=ct6rh_between).T
ct7lh_between = pd.DataFrame(data=ct7lh_between).T
ct7rh_between = pd.DataFrame(data=ct7rh_between).T
ct8lh_between = pd.DataFrame(data=ct8lh_between).T
ct8rh_between = pd.DataFrame(data=ct8rh_between).T

ct_between = pd.concat(
    [
        ct5lh_between,
        ct5rh_between,
        ct6lh_between,
        ct6rh_between,
        ct7lh_between,
        ct7rh_between,
        ct8lh_between,
        ct8rh_between,
    ]
)

# %%
# difference between versions

sub_diff_table_76_a = aseg_table.filter(like="FS7_6_diff_")
sub_diff_table_75_a = aseg_table.filter(like="FS7_5_diff_")
sub_diff_table_65_a = aseg_table.filter(like="FS6_5_diff_")

# FS8 comparisons
sub_diff_table_87_a = aseg_table.filter(like="FS8_7_diff_")
sub_diff_table_86_a = aseg_table.filter(like="FS8_6_diff_")
sub_diff_table_85_a = aseg_table.filter(like="FS8_5_diff_")

sub_diff_table_coh = aseg_table.filter(like="dx_group")

sub_diff_table_75 = pd.concat(
    [sub_diff_table_75_a, sub_diff_table_coh], axis=1
)

sub_diff_table_76 = pd.concat(
    [sub_diff_table_76_a, sub_diff_table_coh], axis=1
)

sub_diff_table_65 = pd.concat(
    [sub_diff_table_65_a, sub_diff_table_coh], axis=1
)

sub_diff_table_87 = pd.concat(
    [sub_diff_table_87_a, sub_diff_table_coh], axis=1
)

sub_diff_table_86 = pd.concat(
    [sub_diff_table_86_a, sub_diff_table_coh], axis=1
)

sub_diff_table_85 = pd.concat(
    [sub_diff_table_85_a, sub_diff_table_coh], axis=1
)

sub_diff_table_76["version"] = "FS76"
sub_diff_table_75["version"] = "FS75"
sub_diff_table_65["version"] = "FS65"

sub_diff_table_87["version"] = "FS87"
sub_diff_table_86["version"] = "FS86"
sub_diff_table_85["version"] = "FS85"

# %%
sub_diff_table_76.columns = sub_diff_table_76.columns.str.replace(
    "FS7_6_diff_", ""
)

sub_diff_table_75.columns = sub_diff_table_75.columns.str.replace(
    "FS7_5_diff_", ""
)

sub_diff_table_65.columns = sub_diff_table_65.columns.str.replace(
    "FS6_5_diff_", ""
)

# FS8
sub_diff_table_87.columns = sub_diff_table_87.columns.str.replace(
    "FS8_7_diff_", ""
)

sub_diff_table_86.columns = sub_diff_table_86.columns.str.replace(
    "FS8_6_diff_", ""
)

sub_diff_table_85.columns = sub_diff_table_85.columns.str.replace(
    "FS8_5_diff_", ""
)

# %%
sub_diff_cols = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
    "version",
]

sub_diff_table_76 = sub_diff_table_76[sub_diff_cols]
sub_diff_table_75 = sub_diff_table_75[sub_diff_cols]
sub_diff_table_65 = sub_diff_table_65[sub_diff_cols]

# FS8 comparisons
sub_diff_table_87 = sub_diff_table_87[sub_diff_cols]
sub_diff_table_86 = sub_diff_table_86[sub_diff_cols]
sub_diff_table_85 = sub_diff_table_85[sub_diff_cols]


sub_diff_all = pd.concat([
    sub_diff_table_76,
    sub_diff_table_75,
    sub_diff_table_65,
    sub_diff_table_87,
    sub_diff_table_86,
    sub_diff_table_85,
])

# sub_diff_all.to_csv(out("sub_diff_all.csv"))

# %%
# create df with difference

diff_table_76_a = aseg_table.filter(like="FS7_6_diff_")
diff_table_75_a = aseg_table.filter(like="FS7_5_diff_")
diff_table_65_a = aseg_table.filter(like="FS6_5_diff_")

diff_table_87_a = aseg_table.filter(like="FS8_7_diff_")
diff_table_86_a = aseg_table.filter(like="FS8_6_diff_")
diff_table_85_a = aseg_table.filter(like="FS8_5_diff_")

# create df with volume difference


diff_table_vol_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?vol"), axis=1)
diff_table_vol_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?vol"), axis=1)
diff_table_vol_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?vol"), axis=1)
diff_table_vol_87_rh = diff_table_87_a.filter(regex=("FS8_7_diff_rh.+?vol"), axis=1)
diff_table_vol_86_rh = diff_table_86_a.filter(regex=("FS8_6_diff_rh.+?vol"), axis=1)
diff_table_vol_85_rh = diff_table_85_a.filter(regex=("FS8_5_diff_rh.+?vol"), axis=1)
diff_table_vol_76_rh["version"] = "FS76"
diff_table_vol_75_rh["version"] = "FS75"
diff_table_vol_65_rh["version"] = "FS65"
diff_table_vol_87_rh["version"] = "FS87"
diff_table_vol_86_rh["version"] = "FS86"
diff_table_vol_85_rh["version"] = "FS85"
diff_table_vol_76_rh["hemi"] = "rh"
diff_table_vol_75_rh["hemi"] = "rh"
diff_table_vol_65_rh["hemi"] = "rh"
diff_table_vol_87_rh["hemi"] = "rh"
diff_table_vol_86_rh["hemi"] = "rh"
diff_table_vol_85_rh["hemi"] = "rh"

diff_table_vol_76_lh = sub_diff_table_76_a.filter(regex=("FS7_6_diff_lh.+?vol"), axis=1)
diff_table_vol_75_lh = sub_diff_table_75_a.filter(regex=("FS7_5_diff_lh.+?vol"), axis=1)
diff_table_vol_65_lh = sub_diff_table_65_a.filter(regex=("FS6_5_diff_lh.+?vol"), axis=1)
diff_table_vol_87_lh = sub_diff_table_87_a.filter(regex=("FS8_7_diff_lh.+?vol"), axis=1)
diff_table_vol_86_lh = sub_diff_table_86_a.filter(regex=("FS8_6_diff_lh.+?vol"), axis=1)
diff_table_vol_85_lh = sub_diff_table_85_a.filter(regex=("FS8_5_diff_lh.+?vol"), axis=1)
diff_table_vol_76_lh["version"] = "FS76"
diff_table_vol_75_lh["version"] = "FS75"
diff_table_vol_65_lh["version"] = "FS65"
diff_table_vol_87_lh["version"] = "FS87"
diff_table_vol_86_lh["version"] = "FS86"
diff_table_vol_85_lh["version"] = "FS85"
diff_table_vol_76_lh["hemi"] = "lh"
diff_table_vol_75_lh["hemi"] = "lh"
diff_table_vol_65_lh["hemi"] = "lh"
diff_table_vol_87_lh["hemi"] = "lh"
diff_table_vol_86_lh["hemi"] = "lh"
diff_table_vol_85_lh["hemi"] = "lh"

diff_table_vol_76_rh.columns = diff_table_vol_76_rh.columns.str.replace(
    "FS7_6_diff_rh_", ""
)
diff_table_vol_76_rh.columns = diff_table_vol_76_rh.columns.str.replace("_vol", "")
diff_table_vol_75_rh.columns = diff_table_vol_75_rh.columns.str.replace(
    "FS7_5_diff_rh_", ""
)
diff_table_vol_75_rh.columns = diff_table_vol_75_rh.columns.str.replace("_vol", "")
diff_table_vol_65_rh.columns = diff_table_vol_65_rh.columns.str.replace(
    "FS6_5_diff_rh_", ""
)
diff_table_vol_65_rh.columns = diff_table_vol_65_rh.columns.str.replace("_vol", "")
diff_table_vol_87_rh.columns = diff_table_vol_87_rh.columns.str.replace(
    "FS8_7_diff_rh_", ""
)
diff_table_vol_87_rh.columns = diff_table_vol_87_rh.columns.str.replace("_vol", "")

diff_table_vol_86_rh.columns = diff_table_vol_86_rh.columns.str.replace(
    "FS8_6_diff_rh_", ""
)
diff_table_vol_86_rh.columns = diff_table_vol_86_rh.columns.str.replace("_vol", "")

diff_table_vol_85_rh.columns = diff_table_vol_85_rh.columns.str.replace(
    "FS8_5_diff_rh_", ""
)
diff_table_vol_85_rh.columns = diff_table_vol_85_rh.columns.str.replace("_vol", "")


diff_table_vol_76_lh.columns = diff_table_vol_76_lh.columns.str.replace(
    "FS7_6_diff_lh_", ""
)
diff_table_vol_76_lh.columns = diff_table_vol_76_lh.columns.str.replace("_vol", "")
diff_table_vol_75_lh.columns = diff_table_vol_75_lh.columns.str.replace(
    "FS7_5_diff_lh_", ""
)
diff_table_vol_75_lh.columns = diff_table_vol_75_lh.columns.str.replace("_vol", "")
diff_table_vol_65_lh.columns = diff_table_vol_65_lh.columns.str.replace(
    "FS6_5_diff_lh_", ""
)
diff_table_vol_65_lh.columns = diff_table_vol_65_lh.columns.str.replace("_vol", "")
diff_table_vol_87_lh.columns = diff_table_vol_87_lh.columns.str.replace(
    "FS8_7_diff_lh_", ""
)
diff_table_vol_87_lh.columns = diff_table_vol_87_lh.columns.str.replace("_vol", "")

diff_table_vol_86_lh.columns = diff_table_vol_86_lh.columns.str.replace(
    "FS8_6_diff_lh_", ""
)
diff_table_vol_86_lh.columns = diff_table_vol_86_lh.columns.str.replace("_vol", "")

diff_table_vol_85_lh.columns = diff_table_vol_85_lh.columns.str.replace(
    "FS8_5_diff_lh_", ""
)
diff_table_vol_85_lh.columns = diff_table_vol_85_lh.columns.str.replace("_vol", "")


diff_table_vol_all = pd.concat(
    [
        diff_table_vol_76_rh,
        diff_table_vol_75_rh,
        diff_table_vol_65_rh,
        diff_table_vol_87_rh,
        diff_table_vol_86_rh,
        diff_table_vol_85_rh,
        diff_table_vol_76_lh,
        diff_table_vol_75_lh,
        diff_table_vol_65_lh,
        diff_table_vol_87_lh,
        diff_table_vol_86_lh,
        diff_table_vol_85_lh,
    ]
)

diff_table_vol_rh = pd.concat(
    [diff_table_vol_76_rh, diff_table_vol_75_rh, diff_table_vol_65_rh, diff_table_vol_87_rh, diff_table_vol_86_rh, diff_table_vol_85_rh]
)
diff_table_vol_lh = pd.concat(
    [diff_table_vol_76_lh, diff_table_vol_75_lh, diff_table_vol_65_lh, diff_table_vol_87_lh, diff_table_vol_86_lh, diff_table_vol_85_lh]
)

# create df with surface area difference

diff_table_surf_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?surf"), axis=1)
diff_table_surf_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?surf"), axis=1)
diff_table_surf_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?surf"), axis=1)
diff_table_surf_87_rh = diff_table_87_a.filter(regex=("FS8_7_diff_rh.+?surf"), axis=1)
diff_table_surf_86_rh = diff_table_86_a.filter(regex=("FS8_6_diff_rh.+?surf"), axis=1)
diff_table_surf_85_rh = diff_table_85_a.filter(regex=("FS8_5_diff_rh.+?surf"), axis=1)
diff_table_surf_76_rh["version"] = "FS76"
diff_table_surf_75_rh["version"] = "FS75"
diff_table_surf_65_rh["version"] = "FS65"
diff_table_surf_87_rh["version"] = "FS87"
diff_table_surf_86_rh["version"] = "FS86"
diff_table_surf_85_rh["version"] = "FS85"
diff_table_surf_76_rh["hemi"] = "rh"
diff_table_surf_75_rh["hemi"] = "rh"
diff_table_surf_65_rh["hemi"] = "rh"
diff_table_surf_87_rh["hemi"] = "rh"
diff_table_surf_86_rh["hemi"] = "rh"
diff_table_surf_85_rh["hemi"] = "rh"

diff_table_surf_76_lh = sub_diff_table_76_a.filter(
    regex=("FS7_6_diff_lh.+?surf"), axis=1
)
diff_table_surf_75_lh = sub_diff_table_75_a.filter(
    regex=("FS7_5_diff_lh.+?surf"), axis=1
)
diff_table_surf_65_lh = sub_diff_table_65_a.filter(
    regex=("FS6_5_diff_lh.+?surf"), axis=1
)

diff_table_surf_87_lh = sub_diff_table_87_a.filter(
    regex=("FS8_7_diff_lh.+?surf"), axis=1
)
diff_table_surf_86_lh = sub_diff_table_86_a.filter(
    regex=("FS8_6_diff_lh.+?surf"), axis=1
)
diff_table_surf_85_lh = sub_diff_table_85_a.filter(
    regex=("FS8_5_diff_lh.+?surf"), axis=1
)


diff_table_surf_76_lh["version"] = "FS76"
diff_table_surf_75_lh["version"] = "FS75"
diff_table_surf_65_lh["version"] = "FS65"
diff_table_surf_87_lh["version"] = "FS87"
diff_table_surf_86_lh["version"] = "FS86"
diff_table_surf_85_lh["version"] = "FS85"
diff_table_surf_76_lh["hemi"] = "lh"
diff_table_surf_75_lh["hemi"] = "lh"
diff_table_surf_65_lh["hemi"] = "lh"
diff_table_surf_87_lh["hemi"] = "lh"
diff_table_surf_86_lh["hemi"] = "lh"
diff_table_surf_85_lh["hemi"] = "lh"

diff_table_surf_76_rh.columns = diff_table_surf_76_rh.columns.str.replace(
    "FS7_6_diff_rh_", ""
)
diff_table_surf_76_rh.columns = diff_table_surf_76_rh.columns.str.replace("_surf", "")
diff_table_surf_75_rh.columns = diff_table_surf_75_rh.columns.str.replace(
    "FS7_5_diff_rh_", ""
)
diff_table_surf_75_rh.columns = diff_table_surf_75_rh.columns.str.replace("_surf", "")
diff_table_surf_65_rh.columns = diff_table_surf_65_rh.columns.str.replace(
    "FS6_5_diff_rh_", ""
)
diff_table_surf_65_rh.columns = diff_table_surf_65_rh.columns.str.replace("_surf", "")
diff_table_surf_76_lh.columns = diff_table_surf_76_lh.columns.str.replace(
    "FS7_6_diff_lh_", ""
)
diff_table_surf_76_lh.columns = diff_table_surf_76_lh.columns.str.replace("_surf", "")
diff_table_surf_75_lh.columns = diff_table_surf_75_lh.columns.str.replace(
    "FS7_5_diff_lh_", ""
)
diff_table_surf_75_lh.columns = diff_table_surf_75_lh.columns.str.replace("_surf", "")
diff_table_surf_65_lh.columns = diff_table_surf_65_lh.columns.str.replace(
    "FS6_5_diff_lh_", ""
)
diff_table_surf_65_lh.columns = diff_table_surf_65_lh.columns.str.replace("_surf", "")
diff_table_surf_87_rh.columns = diff_table_surf_87_rh.columns.str.replace(
    "FS8_7_diff_rh_", ""
)
diff_table_surf_87_rh.columns = diff_table_surf_87_rh.columns.str.replace("_surf", "")

diff_table_surf_86_rh.columns = diff_table_surf_86_rh.columns.str.replace(
    "FS8_6_diff_rh_", ""
)
diff_table_surf_86_rh.columns = diff_table_surf_86_rh.columns.str.replace("_surf", "")

diff_table_surf_85_rh.columns = diff_table_surf_85_rh.columns.str.replace(
    "FS8_5_diff_rh_", ""
)
diff_table_surf_85_rh.columns = diff_table_surf_85_rh.columns.str.replace("_surf", "")


diff_table_surf_all = pd.concat(
    [
        diff_table_surf_76_rh,
        diff_table_surf_75_rh,
        diff_table_surf_65_rh,
        diff_table_surf_76_lh,
        diff_table_surf_75_lh,
        diff_table_surf_65_lh,
        diff_table_surf_87_rh,
        diff_table_surf_86_rh,
        diff_table_surf_85_rh,
        diff_table_surf_87_lh,
        diff_table_surf_86_lh,
        diff_table_surf_85_lh,
    ]
)

diff_table_surf_rh = pd.concat(
    [diff_table_surf_76_rh, diff_table_surf_75_rh, diff_table_surf_65_rh, diff_table_surf_87_rh, diff_table_surf_86_rh, diff_table_surf_85_rh]
)

diff_table_surf_lh = pd.concat(
    [diff_table_surf_76_lh, diff_table_surf_75_lh, diff_table_surf_65_lh, diff_table_surf_87_lh, diff_table_surf_86_lh, diff_table_surf_85_lh]
)

# create df with cortical thickness difference

diff_table_ct_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?_ct"), axis=1)
diff_table_ct_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?_ct"), axis=1)
diff_table_ct_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?_ct"), axis=1)
diff_table_ct_87_rh = diff_table_87_a.filter(regex=("FS8_7_diff_rh.+?_ct"), axis=1)
diff_table_ct_86_rh = diff_table_86_a.filter(regex=("FS8_6_diff_rh.+?_ct"), axis=1)
diff_table_ct_85_rh = diff_table_85_a.filter(regex=("FS8_5_diff_rh.+?_ct"), axis=1)

diff_table_ct_76_rh["version"] = "FS76"
diff_table_ct_75_rh["version"] = "FS75"
diff_table_ct_65_rh["version"] = "FS65"
diff_table_ct_87_rh["version"] = "FS87"
diff_table_ct_86_rh["version"] = "FS86"
diff_table_ct_85_rh["version"] = "FS85"
diff_table_ct_76_rh["hemi"] = "rh"
diff_table_ct_75_rh["hemi"] = "rh"
diff_table_ct_65_rh["hemi"] = "rh"
diff_table_ct_87_rh["hemi"] = "rh"
diff_table_ct_86_rh["hemi"] = "rh"
diff_table_ct_85_rh["hemi"] = "rh"

diff_table_ct_76_lh = sub_diff_table_76_a.filter(regex=("FS7_6_diff_lh.+?_ct"), axis=1)
diff_table_ct_75_lh = sub_diff_table_75_a.filter(regex=("FS7_5_diff_lh.+?_ct"), axis=1)
diff_table_ct_65_lh = sub_diff_table_65_a.filter(regex=("FS6_5_diff_lh.+?_ct"), axis=1)
diff_table_ct_87_lh = sub_diff_table_87_a.filter(regex=("FS8_7_diff_lh.+?_ct"), axis=1)
diff_table_ct_86_lh = sub_diff_table_86_a.filter(regex=("FS8_6_diff_lh.+?_ct"), axis=1)
diff_table_ct_85_lh = sub_diff_table_85_a.filter(regex=("FS8_5_diff_lh.+?_ct"), axis=1)

diff_table_ct_76_lh["version"] = "FS76"
diff_table_ct_75_lh["version"] = "FS75"
diff_table_ct_65_lh["version"] = "FS65"
diff_table_ct_87_lh["version"] = "FS87"
diff_table_ct_86_lh["version"] = "FS86"
diff_table_ct_85_lh["version"] = "FS85"
diff_table_ct_76_lh["hemi"] = "lh"
diff_table_ct_75_lh["hemi"] = "lh"
diff_table_ct_65_lh["hemi"] = "lh"
diff_table_ct_87_lh["hemi"] = "lh"
diff_table_ct_86_lh["hemi"] = "lh"
diff_table_ct_85_lh["hemi"] = "lh"

diff_table_ct_76_rh.columns = diff_table_ct_76_rh.columns.str.replace(
    "FS7_6_diff_rh_", ""
)
diff_table_ct_76_rh.columns = diff_table_ct_76_rh.columns.str.replace("_ct", "")
diff_table_ct_75_rh.columns = diff_table_ct_75_rh.columns.str.replace(
    "FS7_5_diff_rh_", ""
)
diff_table_ct_75_rh.columns = diff_table_ct_75_rh.columns.str.replace("_ct", "")
diff_table_ct_65_rh.columns = diff_table_ct_65_rh.columns.str.replace(
    "FS6_5_diff_rh_", ""
)
diff_table_ct_65_rh.columns = diff_table_ct_65_rh.columns.str.replace("_ct", "")
diff_table_ct_76_lh.columns = diff_table_ct_76_lh.columns.str.replace(
    "FS7_6_diff_lh_", ""
)
diff_table_ct_76_lh.columns = diff_table_ct_76_lh.columns.str.replace("_ct", "")
diff_table_ct_75_lh.columns = diff_table_ct_75_lh.columns.str.replace(
    "FS7_5_diff_lh_", ""
)
diff_table_ct_75_lh.columns = diff_table_ct_75_lh.columns.str.replace("_ct", "")
diff_table_ct_65_lh.columns = diff_table_ct_65_lh.columns.str.replace(
    "FS6_5_diff_lh_", ""
)
diff_table_ct_65_lh.columns = diff_table_ct_65_lh.columns.str.replace("_ct", "")
diff_table_ct_87_rh.columns = diff_table_ct_87_rh.columns.str.replace(
    "FS8_7_diff_rh_", ""
)
diff_table_ct_87_rh.columns = diff_table_ct_87_rh.columns.str.replace("_ct", "")

diff_table_ct_86_rh.columns = diff_table_ct_86_rh.columns.str.replace(
    "FS8_6_diff_rh_", ""
)
diff_table_ct_86_rh.columns = diff_table_ct_86_rh.columns.str.replace("_ct", "")

diff_table_ct_85_rh.columns = diff_table_ct_85_rh.columns.str.replace(
    "FS8_5_diff_rh_", ""
)
diff_table_ct_85_rh.columns = diff_table_ct_85_rh.columns.str.replace("_ct", "")
diff_table_ct_87_lh.columns = diff_table_ct_87_lh.columns.str.replace(
    "FS8_7_diff_lh_", ""
)
diff_table_ct_87_lh.columns = diff_table_ct_87_lh.columns.str.replace("_ct", "")

diff_table_ct_86_lh.columns = diff_table_ct_86_lh.columns.str.replace(
    "FS8_6_diff_lh_", ""
)
diff_table_ct_86_lh.columns = diff_table_ct_86_lh.columns.str.replace("_ct", "")

diff_table_ct_85_lh.columns = diff_table_ct_85_lh.columns.str.replace(
    "FS8_5_diff_lh_", ""
)
diff_table_ct_85_lh.columns = diff_table_ct_85_lh.columns.str.replace("_ct", "")

diff_table_ct_all = pd.concat(
    [
        diff_table_ct_76_rh,
        diff_table_ct_75_rh,
        diff_table_ct_65_rh,
        diff_table_ct_87_rh,
        diff_table_ct_86_rh,
        diff_table_ct_85_rh,
        diff_table_ct_76_lh,
        diff_table_ct_75_lh,
        diff_table_ct_65_lh,
        diff_table_ct_87_lh,
        diff_table_ct_86_lh,
        diff_table_ct_85_lh,
    ]
)

diff_table_ct_rh = pd.concat(
    [diff_table_ct_76_rh, diff_table_ct_75_rh, diff_table_ct_65_rh, diff_table_ct_87_rh, diff_table_ct_86_rh, diff_table_ct_85_rh]
)

diff_table_ct_lh = pd.concat(
    [diff_table_ct_76_lh, diff_table_ct_75_lh, diff_table_ct_65_lh, diff_table_ct_87_lh, diff_table_ct_86_lh, diff_table_ct_85_lh]
)

# %%
# subcortical plot


x = sub_diff_all
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)

x["version"].replace("FS87", "FS 8 vs 7", inplace=True)
x["version"].replace("FS86", "FS 8 vs 6", inplace=True)
x["version"].replace("FS85", "FS 8 vs 5", inplace=True)

fig = px.box(
    x,
    y=y_sub,
    color="version",
)


fig.add_trace(
    go.Scatter(
        x=y_sub,
        y=sub_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
    )
)

fig.add_trace(
    go.Scatter(
        x=y_sub,
        y=sub_between.iloc[1],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    )
)

fig.add_trace(
    go.Scatter(
        x=y_sub,
        y=sub_between.iloc[2],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    )
)

fig.add_trace(
    go.Scatter(
        x=y_sub,
        y=sub_between.iloc[3],
        mode="markers",
        marker_symbol="diamond",
        marker_size=10,
        marker_color="black",
        name="FS 8",
    )
)


fig.update_xaxes(title="", gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume (mm<sup>3</sup>)")
fig.update_yaxes(range=[-1000, 14000])
fig.update_layout(legend_title_text="Legend", font_size=12)
# fig.write_image(out("plot_sub_both_short_large.png"), width=1920, height=1080)
fig.show()

# %%
x = sub_diff_all.copy()
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x["version"].replace("FS87", "FS 8 vs 7", inplace=True)
x["version"].replace("FS86", "FS 8 vs 6", inplace=True)
x["version"].replace("FS85", "FS 8 vs 5", inplace=True)

x = x[
    [
        "CC_Anterior",
        "CC_Mid_Anterior",
        "CC_Central",
        "CC_Mid_Posterior",
        "CC_Posterior",
        "Optic-Chiasm",
        "Brain-Stem",
        "CSF",
        "5th-Ventricle",
        "4th-Ventricle",
        "3rd-Ventricle",
        "Right-choroid-plexus",
        "Left-choroid-plexus",
        "Right-vessel",
        "Left-vessel",
        "Right-VentralDC",
        "Left-VentralDC",
        "Right-Accumbens-area",
        "Left-Accumbens-area",
        "Right-Amygdala",
        "Left-Amygdala",
        "Right-Hippocampus",
        "Left-Hippocampus",
        "Right-Pallidum",
        "Left-Pallidum",
        "Right-Putamen",
        "Left-Putamen",
        "Right-Caudate",
        "Left-Caudate",
        "Right-Thalamus",
        "Left-Thalamus",
        "Right-Cerebellum-Cortex",
        "Left-Cerebellum-Cortex",
        "Right-Cerebellum-White-Matter",
        "Left-Cerebellum-White-Matter",
        "Right-Inf-Lat-Vent",
        "Left-Inf-Lat-Vent",
        "Right-Lateral-Ventricle",
        "Left-Lateral-Ventricle",
        "version",
    ]
]

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_sub,
        x=sub_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.add_trace(
    go.Scatter(
        y=y_sub,
        x=sub_between.iloc[1],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        name="FS 6",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_sub,
        x=sub_between.iloc[2],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        name="FS 7",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_sub,
        x=sub_between.iloc[3],
        mode="markers",
        marker_symbol="diamond",
        marker_size=10,
        marker_color="black",
        name="FS 8",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Subcortical regions", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Volume (mm<sup>3</sup>)")
fig.update_xaxes(range=[-1000, 14000])
fig.update_layout(legend_title_text="", font_size=20)

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

# fig.write_image(out("plot_sub_both_short_large.png"), width=1080, height=1920)
fig.show()

# %%
# plot volume

x = diff_table_vol_all
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)

fig = px.box(
    x,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[2],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[3],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[4],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=vol_between.iloc[5],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=2,
    col=1,
)


fig.update_xaxes(title="Regions", gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume<br> (mm<sup>3</sup>)")
fig.update_layout(legend_title_text="Legend")

# fig.write_image(out("plot_vol_both.png"), width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_vol_bil.png"/>

# %%
# plot volume rh

x = diff_table_vol_rh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[3],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[5],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (right hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Volume<br> (mm<sup>3</sup>)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

# fig.write_image(out("plot_vol_rh_large.png"), width=1080, height=2400)
fig.show()

# %%
# plot volume lh

x = diff_table_vol_lh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[2],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=vol_between.iloc[4],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (left hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Volume<br> (mm<sup>3</sup>)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_vol_lh_large.png"), width=1080, height=2400)
fig.show()

# %%
# plot surface area

x = diff_table_surf_all
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)


fig = px.box(
    x,
    y=y_cort,
    color="version",
    facet_row="hemi",
)


fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[2],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[3],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[4],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=surf_between.iloc[5],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=2,
    col=1,
)


fig.update_xaxes(title="Regions", gridwidth=1, griddash="dot")
fig.update_yaxes(title="Surface area<br> (mm<sup>2</sup>)")
fig.update_layout(legend_title_text="Legend")
# fig.write_image(out("plot_surf_both.png"), width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_surf_bil.png"/>

# %%
# plot surface area rh

x = diff_table_surf_rh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[3],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[5],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (right hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Surface area<br> (mm<sup>2</sup>)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_surf_rh_large.png"), width=1080, height=2400)
fig.show()

# %%
# plot surface area lh

x = diff_table_surf_lh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[2],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=surf_between.iloc[4],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (left hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Surface area<br> (mm<sup>2</sup>)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_surf_lh_large.png"), width=1080, height=2400)
fig.show()

# %%
# plot cortical thickness


x = diff_table_ct_all
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)


fig = px.box(
    x,
    y=y_cort,
    color="version",
    facet_row="hemi",
)


fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[2],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[3],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 6",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[4],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=y_cort,
        y=ct_between.iloc[5],
        mode="markers",
        marker_symbol="circle",
        marker_size=10,
        marker_color="black",
        name="FS 7",
    ),
    row=2,
    col=1,
)


fig.update_xaxes(title="Regions", gridwidth=1, griddash="dot")
fig.update_yaxes(title="Thickness<br> (mm)")
fig.update_layout(legend_title_text="Legend")
# fig.write_image(out("plot_ct_both.png"), width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_ct_bil.png"/>

# %%
# plot cortical thickness rh


x = diff_table_ct_rh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[1],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[3],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[5],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (right hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Thickness<br> (mm)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_ct_rh_large.png"), width=1080, height=2400)
fig.show()

# %%
# plot cortical thickness lh


x = diff_table_ct_lh
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)
x = x.drop(columns=["hemi"])

fig = px.box(
    data_frame=x,
    color="version",
    orientation="h",
)

for data in fig.data:
    data.legendgroup = "group1"
    data.legendgrouptitle = {"text": "Between-version"}

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[0],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 5",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[2],
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        marker_color="black",
        name="FS 6",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)

fig.add_trace(
    go.Scatter(
        y=y_cort,
        x=ct_between.iloc[4],
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        marker_color="black",
        name="FS 7",
        legendgroup="group2",
        legendgrouptitle_text="Between-subject",
        legend="legend2",
    )
)


fig.update_yaxes(title="Regions (left hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Thickness<br> (mm)")
fig.update_layout(legend_title_text="", font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)

fig.update_layout(
    legend2=dict(
        orientation="h",
        yanchor="top",
        y=1.03,
        xanchor="left",
        x=0,
        font_size=20,
        traceorder="normal",
    )
)
# fig.write_image(out("plot_ct_lh_large.png"), width=1080, height=2400)
fig.show()

# %% [markdown]
# # Results - clinical whole-brain
#
# ## FS 5 - whole-brain results
#
# ## Group analysis

# %%
# group differences

result_group = open(out("result_group_FS5.txt"), "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS5/group_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_group_FS5.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## Correlational analysis

# %%
# group differences

result_group = open(out("result_corr_FS5.txt"), "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS5/corr_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_corr_FS5.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## FS 6 - whole-brain results
#
# ## Group analysis

# %%
# group differences

result_group = open(out("result_group_FS6.txt"), "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS6/group_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_group_FS6.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## Correlational analysis

# %%
# group differences

result_group = open(out("result_corr_FS6.txt"), "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS6/corr_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_corr_FS6.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## FS 7 - whole-brain results
#
# ### Group analysis

# %%
# group differences

result_group = open(out("result_group_FS7.txt"), "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS7/group_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_group_FS7.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ### Correlational analysis

# %%
# group differences

result_group = open(out("result_corr_FS7.txt"), "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS7/corr_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_corr_FS7.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## FS 8 - whole-brain results
#
# ### Group analysis

# %%
# group differences

result_group = open(out("result_group_FS8.txt"), "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS8/group_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_group_FS8.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ### Correlational analysis

# %%
# group differences

result_group = open(out("result_corr_FS8.txt"), "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                str(DATA) + "/glm_results/FS8/corr_{time}_{hemi}_con/cache.th13.abs.sig.cluster.summary".format(
                    time=time, hemi=hemi, pair=pair
                ),
                "r",
            )
            always_print = False
            lines = file.readlines()
            result_group.write(time + "\n")
            result_group.write(pair + "_" + hemi + "\n")
            for line in lines:
                if always_print or "ClusterNo" in line:
                    # print(line)
                    result_group.write(line)
                    always_print = True

# %%
results_group = pd.read_csv(inp("result_corr_FS8.txt"),
    sep="\\s+",
    keep_default_na=False,
    na_values=" ",
    comment="#",
    names=[
        "ClusterNo",
        "Max",
        "VtxMax",
        "Size(mm^2)",
        "MNIX",
        "MNIY",
        "MNIZ",
        "CWP",
        "CWPLow",
        "CWPHi",
        "NVtxs",
        "WghtVtx",
        "Annot",
    ],
)
results_group

# %% [markdown]
# ## Vertex-wise results
#
# <img src="images/brain_baseline_corr.png"/>
#
# #### Vertex-wise correlation between UPDRS score and cortical thickness at baseline.
# <br>
#
# <img src="images/brain_long_corr.png"/>
#
# #### Vertex-wise correlation between the rate of change in UPDRS and the rate of change in cortical thickness. Correlations in FreeSurfer 5 and 7 were not significant.
# <br>
#
# <img src="images/brain_baseline_group.png"/>
#
# #### Group differences (HC vs PD-non-MCI) in cortical thickness at baseline. Differences in FreeSurfer 6 and 7 were not significant.
# <br>
#
# <img src="images/brain_long_group.png"/>
#
# #### Group differences (HC vs PD-non-MCI) in the rate of change in cortical thickness.

# %% [markdown]
# # QPN - replication sample
#
# Data in the QPN sample was preprocessed outside this notebook using the same preprocessing pipeline as the main sample.

# %%
QPN = pd.read_csv(str(DATA) + "/qpn/QPN_subjects.csv")[["PATNO", "sex", "age"]]

# %%
sex_f = QPN[QPN["sex"] == "F"]["PATNO"].nunique()
sex_m = QPN[QPN["sex"] == "M"]["PATNO"].nunique()
age_m = QPN["age"].mean()
age_sd = QPN["age"].std(numeric_only=True)

print(
    f"There are {sex_f} woman and {sex_m} men in the QPN sample (Mage = {age_m}; SDage ={age_sd})"
)

# %% [markdown]
# ## QC - preprocessed images 

# %%
import imageio as iio
from pathlib import Path

for ver in ["FS5", "FS6", "FS7"]:
    for view in ["axial", "sagittal", "coronal"]:
        images = list()
        for file in Path(f"{DATA}/qpn/segm/{ver}/{view}").iterdir():
            if not file.is_file():
                continue

            images.append(iio.imread(file))
            iio.mimsave(out(f"qpn/segm/{ver}/segm_{view}.gif"), images, duration=1)

# %% [markdown]
# ## FS5
#
# ### FS 5 axial view
# ![SegmentLocal](images/segm_qpn_FS5_axial.gif "segment")
#
# ### FS 5 coronal view
# ![SegmentLocal](images/segm_qpn_FS5_coronal.gif "segment")
#
# ### FS 5 sagittal view
# ![SegmentLocal](images/segm_qpn_FS5_sagittal.gif "segment")
#
# ## FS6
#
# ### FS 6 axial view
# ![SegmentLocal](images/segm_qpn_FS6_axial.gif "segment")
#
# ### FS 6 coronal view
# ![SegmentLocal](images/segm_qpn_FS6_coronal.gif "segment")
#
# ### FS 6 sagittal view
# ![SegmentLocal](images/segm_qpn_FS6_sagittal.gif "segment")
#
# ## FS7
#
# ### FS 7 axial view
# ![SegmentLocal](images/segm_qpn_FS7_axial.gif "segment")
#
# ### FS 7 coronal view
# ![SegmentLocal](images/segm_qpn_FS7_coronal.gif "segment")
#
# ### FS 7 sagittal view
# ![SegmentLocal](images/segm_qpn_FS7_sagittal.gif "segment")
#

# %% [markdown]
# ## MRIQC - QPN sample

# %%
qpn_qc = QPN

metrics = [
    "cjv",
    "cnr",
    "fber",
    "qi_1",
    "qi_2",
    "rpve_gm",
    "snr_gm",
    "snr_total",
    "snrd_gm",
    "snrd_total",
]

for subj in qpn_qc["PATNO"]:
    for metric in metrics:
        file = f"{DATA}/source_files/clinical/mriqc/output_qpn/sub-{subj}/ses-01/anat/sub-{subj}_T1w.json"

        with open(file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if line.find(f"{metric}") != -1:
                    # print(float(line.split()[1].rstrip(',')))
                    qpn_qc.loc[
                        qpn_qc["PATNO"] == subj,
                        metric,
                    ] = float(line.split()[1].rstrip(","))

# %%
from scipy.stats import ttest_ind

group1 = qpn_qc
group2 = df_qc[df_qc["dx_group"] == "HC"]

group_qpn_qc = {}

for metric in metrics:
    group_qpn_qc[f"t_{metric}"], group_qpn_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )
    group_qpn_qc[f"t_{metric}"], group_qpn_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )

# %%
from scipy.stats import ttest_ind

group1 = qpn_qc
group2 = df_qc.loc[df_qc["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]

group_qpn_qc = {}

for metric in metrics:
    group_qpn_qc[f"t_{metric}"], group_qpn_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )
    group_qpn_qc[f"t_{metric}"], group_qpn_qc[f"p_{metric}"] = ttest_ind(
        group1[f"{metric}"], group2[f"{metric}"], nan_policy="omit"
    )

# %%
group_qpn_qc

# %%
# extract structural measures

qpn_table = QPN

ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

ROIs_aparc = [
    "G_and_S_frontomargin",
    "G_and_S_occipital_inf",
    "G_and_S_paracentral",
    "G_and_S_subcentral",
    "G_and_S_transv_frontopol",
    "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral",
    "G_cuneus",
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
    "G_Ins_lg_and_S_cent_ins",
    "G_insular_short",
    "G_occipital_middle",
    "G_occipital_sup",
    "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual",
    "G_oc-temp_med-Parahip",
    "G_orbital",
    "G_pariet_inf-Angular",
    "G_pariet_inf-Supramar",
    "G_parietal_sup",
    "G_postcentral",
    "G_precentral",
    "G_precuneus",
    "G_rectus",
    "G_subcallosal",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    "G_temporal_inf",
    "G_temporal_middle",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Pole_occipital",
    "Pole_temporal",
    "S_calcarine",
    "S_central",
    "S_cingul-Marginalis",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
    "S_collat_transv_ant",
    "S_collat_transv_post",
    "S_front_inf",
    "S_front_middle",
    "S_front_sup",
    "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans",
    "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal",
    "S_occipital_ant",
    "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral",
    "S_orbital_med-olfact",
    "S_orbital-H_Shaped",
    "S_parieto_occipital",
    "S_pericallosal",
    "S_postcentral",
    "S_precentral-inf-part",
    "S_precentral-sup-part",
    "S_suborbital",
    "S_subparietal",
    "S_temporal_inf",
    "S_temporal_sup",
    "S_temporal_transverse",
]

# FS 6 has different ROI labels
ROIs_aparc_FS6 = [
    "G&S_frontomargin",
    "G&S_occipital_inf",
    "G&S_paracentral",
    "G&S_subcentral",
    "G&S_transv_frontopol",
    "G&S_cingul-Ant",
    "G&S_cingul-Mid-Ant",
    "G&S_cingul-Mid-Post",
    "G_Ins_lg&S_cent_ins",
    "S_intrapariet&P_trans",
    "S_oc_middle&Lunatus",
    "S_oc_sup&transversal",
    "S_oc-temp_med&Lingual",
]

for subj in qpn_table["PATNO"]:
    for version in ["FS5", "FS6", "FS7"]:
        # extract TIV
        file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/aseg.stats".format(
            subidd=subj, version=version
        )
        with open(file, "r") as fp:
            # read all lines in a list
            lines = fp.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find("Estimated Total Intracranial Volume") != -1:
                    qpn_table.loc[
                        qpn_table["PATNO"] == subj,
                        "TIV_{version}".format(version=version),
                    ] = float(line.split(",")[3])

                    # aseg_table["TIV{ses}".format(ses = session)] = float(out)

        # extract ROIs volume
        for roi in ROIs:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/aseg.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_" + version,
                        ] = float(line.split()[3])
        # volumes
        # extract cortical lh volumes
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical rh volumes
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_vol_" + version,
                        ] = float(line.split()[3])

        # surface area
        # extract cortical lh surface area
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical rh surface area
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_surf_" + version,
                        ] = float(line.split()[2])

        # cortical thickness
        # extract cortical lh cortical thickness
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_ct_" + version,
                        ] = float(line.split()[4])

        # extract cortical rh cortical thickness
        for roi in ROIs_aparc:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_ct_" + version,
                        ] = float(line.split()[4])


for subj in qpn_table["PATNO"]:
    for version in ["FS6"]:

        # extract cortical lh volume
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical rh volume
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical lh surface area
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical rh surface area
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical lh cortical thickness
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_lh_ct_" + version,
                        ] = float(line.split()[4])

        # extract cortical rh cortical thickness
        for roi in ROIs_aparc_FS6:

            file = str(DATA) + "/qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        qpn_table.loc[
                            qpn_table["PATNO"] == subj,
                            roi + "_rh_ct_" + version,
                        ] = float(line.split()[4])

# rename FS6 ROIs to match ROI labels in FS5 and 7
qpn_table.columns = qpn_table.columns.str.replace("&", "_and_")

# %%
# calculate V2-V1/V1 and take an absolute value

ROIs = [
    "Left-Lateral-Ventricle",
    "Left-Inf-Lat-Vent",
    "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex",
    "Left-Thalamus",
    "Left-Caudate",
    "Left-Putamen",
    "Left-Pallidum",
    "3rd-Ventricle",
    "4th-Ventricle",
    "Brain-Stem",
    "Left-Hippocampus",
    "Left-Amygdala",
    "CSF",
    "Left-Accumbens-area",
    "Left-VentralDC",
    "Left-vessel",
    "Left-choroid-plexus",
    "Right-Lateral-Ventricle",
    "Right-Inf-Lat-Vent",
    "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex",
    "Right-Thalamus",
    "Right-Caudate",
    "Right-Putamen",
    "Right-Pallidum",
    "Right-Hippocampus",
    "Right-Amygdala",
    "Right-Accumbens-area",
    "Right-VentralDC",
    "Right-vessel",
    "Right-choroid-plexus",
    "5th-Ventricle",
    "Optic-Chiasm",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
]

ROIs_bil = [
    "G_and_S_frontomargin",
    "G_and_S_occipital_inf",
    "G_and_S_paracentral",
    "G_and_S_subcentral",
    "G_and_S_transv_frontopol",
    "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant",
    "G_and_S_cingul-Mid-Post",
    "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral",
    "G_cuneus",
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
    "G_Ins_lg_and_S_cent_ins",
    "G_insular_short",
    "G_occipital_middle",
    "G_occipital_sup",
    "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual",
    "G_oc-temp_med-Parahip",
    "G_orbital",
    "G_pariet_inf-Angular",
    "G_pariet_inf-Supramar",
    "G_parietal_sup",
    "G_postcentral",
    "G_precentral",
    "G_precuneus",
    "G_rectus",
    "G_subcallosal",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    "G_temporal_inf",
    "G_temporal_middle",
    "Lat_Fis-ant-Horizont",
    "Lat_Fis-ant-Vertical",
    "Lat_Fis-post",
    "Pole_occipital",
    "Pole_temporal",
    "S_calcarine",
    "S_central",
    "S_cingul-Marginalis",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
    "S_collat_transv_ant",
    "S_collat_transv_post",
    "S_front_inf",
    "S_front_middle",
    "S_front_sup",
    "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans",
    "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal",
    "S_occipital_ant",
    "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual",
    "S_orbital_lateral",
    "S_orbital_med-olfact",
    "S_orbital-H_Shaped",
    "S_parieto_occipital",
    "S_pericallosal",
    "S_postcentral",
    "S_precentral-inf-part",
    "S_precentral-sup-part",
    "S_suborbital",
    "S_subparietal",
    "S_temporal_inf",
    "S_temporal_sup",
    "S_temporal_transverse",
]


for roi in ROIs:
    for subj in qpn_table["PATNO"]:
        # calculate (Vol 2 - Vol 1) / Vol 1
        qpn_table["FS7_6_pct_{roi}".format(roi=roi)] = (
            (
                qpn_table["{roi}_FS7".format(roi=roi)]
                - qpn_table["{roi}_FS6".format(roi=roi)]
            )
            / qpn_table["{roi}_FS6".format(roi=roi)]
        ).abs()
        qpn_table["FS7_5_pct_{roi}".format(roi=roi)] = (
            (
                qpn_table["{roi}_FS7".format(roi=roi)]
                - qpn_table["{roi}_FS5".format(roi=roi)]
            )
            / qpn_table["{roi}_FS5".format(roi=roi)]
        ).abs()
        qpn_table["FS6_5_pct_{roi}".format(roi=roi)] = (
            (
                qpn_table["{roi}_FS6".format(roi=roi)]
                - qpn_table["{roi}_FS5".format(roi=roi)]
            )
            / qpn_table["{roi}_FS5".format(roi=roi)]
        ).abs()

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        for subj in qpn_table["PATNO"]:
            for i in ["vol", "surf", "ct"]:
                # calculate (Vol 2 - Vol 1) / Vol 1
                qpn_table[
                    "FS7_6_pct_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)
                ] = (
                    (
                        qpn_table[
                            "{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)
                        ]
                        - qpn_table[
                            "{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)
                        ]
                    )
                    / qpn_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)]
                ).abs()
                qpn_table[
                    "FS7_5_pct_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)
                ] = (
                    (
                        qpn_table[
                            "{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)
                        ]
                        - qpn_table[
                            "{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)
                        ]
                    )
                    / qpn_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)]
                ).abs()
                qpn_table[
                    "FS6_5_pct_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)
                ] = (
                    (
                        qpn_table[
                            "{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)
                        ]
                        - qpn_table[
                            "{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)
                        ]
                    )
                    / qpn_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)]
                ).abs()

# %% [markdown]
# ## stats - software variability in QPN sample

# %%
# paired t-test

from scipy import stats

t_paired_76 = {}
t_paired_75 = {}
t_paired_65 = {}

for roi in ROIs:
    (
        t_paired_76["t_{roi}".format(roi=roi)],
        t_paired_76["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        qpn_table["{roi}_FS7".format(roi=roi)],
        qpn_table["{roi}_FS6".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_75["t_{roi}".format(roi=roi)],
        t_paired_75["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        qpn_table["{roi}_FS7".format(roi=roi)],
        qpn_table["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_paired_65["t_{roi}".format(roi=roi)],
        t_paired_65["p_{roi}".format(roi=roi)],
    ) = stats.ttest_rel(
        qpn_table["{roi}_FS6".format(roi=roi)],
        qpn_table["{roi}_FS5".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_paired_76["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_76["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                qpn_table["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                qpn_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_75["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_75["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                qpn_table["{roi}_{hemi}_{i}_FS7".format(roi=roi, hemi=hemi, i=i)],
                qpn_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )
            (
                t_paired_65["t_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
                t_paired_65["p_{hemi}_{roi}_{i}".format(roi=roi, hemi=hemi, i=i)],
            ) = stats.ttest_rel(
                qpn_table["{roi}_{hemi}_{i}_FS6".format(roi=roi, hemi=hemi, i=i)],
                qpn_table["{roi}_{hemi}_{i}_FS5".format(roi=roi, hemi=hemi, i=i)],
                nan_policy="omit",
            )

# temp = pd.DataFrame.from_dict(t_paired_76, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_paired_76.csv"))
# temp = pd.DataFrame.from_dict(t_paired_75, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_paired_75.csv"))
# temp = pd.DataFrame.from_dict(t_paired_65, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_paired_65.csv"))

# %%
# group differences QPN vs HC

group1 = qpn_table
group2 = aseg_table[aseg_table["dx_group"] == "HC"]

t_groups_76 = {}
t_groups_75 = {}
t_groups_65 = {}

for roi in ROIs:
    (
        t_groups_76["t_{roi}".format(roi=roi)],
        t_groups_76["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS7_6_pct_{roi}".format(roi=roi)],
        group2["FS7_6_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_75["t_{roi}".format(roi=roi)],
        t_groups_75["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS7_5_pct_{roi}".format(roi=roi)],
        group2["FS7_5_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )
    (
        t_groups_65["t_{roi}".format(roi=roi)],
        t_groups_65["p_{roi}".format(roi=roi)],
    ) = ttest_ind(
        group1["FS6_5_pct_{roi}".format(roi=roi)],
        group2["FS6_5_pct_{roi}".format(roi=roi)],
        nan_policy="omit",
    )

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            (
                t_groups_76["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_76["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS7_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS7_6_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_75["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_75["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS7_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS7_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )
            (
                t_groups_65["t_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                t_groups_65["p_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
            ) = ttest_ind(
                group1["FS6_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                group2["FS6_5_pct_{hemi}_{roi}_{i}".format(hemi=hemi, roi=roi, i=i)],
                nan_policy="omit",
            )


# temp = pd.DataFrame.from_dict(t_groups_76, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_groups_76.csv"))
# temp = pd.DataFrame.from_dict(t_groups_75, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_groups_75.csv"))
# temp = pd.DataFrame.from_dict(t_groups_65, orient = 'index')
# temp.to_csv(out("results_qpn_ttest_groups_65.csv"))

# %% [markdown]
# ## Sørensen–Dice coefficient variability in the replication sample

# %%
# build sets of significant results

set_76_vol = {}
set_75_vol = {}
set_65_vol = {}
set_76_surf = {}
set_75_surf = {}
set_65_surf = {}
set_76_ct = {}
set_75_ct = {}
set_65_ct = {}

for roi in ROIs:
    set_76_vol[f"p_{roi}"] = t_paired_76[f"p_{roi}"] < (0.05 / 187)
    set_75_vol[f"p_{roi}"] = t_paired_75[f"p_{roi}"] < (0.05 / 187)
    set_65_vol[f"p_{roi}"] = t_paired_65[f"p_{roi}"] < (0.05 / 187)

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        set_76_vol[f"{hemi}_{roi}_vol"] = t_paired_76[f"p_{hemi}_{roi}_vol"] < (
            0.05 / 187
        )
        set_75_vol[f"{hemi}_{roi}_vol"] = t_paired_75[f"p_{hemi}_{roi}_vol"] < (
            0.05 / 187
        )
        set_65_vol[f"{hemi}_{roi}_vol"] = t_paired_65[f"p_{hemi}_{roi}_vol"] < (
            0.05 / 187
        )

        set_76_surf[f"{hemi}_{roi}_surf"] = t_paired_76[f"p_{hemi}_{roi}_surf"] < (
            0.05 / 148
        )
        set_75_surf[f"{hemi}_{roi}_surf"] = t_paired_75[f"p_{hemi}_{roi}_surf"] < (
            0.05 / 148
        )
        set_65_surf[f"{hemi}_{roi}_surf"] = t_paired_65[f"p_{hemi}_{roi}_surf"] < (
            0.05 / 148
        )

        set_76_ct[f"{hemi}_{roi}_ct"] = t_paired_76[f"p_{hemi}_{roi}_ct"] < (0.05 / 148)
        set_75_ct[f"{hemi}_{roi}_ct"] = t_paired_75[f"p_{hemi}_{roi}_ct"] < (0.05 / 148)
        set_65_ct[f"{hemi}_{roi}_ct"] = t_paired_65[f"p_{hemi}_{roi}_ct"] < (0.05 / 148)

set_76_vol = pd.DataFrame.from_dict(set_76_vol, orient="index")
set_75_vol = pd.DataFrame.from_dict(set_75_vol, orient="index")
set_65_vol = pd.DataFrame.from_dict(set_65_vol, orient="index")

set_76_surf = pd.DataFrame.from_dict(set_76_surf, orient="index")
set_75_surf = pd.DataFrame.from_dict(set_75_surf, orient="index")
set_65_surf = pd.DataFrame.from_dict(set_65_surf, orient="index")

set_76_ct = pd.DataFrame.from_dict(set_76_ct, orient="index")
set_75_ct = pd.DataFrame.from_dict(set_75_ct, orient="index")
set_65_ct = pd.DataFrame.from_dict(set_65_ct, orient="index")

set_76_vol = set_76_vol.loc[set_76_vol[0] == True]
set_75_vol = set_75_vol.loc[set_75_vol[0] == True]
set_65_vol = set_65_vol.loc[set_65_vol[0] == True]
set_76_surf = set_76_surf.loc[set_76_surf[0] == True]
set_75_surf = set_75_surf.loc[set_75_surf[0] == True]
set_65_surf = set_65_surf.loc[set_65_surf[0] == True]
set_76_ct = set_76_ct.loc[set_76_ct[0] == True]
set_75_ct = set_75_ct.loc[set_75_ct[0] == True]
set_65_ct = set_65_ct.loc[set_65_ct[0] == True]


# calculate intersection between the sets

set_76_75_vol_inter = set_76_vol.index.intersection(set_75_vol.index)
set_76_65_vol_inter = set_76_vol.index.intersection(set_65_vol.index)
set_75_65_vol_inter = set_75_vol.index.intersection(set_65_vol.index)

set_76_75_surf_inter = set_76_surf.index.intersection(set_75_surf.index)
set_76_65_surf_inter = set_76_surf.index.intersection(set_65_surf.index)
set_75_65_surf_inter = set_75_surf.index.intersection(set_65_surf.index)

set_76_75_ct_inter = set_76_ct.index.intersection(set_75_ct.index)
set_76_65_ct_inter = set_76_ct.index.intersection(set_65_ct.index)
set_75_65_ct_inter = set_75_ct.index.intersection(set_65_ct.index)


# calculate coefficients

dice_76_75_vol = (2 * len(set_76_75_vol_inter)) / (len(set_76_vol) + len(set_75_vol))
dice_76_65_vol = (2 * len(set_76_65_vol_inter)) / (len(set_76_vol) + len(set_65_vol))
dice_75_65_vol = (2 * len(set_75_65_vol_inter)) / (len(set_75_vol) + len(set_65_vol))

dice_76_75_surf = (2 * len(set_76_75_surf_inter)) / (
    len(set_76_surf) + len(set_75_surf)
)
dice_76_65_surf = (2 * len(set_76_65_surf_inter)) / (
    len(set_76_surf) + len(set_65_surf)
)
dice_75_65_surf = (2 * len(set_75_65_surf_inter)) / (
    len(set_75_surf) + len(set_65_surf)
)

dice_76_75_ct = (2 * len(set_76_75_ct_inter)) / (len(set_76_ct) + len(set_75_ct))
dice_76_65_ct = (2 * len(set_76_65_ct_inter)) / (len(set_76_ct) + len(set_65_ct))
dice_75_65_ct = (2 * len(set_75_65_ct_inter)) / (len(set_75_ct) + len(set_65_ct))


print(
    str(len(set_76_vol) / 187)
    + " % regions differed in volume between FS7 and FS6\n"
    + str(len(set_75_vol) / 187)
    + " % regions differed in volume between FS7 and FS5\n"
    + str(len(set_65_vol) / 187)
    + " % regions differed in volume between FS6 and FS5\n"
    + str(len(set_76_surf) / 148)
    + " % regions differed in surface between FS7 and FS6\n"
    + str(len(set_75_surf) / 148)
    + " % regions differed in surface between FS7 and FS5\n"
    + str(len(set_65_surf) / 148)
    + " % regions differed in surface between FS6 and FS5\n"
    + str(len(set_76_ct) / 148)
    + " % regions differed in thickness between FS7 and FS6\n"
    + str(len(set_75_ct) / 148)
    + " % regions differed in thickness between FS7 and FS5\n"
    + str(len(set_65_ct) / 148)
    + " % regions differed in thickness between FS6 and FS5\n"
)


print(
    "The Sørensen–Dice coefficients for software variability are:\n Volume 76 vs 75 = "
    + str(dice_76_75_vol)
    + "\n Volume 76 vs 65 = "
    + str(dice_76_65_vol)
    + "\n Volume 75 vs 65 = "
    + str(dice_75_65_vol)
    + "\n Surface 76 vs 75 = "
    + str(dice_76_75_surf)
    + "\n Surface 76 vs 65 = "
    + str(dice_76_65_surf)
    + "\n Surface 75 vs 65 = "
    + str(dice_75_65_surf)
    + "\n Thickness 76 vs 75 = "
    + str(dice_76_75_ct)
    + "\n Thickness 76 vs 65 = "
    + str(dice_76_65_ct)
    + "\n Thickness 75 vs 65 = "
    + str(dice_75_65_ct)
)

# %% [markdown]
# ## Correlation between MRIQC and software variability - QPN sample

# %%
metrics = [
    "cjv",
    "cnr",
    "fber",
    "qi_1",
    "qi_2",
    "rpve_gm",
    "snr_gm",
    "snr_total",
    "snrd_gm",
    "snrd_total",
]

qc_table = qpn_qc

qc_corr_76 = {}
qc_corr_75 = {}
qc_corr_65 = {}

for i in ["vol", "surf", "ct"]:
    for roi in ROIs_bil:
        for hemi in ["lh", "rh"]:
            for m in metrics:
                qc_table_stat = qc_table.dropna(subset=f"FS7_6_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_76[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_76[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS7_6_pct_{hemi}_{roi}_{i}", f"{m}")

                qc_table_stat = qc_table.dropna(subset=f"FS7_5_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_75[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_75[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS7_5_pct_{hemi}_{roi}_{i}", f"{m}")

                qc_table_stat = qc_table.dropna(subset=f"FS6_5_pct_{hemi}_{roi}_{i}")
                qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
                (
                    qc_corr_65[f"r_{hemi}_{roi}_{i}_{m}"],
                    qc_corr_65[f"p_{hemi}_{roi}_{i}_{m}"],
                ) = correlation(qc_table_stat, f"FS6_5_pct_{hemi}_{roi}_{i}", f"{m}")

for roi in ROIs:
    for m in metrics:
        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS7_6_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_76[f"r_{roi}_{m}"], qc_corr_76[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS7_6_pct_{roi}", f"{m}"
        )

        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS7_5_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_75[f"r_{roi}_{m}"], qc_corr_75[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS7_5_pct_{roi}", f"{m}"
        )

        qc_table_stat = qc_table.replace(np.inf, np.nan)
        qc_table_stat = qc_table_stat.dropna(subset=f"FS6_5_pct_{roi}")
        qc_table_stat = qc_table_stat.dropna(subset=f"{m}")
        qc_corr_65[f"r_{roi}_{m}"], qc_corr_65[f"p_{roi}_{m}"] = correlation(
            qc_table_stat, f"FS6_5_pct_{roi}", f"{m}"
        )

# temp = pd.DataFrame.from_dict(qc_corr_76, orient = 'index')
# temp.to_csv(out("qc_qpn_corr_76.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv(out("qc_qpn_corr_75.csv"))
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv(out("qc_qpn_corr_65.csv"))

# %%
