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
dem = pd.read_csv(os.path.join(utils.study_files_dir, "Demographics.csv"))[
    ["PATNO", "SEX", "BIRTHDT"]
]

# Age at visit
age = pd.read_csv(os.path.join(utils.study_files_dir, "Age_at_visit.csv"))[
    ["PATNO", "EVENT_ID", "AGE_AT_VISIT"]
]

# Education
edu = pd.read_csv(os.path.join(utils.study_files_dir, "Socio-Economics.csv"))[
    ["PATNO", "EDUCYRS"]
]

# Diagnosis
diag = pd.read_csv(
    os.path.join(utils.study_files_dir, "Primary_Clinical_Diagnosis.csv")
)[["PATNO", "EVENT_ID", "PRIMDIAG", "OTHNEURO"]]


# Dx status
dx_status = pd.read_csv(os.path.join(utils.study_files_dir, "Participant_Status.csv"))[
    ["PATNO", "COHORT"]
]

# PD dx history / disease duration calc
pd_hist = pd.read_csv(os.path.join(utils.study_files_dir, "PD_Diagnosis_History.csv"))[
    ["PATNO", "EVENT_ID", "PDDXDT"]
]

# Cognitive Categorization
cog_cat = pd.read_csv(
    os.path.join(utils.study_files_dir, "Cognitive_Categorization.csv")
)[["PATNO", "EVENT_ID", "COGSTATE"]]

# UPDRS and Hoehh Yahr
# updrs = pd.read_csv(os.path.join(utils.study_files_dir, "MDS-UPDRS_Part_III.csv"))[
#     ["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY"]
# ]

# Clean UPDRS file. Impute missing ON/OFF values.
# It produces MDS_UPDRS_Part_III_clean.csv file
# from livingpark_utils.scripts import pd_status

updrs = pd.read_csv(
    os.path.join(utils.study_files_dir, "MDS_UPDRS_Part_III_clean.csv")
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
    os.path.join(utils.study_files_dir, "MDS_UPDRS_Part_III_clean.csv"),
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

mri = pd.read_csv(os.path.join(utils.study_files_dir, "MRI_info.csv"))
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
# calculate time difference between the visits (Duration T2-T1)

cohort["Study Date"] = pd.to_datetime(cohort["Study Date"])
cohort["Study Date_NX"] = pd.to_datetime(cohort["Study Date_NX"])


cohort["durationT2_T1"] = (
    cohort["Study Date_NX"] - cohort["Study Date"]
) / np.timedelta64(1, "M")

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
failed = pd.read_csv("failed.csv")

for i in failed["PATNO_id"]:
    cohort_stat = cohort_stat[cohort_stat["PATNO_id"].str.contains(f"{i}") == False]

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

pd_clinical["durationT2_T1"] = (
    pd_clinical["Study Date_NX"] - pd_clinical["Study Date"]
) / np.timedelta64(1, "M")

pd_clinical["durationT2_T1_y"] = pd_clinical["durationT2_T1"] / 12

# visit ID
pd_clinical["PATNO_id"] = (
    "sub-" + pd_clinical["PATNO"].astype(str) + "_ses-" + pd_clinical["EVENT_ID"]
)

# calculate the change of UPDRS score
pd_clinical["NP3TOT_change"] = pd_clinical["NP3TOT"] - pd_clinical["NP3TOT_NX"]

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

download_data = first_visit.append(second_visit)

download_data["PATNO_id"] = (
    "sub-" + download_data["PATNO"].astype(str) + "_ses-" + download_data["EVENT_ID"]
)

# download_data.to_csv('download_data.csv')

# %%
utils.download_missing_nifti_files(download_data, link_in_outputs=True)

# %% [markdown]
# We will compare PD cohorts from the computational stage and clinical (longitudinal) stage of analyses. Clinical sample is a subsample of the computational stample (i.e., no new subjects are added). However, there might be a difference which visit is considered as baseline or follow-up.

# %%
first_visit_clin = pd_clinical

second_visit_clin = first_visit_clin
second_visit_clin = second_visit_clin.drop(["Description", "EVENT_ID"], axis=1)
second_visit_clin.rename(
    columns={"Description_NX": "Description", "EVENT_ID_NX": "EVENT_ID"}, inplace=True
)

visit_clin = first_visit_clin.append(second_visit_clin)

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

visit_clin_download = pd.read_csv("clinical_diff.csv")

utils.download_missing_nifti_files(visit_clin_download, link_in_outputs=True)

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

# %%
# %load_ext slurm_magic

# %%
# save df with all timepoints as json

import json

download_data["PATNO_id"] = (
    "sub-" + download_data["PATNO"].astype(str) + "_ses-" + download_data["EVENT_ID"]
)

data_to_process = download_data.reset_index()
small_df = data_to_process[["PATNO_id", "File name"]]
json_data = small_df.to_json()
meta = json.loads(json_data)
with open("json_data.json", "w") as fout:
    json.dump(meta, fout, indent=4)

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
failed_long = pd.read_csv("failed_long.csv")

for i in failed_long["PATNO"]:
    stage_two = stage_two[stage_two["PATNO"].astype(str).str.contains(f"{i}") == False]

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
with open("json_data_base.json", "w") as fout:
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
with open("json_data_long.json", "w") as fout:
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

qdec_table.to_csv("qdec_long_groups.dat", sep=" ", index=False)

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
fsgd_cortThick_diff.to_csv(
    "fsgd_group_cort_PDnonMCI_HC_baseline.txt", sep="\t", index=False, header=None
)

# generate file header
with open("fsgd_group_desc_PDnonMCI_HC_baseline.txt", "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex \n"
    )


with open("fsgd_cort_group_HC_PDnonMCI_baseline.fsgd", "wb") as outfile:
    for f in [
        "fsgd_group_desc_PDnonMCI_HC_baseline.txt",
        "fsgd_group_cort_PDnonMCI_HC_baseline.txt",
    ]:
        with open(f, "rb") as infile:
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
fsgd_cortThick_diff.to_csv(
    "fsgd_group_cort_PDnonMCI_HC_long.txt", sep="\t", index=False, header=None
)

# generate file header
with open("fsgd_group_desc_PDnonMCI_HC_long.txt", "w") as f:
    f.write(
        "GroupDescriptorFile 1\nTitle GroupDifferences\nClass HC\nClass PD-non-MCI\nVariables age sex durationT2_T1_y\n"
    )


with open("fsgd_cort_group_HC_PDnonMCI_long.fsgd", "wb") as outfile:
    for f in [
        "fsgd_group_desc_PDnonMCI_HC_long.txt",
        "fsgd_group_cort_PDnonMCI_HC_long.txt",
    ]:
        with open(f, "rb") as infile:
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
        "fsgd_cortThick_mcat_%s.to_csv('fsgd_corr_%s_cort.txt', sep='\t', index=False, header=None)"
        % (group, group)
    )

    # generate file header
    with open("fsgd_corr_{group}_desc.txt".format(group=group), "w") as f:
        f.write(
            "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName NP3TOT\nClass {group}\nVariables NP3TOT age sex \n".format(
                group=group
            )
        )

    # generate fsgd file
    with open("fsgd_corr_{group}_group.fsgd".format(group=group), "wb") as outfile:
        for f in [
            "fsgd_corr_{group}_desc.txt".format(group=group),
            "fsgd_corr_{group}_cort.txt".format(group=group),
        ]:
            with open(f, "rb") as infile:
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
        "fsgd_cortThick_mcat_%s.to_csv('fsgd_corr_%s_cort_long.txt', sep='\t', index=False, header=None)"
        % (group, group)
    )

    # generate file header
    with open("fsgd_corr_{group}_desc_long.txt".format(group=group), "w") as f:
        f.write(
            "GroupDescriptorFile 1\nTitle GroupDifferences\nMeasurementName NP3TOT\nClass {group}\nVariables NP3TOT age sex durationT2_T1_y\n".format(
                group=group
            )
        )

    # generate fsgd file
    with open("fsgd_corr_{group}_group_long.fsgd".format(group=group), "wb") as outfile:
        for f in [
            "fsgd_corr_{group}_desc_long.txt".format(group=group),
            "fsgd_corr_{group}_cort_long.txt".format(group=group),
        ]:
            with open(f, "rb") as infile:
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

with open("con_group_HC_PDnonMCI_base.mtx", "w") as f:
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

with open("con_group_HC_PDnonMCI_long.mtx", "w") as f:
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

with open("con_corr_UPDRS_base.mtx", "w") as f:
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

with open("con_corr_UPDRS_long.mtx", "w") as f:
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
# # Quality control of preprocessed images

# %%
import imageio as iio
from pathlib import Path

for stage in ["clinical", "comput"]:
    for ver in ["FS5", "FS6", "FS7"]:
        for view in ["axial", "sagittal", "coronal"]:
            images = list()
            for file in Path(f"segm/{stage}/{ver}/{view}").iterdir():
                if not file.is_file():
                    continue

                images.append(iio.imread(file))
                iio.mimsave(
                    f"segm/{stage}/{ver}/segm_{stage}_{view}.gif", images, duration=1
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
df_data = cohort
df_data["PATNO_id"] = (
    "sub-" + df_data["PATNO"].astype(str) + "_ses-" + df_data["EVENT_ID"]
)

# exclude images due to the preprocessing failure
failed = pd.read_csv("failed.csv")

for i in failed["PATNO_id"]:
    df_data = df_data[df_data["PATNO_id"].str.contains(f"{i}") == False]

# %%
df_qc = df_data

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
        file = f"mriqc/output/sub-{subj}/ses-{event}/anat/sub-{subj}_T1w.json"

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
# extract structural measures

aseg_table = df_data

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

for subj in aseg_table["PATNO_id"]:
    for version in ["FS5", "FS6", "FS7"]:
        # extract TIV
        file = "stats/{version}/{subidd}/stats/aseg.stats".format(
            subidd=subj, version=version
        )
        with open(file, "r") as fp:
            # read all lines in a list
            lines = fp.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find("Estimated Total Intracranial Volume") != -1:
                    aseg_table.loc[
                        aseg_table["PATNO_id"] == subj,
                        "TIV_{version}".format(version=version),
                    ] = float(line.split(",")[3])

                    # aseg_table["TIV{ses}".format(ses = session)] = float(out)

        # extract ROIs volume
        for roi in ROIs:

            file = "stats/{version}/{subidd}/stats/aseg.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_" + version,
                        ] = float(line.split()[3])
        # volumes
        # extract cortical lh volumes
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical rh volumes
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_vol_" + version,
                        ] = float(line.split()[3])

        # surface area
        # extract cortical lh surface area
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical rh surface area
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_surf_" + version,
                        ] = float(line.split()[2])

        # cortical thickness
        # extract cortical lh cortical thickness
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_ct_" + version,
                        ] = float(line.split()[4])

        # extract cortical rh cortical thickness
        for roi in ROIs_aparc:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_ct_" + version,
                        ] = float(line.split()[4])

for subj in aseg_table["PATNO_id"]:
    for version in ["FS6"]:

        # extract cortical lh volume
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical rh volume
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_vol_" + version,
                        ] = float(line.split()[3])

        # extract cortical lh surface area
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical rh surface area
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_surf_" + version,
                        ] = float(line.split()[2])

        # extract cortical lh cortical thickness
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_lh_ct_" + version,
                        ] = float(line.split()[4])

        # extract cortical rh cortical thickness
        for roi in ROIs_aparc_FS6:

            file = "stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
                subidd=subj, version=version
            )
            with open(file, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    if line.find(roi) != -1:
                        aseg_table.loc[
                            aseg_table["PATNO_id"] == subj,
                            roi + "_rh_ct_" + version,
                        ] = float(line.split()[4])

# rename FS6 ROIs to match ROI labels in FS5 and 7
aseg_table.columns = aseg_table.columns.str.replace("&", "_and_")

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
    for subj in aseg_table["PATNO_id"]:
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

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        for subj in aseg_table["PATNO_id"]:
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

# %%
# calculate V1 - V2 diff and take an absolute value


for roi in ROIs:
    for subj in aseg_table["PATNO_id"]:
        # calculate (Vol 2 - Vol 1)
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

for roi in ROIs_bil:
    for hemi in ["lh", "rh"]:
        for subj in aseg_table["PATNO_id"]:
            for i in ["vol", "surf", "ct"]:
                # calculate (Vol 2 - Vol 1)
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

# temp = pd.DataFrame.from_dict(t_paired_76, orient = 'index')
# temp.to_csv("results_ttest_paired_76.csv")
# temp = pd.DataFrame.from_dict(t_paired_75, orient = 'index')
# temp.to_csv("results_ttest_paired_75.csv")
# temp = pd.DataFrame.from_dict(t_paired_65, orient = 'index')
# temp.to_csv("results_ttest_paired_65.csv")

# %% [markdown]
# ## stats - between group differences (PD vs HC) in software variability

# %%
from scipy.stats import ttest_ind

group1 = aseg_table.loc[aseg_table["dx_group"].isin(["PD-MCI", "PD-non-MCI"])]
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
# temp.to_csv("results_ttest_groups_76.csv")
# temp = pd.DataFrame.from_dict(t_groups_75, orient = 'index')
# temp.to_csv("results_ttest_groups_75.csv")
# temp = pd.DataFrame.from_dict(t_groups_65, orient = 'index')
# temp.to_csv("results_ttest_groups_65.csv")

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
# temp.to_csv("qc_corr_76.csv")
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv("qc_corr_75.csv")
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv("qc_corr_65.csv")

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
# temp.to_csv("qc_PD_corr_76.csv")
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv("qc_PD_corr_75.csv")
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv("qc_PD_corr_65.csv")

# %% [markdown]
# ## stats - Sørensen–Dice coefficient variability 

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

# %%
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

# %%
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

# %%
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

# %%
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
# ## Whole-brain Sørensen–Dice coefficients

# %% language="bash"
#
# for sub in subj;
# do
#     mri_seg_overlap data/FS7/$sub/mri/aparc.a2009s+aseg.mgz data/FS6/$sub/mri/aparc.a2009s+aseg.mgz -o stats/dice/dice_76_$sub.aparc2009.json
#     mri_seg_overlap data/FS7/$sub/mri/aparc+aseg.mgz data/FS6/$sub/mri/aparc+aseg.mgz -o stats/dice/dice_76_$sub.aparc.json
#     mri_seg_overlap data/FS7/$sub/mri/aseg.mgz data/FS6/$sub/mri/aseg.mgz -o stats/dice/dice_76_$sub.aseg.json
# done
#
# for sub in subj;
# do
#     mri_seg_overlap data/FS7/$sub/mri/aparc.a2009s+aseg.mgz data/FS5/$sub/mri/aparc.a2009s+aseg.mgz -o stats/dice/dice_75_$sub.aparc2009.json
#     mri_seg_overlap data/FS7/$sub/mri/aparc+aseg.mgz data/FS5/$sub/mri/aparc+aseg.mgz -o stats/dice/dice_75_$sub.aparc.json
#     mri_seg_overlap data/FS7/$sub/mri/aseg.mgz data/FS5/$sub/mri/aseg.mgz -o stats/dice/dice_75_$sub.aseg.json
# done
#
# for sub in subj;
# do
#     mri_seg_overlap data/FS6/$sub/mri/aparc.a2009s+aseg.mgz data/FS5/$sub/mri/aparc.a2009s+aseg.mgz -o stats/dice/dice_65_$sub.aparc2009.json
#     mri_seg_overlap data/FS6/$sub/mri/aparc+aseg.mgz data/FS5/$sub/mri/aparc+aseg.mgz -o stats/dice/dice_65_$sub.aparc.json
#     mri_seg_overlap data/FS6/$sub/mri/aseg.mgz data/FS5/$sub/mri/aseg.mgz -o stats/dice/dice_65_$sub.aseg.json
# done

# %%
import json

for subj in aseg_table["PATNO_id"]:
    for pair in ["76", "75", "65"]:
        for file in ["aseg", "aparc", "aparc2009"]:
            with open(
                "stats/dice/dice_{pair}_{subj}.{file}.json".format(
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
sd = dice_table.std()

dice_results = {}
dice_results["mean"] = mean
dice_results["sd"] = sd
dice_results = pd.DataFrame.from_dict(dice_results, orient="index")
dice_results = dice_results.transpose()
# dice_results.to_csv("dice_results.csv")

# %% [markdown]
# # stats - software variability separately in HC and PD cohorts

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

# temp = pd.DataFrame.from_dict(t_paired_76_pd, orient = 'index')
# temp.to_csv("results_ttest_paired_76_pd.csv")
# temp = pd.DataFrame.from_dict(t_paired_75_pd, orient = 'index')
# temp.to_csv("results_ttest_paired_75_pd.csv")
# temp = pd.DataFrame.from_dict(t_paired_65_pd, orient = 'index')
# temp.to_csv("results_ttest_paired_65_pd.csv")

# temp = pd.DataFrame.from_dict(t_paired_76_hc, orient = 'index')
# temp.to_csv("results_ttest_paired_76_hc.csv")
# temp = pd.DataFrame.from_dict(t_paired_75_hc, orient = 'index')
# temp.to_csv("results_ttest_paired_75_hc.csv")
# temp = pd.DataFrame.from_dict(t_paired_65_hc, orient = 'index')
# temp.to_csv("results_ttest_paired_65_hc.csv")

# %% [markdown]
# # Clinical analyses
# ## subcortical volume analysis

# %%
aseg_table2 = stage_two

aseg_table2["PATNO_base"] = "sub-" + aseg_table2["PATNO"].astype(str) + "_base"
aseg_table2["first_id"] = (
    aseg_table2["first_visit"] + ".long." + aseg_table2["PATNO_base"]
)
aseg_table2["second_id"] = (
    aseg_table2["second_visit"] + ".long." + aseg_table2["PATNO_base"]
)

# select only PD-non-MCI patients and HC
aseg_table2 = stage_two.loc[stage_two["dx_group"].isin(["HC", "PD-non-MCI"])]

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
    for version in ["FS5", "FS6", "FS7"]:
        # extract TIV
        file = "stats/{version}/{subj}/stats/aseg.stats".format(
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

            file = "stats/{version}/{subj}/stats/aseg.stats".format(
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
    for version in ["FS5", "FS6", "FS7"]:
        # extract TIV
        file = "stats/{version}/{subj}/stats/aseg.stats".format(
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

            file = "stats/{version}/{subj}/stats/aseg.stats".format(
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
    for version in ["FS5", "FS6", "FS7"]:
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
# partial correlation

from pingouin import partial_corr

ROI_PDnonMCI = aseg_table2.loc[aseg_table2["dx_group"] == "PD-non-MCI"]
corr_cov = {}

# baseline
for roi_bil in ROIs:
    for version in ["FS5", "FS6", "FS7"]:

        corr_temp = partial_corr(
            data=ROI_PDnonMCI,
            x="{roi}_{ver}_first".format(roi=roi_bil, ver=version),
            y="NP3TOT",
            covar=["AGE_AT_VISIT", "SEX"],
            method="pearson",
        )
        corr_cov["corr_n_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["n"][
            "pearson"
        ]
        corr_cov["corr_r_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp["r"][
            "pearson"
        ]
        corr_cov["corr_ci_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp[
            "CI95%"
        ]["pearson"]
        corr_cov["corr_p_" + version + "_" + roi_bil + "_UPDRS_base"] = corr_temp[
            "p-val"
        ]["pearson"]

# longitudinal
for roi_bil in ROIs:
    for version in ["FS5", "FS6", "FS7"]:
        corr_temp = partial_corr(
            data=ROI_PDnonMCI,
            x="{roi}_{ver}_change_pct".format(roi=roi_bil, ver=version),
            y="NP3TOT_change",
            covar=["AGE_AT_VISIT", "SEX", "durationT2_T1_y"],
            method="pearson",
        )
        corr_cov["corr_n_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["n"][
            "pearson"
        ]
        corr_cov["corr_r_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp["r"][
            "pearson"
        ]
        corr_cov["corr_ci_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp[
            "CI95%"
        ]["pearson"]
        corr_cov["corr_p_" + version + "_" + roi_bil + "_UPDRS_long"] = corr_temp[
            "p-val"
        ]["pearson"]

df_corr_cov = pd.DataFrame.from_dict(corr_cov, orient="index")
# df_corr_cov.to_csv("results_corr_subcortical_cov.csv")

# %%
## group analysis with covariates

from pingouin import ancova as pg_ancova

ROI_PDnonMCI = aseg_table2.loc[aseg_table2["dx_group"] == "PD-non-MCI"]
ROI_HC = aseg_table2.loc[aseg_table2["dx_group"] == "HC"]
HC_PDMCI = pd.concat([ROI_HC, ROI_PDnonMCI], ignore_index=True)

ancova = {}

# baseline
for roi in ROIs:
    for ver in ["FS5", "FS6", "FS7"]:
        temp = pg_ancova(
            data=HC_PDMCI,
            dv="{roi}_{ver}_first".format(roi=roi, ver=ver),
            between="dx_group",
            covar=["AGE_AT_VISIT", "SEX"],
        )
        ancova["ancova_F_" + ver + "_" + roi + "_base"] = temp["F"][0]
        ancova["ancova_p_" + ver + "_" + roi + "_base"] = temp["p-unc"][0]

# longitudinal
for roi in ROIs:
    for ver in ["FS5", "FS6", "FS7"]:

        temp = pg_ancova(
            data=HC_PDMCI,
            dv="{roi}_{ver}_change_pct".format(roi=roi, ver=ver),
            between="dx_group",
            covar=["AGE_AT_VISIT", "SEX", "durationT2_T1_y"],
        )
        ancova["ancova_F_" + ver + "_" + roi + "_long"] = temp["F"][0]
        ancova["ancova_p_" + ver + "_" + roi + "_long"] = temp["p-unc"][0]

df_ancova = pd.DataFrame.from_dict(ancova, orient="index")
# df_ancova.to_csv("results_ancova_subcortical_cov.csv")

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

table.add_row("", "r", "p", "r", "p", "r", "p")

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
        str(
            (corr_cov["corr_p_FS5_{rois}_UPDRS_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_r_FS6_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_p_FS6_{rois}_UPDRS_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_r_FS7_{rois}_UPDRS_base".format(rois=rois)]).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_p_FS7_{rois}_UPDRS_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
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

table.add_row("", "r", "p", "r", "p", "r", "p")

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
        str(
            (corr_cov["corr_p_FS5_{rois}_UPDRS_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_r_FS6_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_p_FS6_{rois}_UPDRS_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_r_FS7_{rois}_UPDRS_long".format(rois=rois)]).round(
                decimals=3
            )
        ),
        str(
            (corr_cov["corr_p_FS7_{rois}_UPDRS_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
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

table.add_row("", "F", "p", "F", "p", "F", "p")

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
        str(
            (ancova["ancova_p_FS5_{rois}_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str((ancova["ancova_F_FS6_{rois}_base".format(rois=rois)]).round(decimals=3)),
        str(
            (ancova["ancova_p_FS6_{rois}_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str((ancova["ancova_F_FS7_{rois}_base".format(rois=rois)]).round(decimals=3)),
        str(
            (ancova["ancova_p_FS7_{rois}_base".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
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

table.add_row("", "F", "p", "F", "p", "F", "p")

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
        str(
            (ancova["ancova_p_FS5_{rois}_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str((ancova["ancova_F_FS6_{rois}_long".format(rois=rois)]).round(decimals=3)),
        str(
            (ancova["ancova_p_FS6_{rois}_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
        str((ancova["ancova_F_FS7_{rois}_long".format(rois=rois)]).round(decimals=3)),
        str(
            (ancova["ancova_p_FS7_{rois}_long".format(rois=rois)] * 14).round(
                decimals=3
            )
        ),
    )


console = Console()
console.print(table)

# %% [markdown]
# # Data visualisation
#
# ## Computational analyses
# Dice coefficients

# %%
dice_table = aseg_table.filter(like="dice_")
mean = dice_table.mean()
sd = dice_table.std()

dice_results = {}
dice_results["mean"] = mean
dice_results["sd"] = sd
dice_results = pd.DataFrame.from_dict(dice_results, orient="index")
dice_results = dice_results.transpose()
# dice_results.to_csv("dice_results.csv")

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

dice_table_all = pd.concat(
    [
        dice_table_76_rh,
        dice_table_76_lh,
        dice_table_75_rh,
        dice_table_75_lh,
        dice_table_65_rh,
        dice_table_65_lh,
    ]
)

dice_table_lh = pd.concat([dice_table_76_lh, dice_table_75_lh, dice_table_65_lh])
dice_table_rh = pd.concat([dice_table_76_rh, dice_table_75_rh, dice_table_65_rh])

# dice_table_all.to_csv('dice_table_all.csv')

# %%
import plotly
import plotly.express as px
import plotly.graph_objects as go

y_dice = [
    "Cerebral-White-Matter",
    "Cerebral-Cortex",
    "Lateral-Ventricle",
    "Inf-Lat-Vent",
    "Cerebellum-White-Matter",
    "Cerebellum-Cortex",
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Hippocampus",
    "Amygdala",
    "Accumbens-area",
    "VentralDC",
    "vessel",
    "choroid-plexus",
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


fig = px.box(dice_table_all, y=y_dice, color="group", facet_row="hemi")


fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Sørensen–Dice coefficients")
fig.update_layout(legend_title_text="Legend")
# fig.write_image("plot_dice.png", width=2800, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_dice.png"/>

# %%
import plotly
import plotly.express as px
import plotly.graph_objects as go

x = dice_table_lh.copy()
x = x.drop(columns=["hemi", "undetermined"])

fig = px.box(
    data_frame=x,
    color="group",
    orientation="h",
)

fig.update_yaxes(title="Regions (left hemisphere)", gridwidth=1, griddash="dot")
fig.update_xaxes(title="Sørensen–Dice coefficients")
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
# fig.write_image("plot_dice_lh.png", width=1080, height=2700)
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
# fig.write_image("plot_dice_rh.png", width=1080, height=2700)
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

# %%
sub_table_coh = aseg_table.filter(like="dx_group")

sub_table_5 = pd.concat([sub_table_5_a, sub_table_coh], axis=1)
sub_table_6 = pd.concat([sub_table_6_a, sub_table_coh], axis=1)
sub_table_7 = pd.concat([sub_table_7_a, sub_table_coh], axis=1)

sub_table_5["version"] = "FS5"
sub_table_6["version"] = "FS6"
sub_table_7["version"] = "FS7"

# %%
sub_table_5.columns = sub_table_5.columns.str.replace("_FS5", "")
sub_table_6.columns = sub_table_6.columns.str.replace("_FS6", "")
sub_table_7.columns = sub_table_7.columns.str.replace("_FS7", "")

sub_all = pd.concat([sub_table_5, sub_table_6, sub_table_7])

# sub_all.to_csv('sub_all.csv')

# %% [markdown]
# ### volumes

# %%
vol_table_5_a = aseg_table.filter(like="_vol_FS5")
vol_table_6_a = aseg_table.filter(like="_vol_FS6")
vol_table_7_a = aseg_table.filter(like="_vol_FS7")

vol_table_coh = aseg_table.filter(like="dx_group")

vol_table_5 = pd.concat([vol_table_5_a, vol_table_coh], axis=1)
vol_table_6 = pd.concat([vol_table_6_a, vol_table_coh], axis=1)
vol_table_7 = pd.concat([vol_table_7_a, vol_table_coh], axis=1)

vol_table_5_lh = vol_table_5.filter(like="lh_vol_FS5")
vol_table_5_rh = vol_table_5.filter(like="rh_vol_FS5")
vol_table_6_lh = vol_table_6.filter(like="lh_vol_FS6")
vol_table_6_rh = vol_table_6.filter(like="rh_vol_FS6")
vol_table_7_lh = vol_table_7.filter(like="lh_vol_FS7")
vol_table_7_rh = vol_table_7.filter(like="rh_vol_FS7")

vol_table_5_coh = vol_table_5.filter(like="dx_group")
vol_table_6_coh = vol_table_6.filter(like="dx_group")
vol_table_7_coh = vol_table_7.filter(like="dx_group")

vol_table_5_lh = pd.concat([vol_table_5_lh, vol_table_5_coh], axis=1)
vol_table_6_lh = pd.concat([vol_table_6_lh, vol_table_6_coh], axis=1)
vol_table_7_lh = pd.concat([vol_table_7_lh, vol_table_7_coh], axis=1)
vol_table_5_rh = pd.concat([vol_table_5_rh, vol_table_5_coh], axis=1)
vol_table_6_rh = pd.concat([vol_table_6_rh, vol_table_6_coh], axis=1)
vol_table_7_rh = pd.concat([vol_table_7_rh, vol_table_7_coh], axis=1)

vol_table_5_lh["hemi"] = "lh"
vol_table_5_rh["hemi"] = "rh"
vol_table_6_lh["hemi"] = "lh"
vol_table_6_rh["hemi"] = "rh"
vol_table_7_lh["hemi"] = "lh"
vol_table_7_rh["hemi"] = "rh"

vol_table_5_lh["version"] = "FS5"
vol_table_5_rh["version"] = "FS5"
vol_table_6_lh["version"] = "FS6"
vol_table_6_rh["version"] = "FS6"
vol_table_7_lh["version"] = "FS7"
vol_table_7_rh["version"] = "FS7"

# %%
# rename columns
vol_table_5_lh.columns = vol_table_5_lh.columns.str.replace("_lh_vol_FS5", "")
vol_table_5_rh.columns = vol_table_5_rh.columns.str.replace("_rh_vol_FS5", "")
vol_table_6_lh.columns = vol_table_6_lh.columns.str.replace("_lh_vol_FS6", "")
vol_table_6_rh.columns = vol_table_6_rh.columns.str.replace("_rh_vol_FS6", "")
vol_table_7_lh.columns = vol_table_7_lh.columns.str.replace("_lh_vol_FS7", "")
vol_table_7_rh.columns = vol_table_7_rh.columns.str.replace("_rh_vol_FS7", "")

vol_all = pd.concat(
    [
        vol_table_5_lh,
        vol_table_5_rh,
        vol_table_6_lh,
        vol_table_6_rh,
        vol_table_7_lh,
        vol_table_7_rh,
    ]
)

# vol_all.to_csv('vol_all.csv')

# %% [markdown]
# ### cortical thickness

# %%
ct_table_5_a = aseg_table.filter(like="_ct_FS5")
ct_table_6_a = aseg_table.filter(like="_ct_FS6")
ct_table_7_a = aseg_table.filter(like="_ct_FS7")

ct_table_coh = aseg_table.filter(like="dx_group")

ct_table_5 = pd.concat([ct_table_5_a, ct_table_coh], axis=1)
ct_table_6 = pd.concat([ct_table_6_a, ct_table_coh], axis=1)
ct_table_7 = pd.concat([ct_table_7_a, ct_table_coh], axis=1)

ct_table_5_lh = ct_table_5.filter(like="lh_ct_FS5")
ct_table_5_rh = ct_table_5.filter(like="rh_ct_FS5")
ct_table_6_lh = ct_table_6.filter(like="lh_ct_FS6")
ct_table_6_rh = ct_table_6.filter(like="rh_ct_FS6")
ct_table_7_lh = ct_table_7.filter(like="lh_ct_FS7")
ct_table_7_rh = ct_table_7.filter(like="rh_ct_FS7")

ct_table_5_coh = ct_table_5.filter(like="dx_group")
ct_table_6_coh = ct_table_6.filter(like="dx_group")
ct_table_7_coh = ct_table_7.filter(like="dx_group")

ct_table_5_lh = pd.concat([ct_table_5_lh, ct_table_5_coh], axis=1)
ct_table_6_lh = pd.concat([ct_table_6_lh, ct_table_6_coh], axis=1)
ct_table_7_lh = pd.concat([ct_table_7_lh, ct_table_7_coh], axis=1)
ct_table_5_rh = pd.concat([ct_table_5_rh, ct_table_5_coh], axis=1)
ct_table_6_rh = pd.concat([ct_table_6_rh, ct_table_6_coh], axis=1)
ct_table_7_rh = pd.concat([ct_table_7_rh, ct_table_7_coh], axis=1)

ct_table_5_lh["hemi"] = "lh"
ct_table_5_rh["hemi"] = "rh"
ct_table_6_lh["hemi"] = "lh"
ct_table_6_rh["hemi"] = "rh"
ct_table_7_lh["hemi"] = "lh"
ct_table_7_rh["hemi"] = "rh"

ct_table_5_lh["version"] = "FS5"
ct_table_5_rh["version"] = "FS5"
ct_table_6_lh["version"] = "FS6"
ct_table_6_rh["version"] = "FS6"
ct_table_7_lh["version"] = "FS7"
ct_table_7_rh["version"] = "FS7"

# %%
# rename columns
ct_table_5_lh.columns = ct_table_5_lh.columns.str.replace("_lh_ct_FS5", "")
ct_table_5_rh.columns = ct_table_5_rh.columns.str.replace("_rh_ct_FS5", "")
ct_table_6_lh.columns = ct_table_6_lh.columns.str.replace("_lh_ct_FS6", "")
ct_table_6_rh.columns = ct_table_6_rh.columns.str.replace("_rh_ct_FS6", "")
ct_table_7_lh.columns = ct_table_7_lh.columns.str.replace("_lh_ct_FS7", "")
ct_table_7_rh.columns = ct_table_7_rh.columns.str.replace("_rh_ct_FS7", "")

ct_all = pd.concat(
    [
        ct_table_5_lh,
        ct_table_5_rh,
        ct_table_6_lh,
        ct_table_6_rh,
        ct_table_7_lh,
        ct_table_7_rh,
    ]
)


ct_table_rh = pd.concat([ct_table_5_rh, ct_table_6_rh, ct_table_7_rh])
ct_table_lh = pd.concat([ct_table_5_lh, ct_table_6_lh, ct_table_7_lh])


# ct_all.to_csv('ct_all.csv')

# %% [markdown]
# ### surface area

# %%
surf_table_5_a = aseg_table.filter(like="_surf_FS5")
surf_table_6_a = aseg_table.filter(like="_surf_FS6")
surf_table_7_a = aseg_table.filter(like="_surf_FS7")

surf_table_coh = aseg_table.filter(like="dx_group")

surf_table_5 = pd.concat([surf_table_5_a, surf_table_coh], axis=1)
surf_table_6 = pd.concat([surf_table_6_a, surf_table_coh], axis=1)
surf_table_7 = pd.concat([surf_table_7_a, surf_table_coh], axis=1)

surf_table_5_lh = surf_table_5.filter(like="lh_surf_FS5")
surf_table_5_rh = surf_table_5.filter(like="rh_surf_FS5")
surf_table_6_lh = surf_table_6.filter(like="lh_surf_FS6")
surf_table_6_rh = surf_table_6.filter(like="rh_surf_FS6")
surf_table_7_lh = surf_table_7.filter(like="lh_surf_FS7")
surf_table_7_rh = surf_table_7.filter(like="rh_surf_FS7")

surf_table_5_coh = surf_table_5.filter(like="dx_group")
surf_table_6_coh = surf_table_6.filter(like="dx_group")
surf_table_7_coh = surf_table_7.filter(like="dx_group")

surf_table_5_lh = pd.concat([surf_table_5_lh, surf_table_5_coh], axis=1)
surf_table_6_lh = pd.concat([surf_table_6_lh, surf_table_6_coh], axis=1)
surf_table_7_lh = pd.concat([surf_table_7_lh, surf_table_7_coh], axis=1)
surf_table_5_rh = pd.concat([surf_table_5_rh, surf_table_5_coh], axis=1)
surf_table_6_rh = pd.concat([surf_table_6_rh, surf_table_6_coh], axis=1)
surf_table_7_rh = pd.concat([surf_table_7_rh, surf_table_7_coh], axis=1)

surf_table_5_lh["hemi"] = "lh"
surf_table_5_rh["hemi"] = "rh"
surf_table_6_lh["hemi"] = "lh"
surf_table_6_rh["hemi"] = "rh"
surf_table_7_lh["hemi"] = "lh"
surf_table_7_rh["hemi"] = "rh"

surf_table_5_lh["version"] = "FS5"
surf_table_5_rh["version"] = "FS5"
surf_table_6_lh["version"] = "FS6"
surf_table_6_rh["version"] = "FS6"
surf_table_7_lh["version"] = "FS7"
surf_table_7_rh["version"] = "FS7"

# %%
# rename columns
surf_table_5_lh.columns = surf_table_5_lh.columns.str.replace("_lh_surf_FS5", "")
surf_table_5_rh.columns = surf_table_5_rh.columns.str.replace("_rh_surf_FS5", "")
surf_table_6_lh.columns = surf_table_6_lh.columns.str.replace("_lh_surf_FS6", "")
surf_table_6_rh.columns = surf_table_6_rh.columns.str.replace("_rh_surf_FS6", "")
surf_table_7_lh.columns = surf_table_7_lh.columns.str.replace("_lh_surf_FS7", "")
surf_table_7_rh.columns = surf_table_7_rh.columns.str.replace("_rh_surf_FS7", "")

surf_all = pd.concat(
    [
        surf_table_5_lh,
        surf_table_5_rh,
        surf_table_6_lh,
        surf_table_6_rh,
        surf_table_7_lh,
        surf_table_7_rh,
    ]
)

surf_table_rh = pd.concat([surf_table_5_rh, surf_table_6_rh, surf_table_7_rh])
surf_table_lh = pd.concat([surf_table_5_lh, surf_table_6_lh, surf_table_7_lh])

# surf_all.to_csv('surf_all.csv')

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
# fig.write_image("plot_sub.png", width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_est_sub.png"/>

# %%
fig = px.box(
    vol_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume<br> (mm<sup>3</sup>)")
# fig.write_image("plot_vol.png", width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_est_vol.png"/>

# %%
fig = px.box(
    surf_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Surface area<br> (mm<sup>2</sup>)")
# fig.write_image("plot_surf.png", width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_est_surf.png"/>

# %%
fig = px.box(
    ct_all,
    y=y_cort,
    color="version",
    facet_row="hemi",
)

fig.update_xaxes(gridwidth=1, griddash="dot")
fig.update_yaxes(title="Thickness<br> (mm)")
# fig.write_image("plot_ct.png", width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_est_ct.png"/>

# %% [markdown]
# # Plots - between-version and between-subject variability

# %%
# sd across subjects

sub5_between = sub_all.loc[(sub_all["version"] == "FS5")]
sub5_between = sub5_between.std()
sub5_between["version"] = "FS5"


sub6_between = sub_all.loc[(sub_all["version"] == "FS6")]
sub6_between = sub6_between.std()
sub6_between["version"] = "FS6"


sub7_between = sub_all.loc[(sub_all["version"] == "FS7")]
sub7_between = sub7_between.std()
sub7_between["version"] = "FS7"


sub5_between = pd.DataFrame(data=sub5_between).T
sub6_between = pd.DataFrame(data=sub6_between).T
sub7_between = pd.DataFrame(data=sub7_between).T

sub_between = pd.concat([sub5_between, sub6_between, sub7_between])

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
vol5lh_between = vol5lh_between.std()
vol5lh_between["version"] = "FS5"
vol5lh_between["hemi"] = "lh"

vol5rh_between = vol_all.loc[(vol_all["version"] == "FS5") & (vol_all["hemi"] == "rh")]
vol5rh_between = vol5rh_between.std()
vol5rh_between["version"] = "FS5"
vol5rh_between["hemi"] = "rh"

vol6lh_between = vol_all.loc[(vol_all["version"] == "FS6") & (vol_all["hemi"] == "lh")]
vol6lh_between = vol6lh_between.std()
vol6lh_between["version"] = "FS6"
vol6lh_between["hemi"] = "lh"

vol6rh_between = vol_all.loc[(vol_all["version"] == "FS6") & (vol_all["hemi"] == "rh")]
vol6rh_between = vol6rh_between.std()
vol6rh_between["version"] = "FS6"
vol6rh_between["hemi"] = "rh"

vol7lh_between = vol_all.loc[(vol_all["version"] == "FS7") & (vol_all["hemi"] == "lh")]
vol7lh_between = vol7lh_between.std()
vol7lh_between["version"] = "FS7"
vol7lh_between["hemi"] = "lh"

vol7rh_between = vol_all.loc[(vol_all["version"] == "FS7") & (vol_all["hemi"] == "rh")]
vol7rh_between = vol7rh_between.std()
vol7rh_between["version"] = "FS7"
vol7rh_between["hemi"] = "rh"


vol5lh_between = pd.DataFrame(data=vol5lh_between).T
vol5rh_between = pd.DataFrame(data=vol5rh_between).T
vol6lh_between = pd.DataFrame(data=vol6lh_between).T
vol6rh_between = pd.DataFrame(data=vol6rh_between).T
vol7lh_between = pd.DataFrame(data=vol7lh_between).T
vol7rh_between = pd.DataFrame(data=vol7rh_between).T

vol_between = pd.concat(
    [
        vol5lh_between,
        vol5rh_between,
        vol6lh_between,
        vol6rh_between,
        vol7lh_between,
        vol7rh_between,
    ]
)

# surf
surf5lh_between = surf_all.loc[
    (surf_all["version"] == "FS5") & (surf_all["hemi"] == "lh")
]
surf5lh_between = surf5lh_between.std()
surf5lh_between["version"] = "FS5"
surf5lh_between["hemi"] = "lh"

surf5rh_between = surf_all.loc[
    (surf_all["version"] == "FS5") & (surf_all["hemi"] == "rh")
]
surf5rh_between = surf5rh_between.std()
surf5rh_between["version"] = "FS5"
surf5rh_between["hemi"] = "rh"

surf6lh_between = surf_all.loc[
    (surf_all["version"] == "FS6") & (surf_all["hemi"] == "lh")
]
surf6lh_between = surf6lh_between.std()
surf6lh_between["version"] = "FS6"
surf6lh_between["hemi"] = "lh"

surf6rh_between = surf_all.loc[
    (surf_all["version"] == "FS6") & (surf_all["hemi"] == "rh")
]
surf6rh_between = surf6rh_between.std()
surf6rh_between["version"] = "FS6"
surf6rh_between["hemi"] = "rh"

surf7lh_between = surf_all.loc[
    (surf_all["version"] == "FS7") & (surf_all["hemi"] == "lh")
]
surf7lh_between = surf7lh_between.std()
surf7lh_between["version"] = "FS7"
surf7lh_between["hemi"] = "lh"

surf7rh_between = surf_all.loc[
    (surf_all["version"] == "FS7") & (surf_all["hemi"] == "rh")
]
surf7rh_between = surf7rh_between.std()
surf7rh_between["version"] = "FS7"
surf7rh_between["hemi"] = "rh"


surf5lh_between = pd.DataFrame(data=surf5lh_between).T
surf5rh_between = pd.DataFrame(data=surf5rh_between).T
surf6lh_between = pd.DataFrame(data=surf6lh_between).T
surf6rh_between = pd.DataFrame(data=surf6rh_between).T
surf7lh_between = pd.DataFrame(data=surf7lh_between).T
surf7rh_between = pd.DataFrame(data=surf7rh_between).T

surf_between = pd.concat(
    [
        surf5lh_between,
        surf5rh_between,
        surf6lh_between,
        surf6rh_between,
        surf7lh_between,
        surf7rh_between,
    ]
)

# ct
ct5lh_between = ct_all.loc[(ct_all["version"] == "FS5") & (ct_all["hemi"] == "lh")]
ct5lh_between = ct5lh_between.std()
ct5lh_between["version"] = "FS5"
ct5lh_between["hemi"] = "lh"

ct5rh_between = ct_all.loc[(ct_all["version"] == "FS5") & (ct_all["hemi"] == "rh")]
ct5rh_between = ct5rh_between.std()
ct5rh_between["version"] = "FS5"
ct5rh_between["hemi"] = "rh"

ct6lh_between = ct_all.loc[(ct_all["version"] == "FS6") & (ct_all["hemi"] == "lh")]
ct6lh_between = ct6lh_between.std()
ct6lh_between["version"] = "FS6"
ct6lh_between["hemi"] = "lh"

ct6rh_between = ct_all.loc[(ct_all["version"] == "FS6") & (ct_all["hemi"] == "rh")]
ct6rh_between = ct6rh_between.std()
ct6rh_between["version"] = "FS6"
ct6rh_between["hemi"] = "rh"

ct7lh_between = ct_all.loc[(ct_all["version"] == "FS7") & (ct_all["hemi"] == "lh")]
ct7lh_between = ct7lh_between.std()
ct7lh_between["version"] = "FS7"
ct7lh_between["hemi"] = "lh"

ct7rh_between = ct_all.loc[(ct_all["version"] == "FS7") & (ct_all["hemi"] == "rh")]
ct7rh_between = ct7rh_between.std()
ct7rh_between["version"] = "FS7"
ct7rh_between["hemi"] = "rh"


ct5lh_between = pd.DataFrame(data=ct5lh_between).T
ct5rh_between = pd.DataFrame(data=ct5rh_between).T
ct6lh_between = pd.DataFrame(data=ct6lh_between).T
ct6rh_between = pd.DataFrame(data=ct6rh_between).T
ct7lh_between = pd.DataFrame(data=ct7lh_between).T
ct7rh_between = pd.DataFrame(data=ct7rh_between).T

ct_between = pd.concat(
    [
        ct5lh_between,
        ct5rh_between,
        ct6lh_between,
        ct6rh_between,
        ct7lh_between,
        ct7rh_between,
    ]
)

# %%
# difference between versions

sub_diff_table_76_a = aseg_table.filter(like="FS7_6_diff_")
sub_diff_table_75_a = aseg_table.filter(like="FS7_5_diff_")
sub_diff_table_65_a = aseg_table.filter(like="FS6_5_diff_")

sub_diff_table_coh = aseg_table.filter(like="dx_group")

sub_diff_table_75 = pd.concat([sub_diff_table_75_a, sub_diff_table_coh], axis=1)
sub_diff_table_76 = pd.concat([sub_diff_table_76_a, sub_diff_table_coh], axis=1)
sub_diff_table_65 = pd.concat([sub_diff_table_65_a, sub_diff_table_coh], axis=1)

sub_diff_table_76["version"] = "FS76"
sub_diff_table_75["version"] = "FS75"
sub_diff_table_65["version"] = "FS65"

# %%
sub_diff_table_76.columns = sub_diff_table_76.columns.str.replace("FS7_6_diff_", "")
sub_diff_table_75.columns = sub_diff_table_75.columns.str.replace("FS7_5_diff_", "")
sub_diff_table_65.columns = sub_diff_table_65.columns.str.replace("FS6_5_diff_", "")

# %%
sub_diff_table_76 = sub_diff_table_76[
    [
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
]

sub_diff_table_75 = sub_diff_table_75[
    [
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
]

sub_diff_table_65 = sub_diff_table_65[
    [
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
]


sub_diff_all = pd.concat([sub_diff_table_76, sub_diff_table_75, sub_diff_table_65])


# sub_diff_all.to_csv('sub_diff_all.csv')

# %%
# create df with difference

diff_table_76_a = aseg_table.filter(like="FS7_6_diff_")
diff_table_75_a = aseg_table.filter(like="FS7_5_diff_")
diff_table_65_a = aseg_table.filter(like="FS6_5_diff_")

# create df with volume difference

diff_table_vol_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?vol"), axis=1)
diff_table_vol_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?vol"), axis=1)
diff_table_vol_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?vol"), axis=1)
diff_table_vol_76_rh["version"] = "FS76"
diff_table_vol_75_rh["version"] = "FS75"
diff_table_vol_65_rh["version"] = "FS65"
diff_table_vol_76_rh["hemi"] = "rh"
diff_table_vol_75_rh["hemi"] = "rh"
diff_table_vol_65_rh["hemi"] = "rh"

diff_table_vol_76_lh = sub_diff_table_76_a.filter(regex=("FS7_6_diff_lh.+?vol"), axis=1)
diff_table_vol_75_lh = sub_diff_table_75_a.filter(regex=("FS7_5_diff_lh.+?vol"), axis=1)
diff_table_vol_65_lh = sub_diff_table_65_a.filter(regex=("FS6_5_diff_lh.+?vol"), axis=1)
diff_table_vol_76_lh["version"] = "FS76"
diff_table_vol_75_lh["version"] = "FS75"
diff_table_vol_65_lh["version"] = "FS65"
diff_table_vol_76_lh["hemi"] = "lh"
diff_table_vol_75_lh["hemi"] = "lh"
diff_table_vol_65_lh["hemi"] = "lh"

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

diff_table_vol_all = pd.concat(
    [
        diff_table_vol_76_rh,
        diff_table_vol_75_rh,
        diff_table_vol_65_rh,
        diff_table_vol_76_lh,
        diff_table_vol_75_lh,
        diff_table_vol_65_lh,
    ]
)

diff_table_vol_rh = pd.concat(
    [diff_table_vol_76_rh, diff_table_vol_75_rh, diff_table_vol_65_rh]
)
diff_table_vol_lh = pd.concat(
    [diff_table_vol_76_lh, diff_table_vol_75_lh, diff_table_vol_65_lh]
)

# create df with surface area difference

diff_table_surf_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?surf"), axis=1)
diff_table_surf_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?surf"), axis=1)
diff_table_surf_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?surf"), axis=1)
diff_table_surf_76_rh["version"] = "FS76"
diff_table_surf_75_rh["version"] = "FS75"
diff_table_surf_65_rh["version"] = "FS65"
diff_table_surf_76_rh["hemi"] = "rh"
diff_table_surf_75_rh["hemi"] = "rh"
diff_table_surf_65_rh["hemi"] = "rh"

diff_table_surf_76_lh = sub_diff_table_76_a.filter(
    regex=("FS7_6_diff_lh.+?surf"), axis=1
)
diff_table_surf_75_lh = sub_diff_table_75_a.filter(
    regex=("FS7_5_diff_lh.+?surf"), axis=1
)
diff_table_surf_65_lh = sub_diff_table_65_a.filter(
    regex=("FS6_5_diff_lh.+?surf"), axis=1
)
diff_table_surf_76_lh["version"] = "FS76"
diff_table_surf_75_lh["version"] = "FS75"
diff_table_surf_65_lh["version"] = "FS65"
diff_table_surf_76_lh["hemi"] = "lh"
diff_table_surf_75_lh["hemi"] = "lh"
diff_table_surf_65_lh["hemi"] = "lh"

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

diff_table_surf_all = pd.concat(
    [
        diff_table_surf_76_rh,
        diff_table_surf_75_rh,
        diff_table_surf_65_rh,
        diff_table_surf_76_lh,
        diff_table_surf_75_lh,
        diff_table_surf_65_lh,
    ]
)

diff_table_surf_rh = pd.concat(
    [diff_table_surf_76_rh, diff_table_surf_75_rh, diff_table_surf_65_rh]
)

diff_table_surf_lh = pd.concat(
    [diff_table_surf_76_lh, diff_table_surf_75_lh, diff_table_surf_65_lh]
)

# create df with cortical thickness difference

diff_table_ct_76_rh = diff_table_76_a.filter(regex=("FS7_6_diff_rh.+?_ct"), axis=1)
diff_table_ct_75_rh = diff_table_75_a.filter(regex=("FS7_5_diff_rh.+?_ct"), axis=1)
diff_table_ct_65_rh = diff_table_65_a.filter(regex=("FS6_5_diff_rh.+?_ct"), axis=1)
diff_table_ct_76_rh["version"] = "FS76"
diff_table_ct_75_rh["version"] = "FS75"
diff_table_ct_65_rh["version"] = "FS65"
diff_table_ct_76_rh["hemi"] = "rh"
diff_table_ct_75_rh["hemi"] = "rh"
diff_table_ct_65_rh["hemi"] = "rh"

diff_table_ct_76_lh = sub_diff_table_76_a.filter(regex=("FS7_6_diff_lh.+?_ct"), axis=1)
diff_table_ct_75_lh = sub_diff_table_75_a.filter(regex=("FS7_5_diff_lh.+?_ct"), axis=1)
diff_table_ct_65_lh = sub_diff_table_65_a.filter(regex=("FS6_5_diff_lh.+?_ct"), axis=1)
diff_table_ct_76_lh["version"] = "FS76"
diff_table_ct_75_lh["version"] = "FS75"
diff_table_ct_65_lh["version"] = "FS65"
diff_table_ct_76_lh["hemi"] = "lh"
diff_table_ct_75_lh["hemi"] = "lh"
diff_table_ct_65_lh["hemi"] = "lh"

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

diff_table_ct_all = pd.concat(
    [
        diff_table_ct_76_rh,
        diff_table_ct_75_rh,
        diff_table_ct_65_rh,
        diff_table_ct_76_lh,
        diff_table_ct_75_lh,
        diff_table_ct_65_lh,
    ]
)

diff_table_ct_rh = pd.concat(
    [diff_table_ct_76_rh, diff_table_ct_75_rh, diff_table_ct_65_rh]
)

diff_table_ct_lh = pd.concat(
    [diff_table_ct_76_lh, diff_table_ct_75_lh, diff_table_ct_65_lh]
)

# %%
# subcortical plot


x = sub_diff_all
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)

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


fig.update_xaxes(title="", gridwidth=1, griddash="dot")
fig.update_yaxes(title="Volume (mm<sup>3</sup>)")
fig.update_yaxes(range=[-1000, 14000])
fig.update_layout(legend_title_text="Legend", font_size=12)
# fig.write_image("plot_sub_both_short_large.png", width=1920, height=1080)
fig.show()

# %% [markdown]
# <img src="images/plot_sub.png"/>

# %%
x = sub_diff_all.copy()
x["version"].replace("FS76", "FS 7 vs 6", inplace=True)
x["version"].replace("FS75", "FS 7 vs 5", inplace=True)
x["version"].replace("FS65", "FS 6 vs 5", inplace=True)

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

# fig.write_image("plot_sub_both_short_large.png", width=1080, height=1920)
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

# fig.write_image("plot_vol_both.png", width=1920, height=1080)
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

# fig.write_image("plot_vol_rh_large.png", width=1080, height=2400)
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
# fig.write_image("plot_vol_lh_large.png", width=1080, height=2400)
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
# fig.write_image("plot_surf_both.png", width=1920, height=1080)
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
# fig.write_image("plot_surf_rh_large.png", width=1080, height=2400)
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
# fig.write_image("plot_surf_lh_large.png", width=1080, height=2400)
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
# fig.write_image("plot_ct_both.png", width=1920, height=1080)
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
# fig.write_image("plot_ct_rh_large.png", width=1080, height=2400)
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
# fig.write_image("plot_ct_lh_large.png", width=1080, height=2400)
fig.show()

# %% [markdown]
# # Results - clinical whole-brain
#
# ## FS 5 - whole-brain results
#
# ## Group analysis

# %%
# group differences

result_group = open("result_group_FS5.txt", "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS5/results_group_{time}_{pair}_{hemi}/con_group_{pair}_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_group_FS5.txt",
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

result_group = open("result_corr_FS5.txt", "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS5/results_corr_{pair}_{hemi}_{time}/con_corr_UPDRS_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_corr_FS5.txt",
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

result_group = open("result_group_FS6.txt", "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS6/results_group_{time}_{pair}_{hemi}/con_group_{pair}_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_group_FS6.txt",
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

result_group = open("result_corr_FS6.txt", "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS6/results_corr_{time}_{pair}_{hemi}/con_corr_UPDRS_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_corr_FS6.txt",
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

result_group = open("result_group_FS7.txt", "w")

# print("Group analysis\n")
for pair in ["HC_PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS7/results_group_{time}_{pair}_{hemi}/con_group_{pair}_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_group_FS7.txt",
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

result_group = open("result_corr_FS7.txt", "w")

# print("Group analysis\n")
for pair in ["PDnonMCI"]:
    # print("** Significant group differences between {pair} **\n".format(pair=pair))
    for hemi in ["lh", "rh"]:
        for time in ["base", "long"]:
            # print("Hemisphere {hemi}\n".format(hemi=hemi))
            file = open(
                "stats/results/FS7/results_corr_{time}_{pair}_{hemi}/con_corr_UPDRS_{time}/cache.th13.abs.sig.cluster.summary".format(
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
results_group = pd.read_csv(
    "result_corr_FS7.txt",
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
QPN = pd.read_csv("qpn/QPN_subjects.csv")[["PATNO", "sex", "age"]]

# %%
sex_f = QPN[QPN["sex"] == "F"]["PATNO"].nunique()
sex_m = QPN[QPN["sex"] == "M"]["PATNO"].nunique()
age_m = QPN["age"].mean()
age_sd = QPN["age"].std()

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
        for file in Path(f"qpn/segm/{ver}/{view}").iterdir():
            if not file.is_file():
                continue

            images.append(iio.imread(file))
            iio.mimsave(f"qpn/segm/{ver}/segm_{view}.gif", images, duration=1)

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
        file = f"mriqc/output_qpn/sub-{subj}/ses-01/anat/sub-{subj}_T1w.json"

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
        file = "qpn/stats/{version}/{subidd}/stats/aseg.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/aseg.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/lh.aparc.a2009s.stats".format(
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

            file = "qpn/stats/{version}/{subidd}/stats/rh.aparc.a2009s.stats".format(
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
# temp.to_csv("results_qpn_ttest_paired_76.csv")
# temp = pd.DataFrame.from_dict(t_paired_75, orient = 'index')
# temp.to_csv("results_qpn_ttest_paired_75.csv")
# temp = pd.DataFrame.from_dict(t_paired_65, orient = 'index')
# temp.to_csv("results_qpn_ttest_paired_65.csv")

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
# temp.to_csv("results_qpn_ttest_groups_76.csv")
# temp = pd.DataFrame.from_dict(t_groups_75, orient = 'index')
# temp.to_csv("results_qpn_ttest_groups_75.csv")
# temp = pd.DataFrame.from_dict(t_groups_65, orient = 'index')
# temp.to_csv("results_qpn_ttest_groups_65.csv")

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
# temp.to_csv("qc_qpn_corr_76.csv")
# temp = pd.DataFrame.from_dict(qc_corr_75, orient = 'index')
# temp.to_csv("qc_qpn_corr_75.csv")
# temp = pd.DataFrame.from_dict(qc_corr_65, orient = 'index')
# temp.to_csv("qc_qpn_corr_65.csv")

# %%
