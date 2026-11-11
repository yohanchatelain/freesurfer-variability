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
# # 01 · Cohort selection (PPMI)
#
# Builds the computational cohort (HC, PD-non-MCI, PD-MCI; two visits per subject) and the clinical
# cohort (PD-non-MCI with UPDRS-III OFF at both visits) used in the paper, and writes:
#
# | file (`outputs/`) | content |
# |---|---|
# | `download_data.csv` | both visits of every subject of the computational cohort (input of FreeSurfer) |
# | `pd_clinical.csv` | clinical PD-non-MCI cohort |
# | `stage_two_clinical.csv` | clinical cohort + HC (input of the longitudinal stream and of `03_clinical`) |
#
# PPMI study files were downloaded with `livingpark_utils` (requires PPMI credentials);
# `MDS_UPDRS_Part_III_clean.csv` and `MRI_info.csv` were produced by the LivingPark
# `pd_status` and `mri_metadata` scripts.
#
# The revision uses the same criteria as the paper on a newer snapshot of these files
# (`fsvv.STUDY_FILES`, default `data/study_files_2025-09`, MRI listing up to April 2024).
# The published cohort is reproduced with `FSVV_STUDY_FILES=<repo>/data/inputs/study_files`
# (the files used for the paper); run it with `FSVV_OUT` set to keep it apart from `outputs/`.

# %%
from itertools import combinations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2_contingency, ttest_ind

import fsvv
from fsvv import VISIT2MONTH, pair_visits

STUDY = fsvv.STUDY_FILES


def read(name, cols):
    return pd.read_csv(STUDY / name, low_memory=False)[cols]


dem = read("Demographics.csv", ["PATNO", "SEX", "BIRTHDT"])
age = read("Age_at_visit.csv", ["PATNO", "EVENT_ID", "AGE_AT_VISIT"])
edu = read("Socio-Economics.csv", ["PATNO", "EDUCYRS"])
diag = read("Primary_Clinical_Diagnosis.csv", ["PATNO", "EVENT_ID", "PRIMDIAG", "OTHNEURO"])
dx_status = read("Participant_Status.csv", ["PATNO", "COHORT"])
pd_hist = read("PD_Diagnosis_History.csv", ["PATNO", "EVENT_ID", "PDDXDT"])
cog_cat = read("Cognitive_Categorization.csv", ["PATNO", "EVENT_ID", "COGSTATE"])
updrs = read("MDS_UPDRS_Part_III_clean.csv", ["PATNO", "EVENT_ID", "PDSTATE", "NP3TOT", "NHY", "PDTRTMNT"])

mri = pd.read_csv(STUDY / "MRI_info.csv")
mri = mri.assign(EVENT_ID=mri["Visit code"], PATNO=mri["Subject ID"]).drop(
    columns=["Subject ID", "Visit code", "Visit", "Age", "Sex"])
mri.groupby("EVENT_ID").size().sort_values(ascending=False).head(5)

# %% [markdown]
# ## Visit pairs available per group

# %%
EVENTS = ["BL", "V04", "V06", "V08", "V10"]


def count_visit_pairs(df, mask, title):
    print(title)
    for a, b in combinations(EVENTS, 2):
        n = len(set(df.loc[mask & (df["EVENT_ID"] == a), "PATNO"]) & set(df.loc[mask & (df["EVENT_ID"] == b), "PATNO"]))
        if n:
            print(f"  {a:3} & {b:3} = {n:>3} | Month difference: {VISIT2MONTH[b] - VISIT2MONTH[a]}")


def merge_visits(diagnosis, extra=()):
    d = (mri.merge(diagnosis, on=["PATNO", "EVENT_ID"])
         .merge(age, on=["PATNO", "EVENT_ID"], how="left"))
    for table, kw in extra:
        d = d.merge(table, **kw)
    return d.drop_duplicates().groupby("PATNO").filter(lambda g: g["EVENT_ID"].nunique() > 1)


# HC: use the screening diagnosis when PRIMDIAG is missing at baseline (SC -> BL).
# Note: the original `diag_other["EVENT_ID"].mask(..., inplace=True)` is a silent no-op with
# pandas >= 3 (copy-on-write) and drops 64 HC; an explicit .loc assignment keeps the published cohort.
diag_other = diag[diag["EVENT_ID"] != "BL"].copy()
diag_other.loc[diag_other["EVENT_ID"] == "SC", "EVENT_ID"] = "BL"
diag_hc = pd.concat([diag[diag["EVENT_ID"] == "BL"], diag_other]).drop_duplicates()

df_hc = merge_visits(diag_hc, [
    (dem, dict(on=["PATNO"], how="left")), (dx_status, dict(on=["PATNO"], how="left")),
    (edu, dict(on=["PATNO"], how="left")), (cog_cat, dict(on=["PATNO", "EVENT_ID"], how="left")),
    (pd_hist, dict(on=["PATNO", "EVENT_ID"], how="left"))])
df = merge_visits(diag, [
    (dem, dict(on=["PATNO"])), (edu, dict(on=["PATNO"], how="left")), (dx_status, dict(on=["PATNO"])),
    (pd_hist, dict(on=["PATNO", "EVENT_ID"], how="left")), (cog_cat, dict(on=["PATNO", "EVENT_ID"]))])

pd_mask = lambda d, cog: (d["PRIMDIAG"] == 1) & (d["COGSTATE"] == cog) & (d["COHORT"] == 1) & d["OTHNEURO"].isnull()
count_visit_pairs(df_hc, df_hc["PRIMDIAG"] == 17, "Unique HC subjects per visit pair:")
count_visit_pairs(df, pd_mask(df, 1), "Unique PD-non-MCI subjects per visit pair:")
count_visit_pairs(df, pd_mask(df, 2), "Unique PD-MCI subjects per visit pair:")


# %% [markdown]
# ## Select HC, PD-MCI and PD-non-MCI (visit pairs ~12, 24, 36 months apart)

# %%
def select(df, keep, group, months=(12, 24, 36)):
    sel = pd.concat([pair_visits(df, m, keep) for m in months], ignore_index=True)
    return sel.drop_duplicates(subset=["PATNO"]).assign(dx_group=group)


both_visits = lambda cond: (lambda v: cond(v, "") & cond(v, "_NX"))
pd_visit = lambda cog: (lambda v, s: (v["COGSTATE" + s] == cog) & (v["PRIMDIAG" + s] == 1)
                        & (v["COHORT" + s] == 1) & v["OTHNEURO" + s].isnull())

hc = select(df_hc, both_visits(lambda v, s: v["PRIMDIAG" + s] == 17), "HC")
hc = hc.loc[hc["COHORT"].isin([2, 4])]  # PPMI COHORT codes kept by the original selection
print("Unique HC:", hc["PATNO"].nunique())
mci = select(df, both_visits(pd_visit(2)), "PD-MCI")
nonmci = select(df, both_visits(pd_visit(1)), "PD-non-MCI")

cohort = pd.concat([mci, nonmci, hc], ignore_index=True).drop_duplicates(subset=["PATNO"])
for c in ["Study Date", "Study Date_NX"]:
    cohort[c] = pd.to_datetime(cohort[c])


def months_between(d1, d2):
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    rd = relativedelta(d2, d1)
    return rd.years * 12 + rd.months + rd.days / 30.44


cohort["durationT2_T1"] = [months_between(a, b) for a, b in zip(cohort["Study Date"], cohort["Study Date_NX"])]
cohort["durationT2_T1_y"] = cohort["durationT2_T1"] / 12
cohort["PATNO_id"] = "sub-" + cohort["PATNO"].astype(str) + "_ses-" + cohort["EVENT_ID"]
cohort["dx_group"].value_counts()


# %% [markdown]
# ## Descriptive statistics (computational cohort)

# %%
def cohort_summary(groups, clinical=False):
    """Table 1-style summary: mean ± SD (range for age)."""
    ms = lambda d, c: f"{d[c].mean():4.1f} ± {d[c].std():4.1f}"
    rows = {
        "n": {g: f"{d['PATNO'].count()}" for g, d in groups.items()},
        "Age (y)": {g: ms(d, "AGE_AT_VISIT") for g, d in groups.items()},
        "Age range": {g: f"{d['AGE_AT_VISIT'].min():4.1f} - {d['AGE_AT_VISIT'].max():4.1f}" for g, d in groups.items()},
        "Gender (male, %)": {g: f"{(d['SEX'] == 1).sum()}, {(d['SEX'] == 1).sum() / d['PATNO'].nunique() * 100:.1f}%"
                             for g, d in groups.items()},
        "Education (y)": {g: ms(d, "EDUCYRS") for g, d in groups.items()},
    }
    if clinical:
        rows["UPDRS III OFF baseline"] = {g: ms(d, "NP3TOT") if "NP3TOT" in d and d["NP3TOT"].notna().any() else ""
                                          for g, d in groups.items()}
        rows["UPDRS III OFF follow-up"] = {g: ms(d, "NP3TOT_NX") if "NP3TOT_NX" in d and d["NP3TOT_NX"].notna().any()
                                           else "" for g, d in groups.items()}
        rows["Duration T2 - T1 (y)"] = {g: ms(d, "durationT2_T1_y") for g, d in groups.items()}
    return pd.DataFrame(rows).T


def compare_pd_hc(pd_group, hc_group, extra=()):
    for c in ["AGE_AT_VISIT", "EDUCYRS", *extra]:
        t, p = ttest_ind(pd_group[c], hc_group[c], nan_policy="omit")
        print(f"PD vs HC, {c}: t = {t:.2f}, p = {p:.3g}")
    obs = [[(pd_group["SEX"] == s).sum() for s in (0, 1)], [(hc_group["SEX"] == s).sum() for s in (0, 1)]]
    chi2, p, dof, _ = chi2_contingency(obs)
    print(f"PD vs HC, sex: chi2 = {chi2:.2f}, p = {p:.3g}")


hc = cohort[cohort["dx_group"] == "HC"]
nc = cohort[cohort["dx_group"] == "PD-non-MCI"]
mci = cohort[cohort["dx_group"] == "PD-MCI"]
display(cohort_summary({"HC": hc, "PD-non-MCI": nc, "PD-MCI": mci}))
compare_pd_hc(pd.concat([nc, mci]), hc)

# %% [markdown]
# ## Clinical cohort: PD-non-MCI with UPDRS-III (OFF) at both visits

# %%
df_clinical = merge_visits(diag, [
    (dem, dict(on=["PATNO"])), (edu, dict(on=["PATNO"], how="left")), (dx_status, dict(on=["PATNO"])),
    (pd_hist, dict(on=["PATNO", "EVENT_ID"], how="left")), (cog_cat, dict(on=["PATNO", "EVENT_ID"])),
    (updrs, dict(on=["PATNO", "EVENT_ID"]))])
off = lambda v, s: pd_visit(1)(v, s) & (v["PDSTATE" + s] == "OFF")
pd_clinical = select(df_clinical, both_visits(off), "PD-non-MCI")
pd_clinical = pd_clinical[pd_clinical["NP3TOT"].notna() & pd_clinical["NP3TOT_NX"].notna()].copy()
for c in ["Study Date", "Study Date_NX"]:
    pd_clinical[c] = pd.to_datetime(pd_clinical[c])
pd_clinical["durationT2_T1_y"] = (pd_clinical["Study Date_NX"] - pd_clinical["Study Date"]).dt.days / 365.25
pd_clinical["durationT2_T1"] = pd_clinical["durationT2_T1_y"] * 12
pd_clinical["PATNO_id"] = "sub-" + pd_clinical["PATNO"].astype(str) + "_ses-" + pd_clinical["EVENT_ID"]
# Change in UPDRS-III = baseline - follow-up (negative when motor symptoms worsen).
pd_clinical["NP3TOT_change"] = pd_clinical["NP3TOT"] - pd_clinical["NP3TOT_NX"]
pd_clinical.to_csv(fsvv.out("pd_clinical.csv"), index=False)
print("PD-non-MCI with UPDRS at both visits:", len(pd_clinical))

display(cohort_summary({"HC": hc, "PD-non-MCI": pd_clinical}, clinical=True))
compare_pd_hc(pd_clinical, hc, extra=["durationT2_T1_y"])


# %% [markdown]
# ## Outputs

# %%
def both_visits_rows(d):
    second = d.drop(columns=["Description", "EVENT_ID"]).rename(
        columns={"Description_NX": "Description", "EVENT_ID_NX": "EVENT_ID"})
    both = pd.concat([d, second])
    both["PATNO_id"] = "sub-" + both["PATNO"].astype(str) + "_ses-" + both["EVENT_ID"]
    return both


download_data = both_visits_rows(cohort)
download_data.to_csv(fsvv.out("download_data.csv"))

# The clinical sample must be a subsample of the computational one (no new image to process).
visit_clin = both_visits_rows(pd_clinical)
new_images = set(visit_clin["PATNO_id"]) - set(download_data["PATNO_id"])
print("Images of the clinical cohort not in the computational cohort:", len(new_images))

# Clinical cohort + HC: input of the longitudinal stream and of 03_clinical
stage_two = pd.concat([pd_clinical, cohort[cohort["dx_group"] == "HC"]])
stage_two["first_visit"] = "sub-" + stage_two["PATNO"].astype(str) + "_ses-" + stage_two["EVENT_ID"]
stage_two["second_visit"] = "sub-" + stage_two["PATNO"].astype(str) + "_ses-" + stage_two["NEXT_VISIT"]
stage_two.to_csv(fsvv.out("stage_two_clinical.csv"), index=False)

# %% [markdown]
# ## Check against the cohort used in the paper (`data/tables`)

# %%
for name, cols in [("download_data.csv", ["PATNO", "EVENT_ID", "dx_group"]),
                   ("stage_two_clinical.csv", ["PATNO", "EVENT_ID", "NEXT_VISIT", "dx_group"])]:
    new = pd.read_csv(fsvv.out(name))[cols]
    old = pd.read_csv(fsvv.DATA / "tables" / name)[cols]
    same = set(map(tuple, new.astype(str).values)) == set(map(tuple, old.astype(str).values))
    print(f"{name}: {len(new)} rows (published: {len(old)}), identical subjects/visits/groups: {same}")


# %% [markdown]
# ## FreeSurfer outputs available for the selected images
#
# The paper's runs are in `data/FSx`, the participants added for the revision in
# `data/new_participants/light/FSx` (`fsvv.SUBJECT_ROOTS`). Images selected by the criteria but not
# processed in every version (failed runs, or never downloaded) are listed in `outputs/not_processed.csv`.

# %%
def processed(subject):
    return [v for v in fsvv.VERSIONS if not (fsvv.subject_dir(v, subject) / "stats" / "aseg.stats").exists()]


cross = download_data.drop_duplicates("PATNO_id")[["PATNO", "PATNO_id", "dx_group"]].assign(stream="cross")
base = "sub-" + stage_two["PATNO"].astype(str)
long_ = pd.concat([stage_two.assign(PATNO_id=stage_two[c] + ".long." + base + "_base")
                   for c in ["first_visit", "second_visit"]])[["PATNO", "PATNO_id", "dx_group"]].assign(stream="long")
status = pd.concat([cross, long_], ignore_index=True)
status["missing_versions"] = [" ".join(processed(s)) for s in status["PATNO_id"]]
missing = status[status["missing_versions"] != ""]
missing.to_csv(fsvv.out("not_processed.csv"), index=False)
for stream, d in status.groupby("stream"):
    pts = d.groupby("PATNO")["missing_versions"].apply(lambda m: (m == "").all())
    print(f"{stream}: {len(d)} images of {d['PATNO'].nunique()} participants; "
          f"{pts.sum()} participants processed in all versions, {(~pts).sum()} not")
