# New participants: FS5, FS6, FS7, FS8 runs

**Who:** 286 participants (183 PD-non-MCI, 103 HC) of vinuyans' extended clinical cohort
(`data/tables/stage_two_clinical.csv`, 455 participants) have no FS5–7 run. FS8 tried 170 of them and every base
template failed, because their cross-sectional runs were never done (`norm.mgz` missing). Their T1w images are
not on the cluster.

**What:** for each participant and version, the same chain as the FS8 and paper runs:
`recon-all -all -qcache` on both visits → `-base` → `-long` on both visits → `-long -qcache` → `long_mris_slopes` (lh, rh).

**Where:** everything runs on `$SCRATCH/fsvv_newpart` (`FSVV_NEW_WORK`):

| Path | Content |
|---|---|
| `inputs/sub-*/ses-*/anat/` | T1w images (from `organize_downloads.sh`) |
| `FSx/` | `SUBJECTS_DIR` while a participant is running |
| `archives/FSx/sub-*_base.tar` | full outputs, one uncompressed tar per participant and version (5 folders) |
| `light/FSx/` | what the analyses read: `stats/` of the 5 folders, cross-sectional segmentations, `thickness.fwhm10`, `long.thickness-*.fwhm10` |
| `logs/`, `slurm/` | bosh invocations and logs, SLURM logs |

Once a participant is archived, its working folders are deleted, so scratch never holds more than the running
participants (about 2k files each). The scratch quota is 1M files.

## Files

| File | Purpose |
|---|---|
| `make_lists.py` | builds `lists/`: `new_participants.csv` (array index = `task`), `qdec/<PATNO>.dat`, `loni_image_ids.txt` |
| `boutiques/fsN_{cross,base,long,long_qcache,slopes}.json` | descriptors used for the paper: FS5–7 Zenodo records from `preprocess_FS{5,6,7}*.py`, FS8 from `vinuyans/boutiques` with `SUBJECTS_DIR=$(pwd)` instead of `$(pwd)/data/FS8`. The FS5 qcache and slopes Zenodo IDs are swapped in vinuyans' scripts, so these files are named by what they run |
| `run_chain.py <FSx> <task>` | full chain for one participant; skips finished stages, so failed tasks can be resubmitted |
| `run_chain.sbatch`, `submit.sh <FSx> [array] [throttle]` | SLURM array, 2 CPUs (the two visits run in parallel), 10 GB (48 GB for FS8), 48 h |
| `download_ppmi.py` | automated download with livingpark + Chrome 114 container; **blocked by the Turnstile check at login** |
| `organize_downloads.sh <zip>...` | LONI DICOM zip(s) → `inputs/.../PPMI_<PATNO>_I<ImageID>.nii` (dcm2niix) |

## Steps

1. Download from LONI IDA **in a browser**: the IDA login now has a Cloudflare Turnstile check (anti-bot), so
   `ppmi_downloader`/livingpark can't log in any more (tried on 2026-10-13; `download_ppmi.py` is kept but fails at login).
   Advanced Image Search → Image ID = the list in `lists/loni_image_ids.txt` → add all to a collection →
   **Advanced Download** → download the list of URLs. Then, on the cluster, fetch the zips with LONI's
   **IDA Downloader** (Java jar from the IDA support pages, needs Java ≥ 12: `module load java/17.0.6`; syntax in its
   README; IDA User Manual §4.2.5.3), or copy browser-downloaded zips to scratch. Then run
   `salloc --account=rrg-glatard_cpu --time=2:00:00 ./organize_downloads.sh <zips>` (it prints missing images).
2. Test: one participant in the 4 versions (`FSVV_NEW_WORK=$SCRATCH/fsvv_newpart_test FSVV_NEW_LISTS=$SCRATCH/fsvv_newpart_test/lists ./submit.sh FSx 0 1`).
3. Run: `for v in FS5 FS6 FS7 FS8; do ./submit.sh $v; done`. That's 1,144 tasks and about 27k core-hours.
   Runtimes from the paper runs: cross 4–8 h (FS6 outliers up to 36 h), base 2–7 h, long < 1 h.
4. Check: `ls archives/FSx | wc -l`, `grep -l 'failed\|Error' slurm/*.out`; resubmit the failed indices with `./submit.sh FSx <i,j,...>`.
5. Store: move `archives/` to the project (one file per tar) and keep `light/` for the analyses (~140k files);
   link both under `data/`. Scratch is purged after 60 days.
6. Cohort-level steps (after the cohort decision in `TODO.md`): FSGD files, `mris_preproc` stacks, GLMs.
