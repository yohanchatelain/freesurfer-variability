# Notebooks

Run in order with the `fsvv` kernel (Python 3.11 of the project venv linked in `data/`), from this folder.
Shared code (paths, region lists, stats parsers, tests, figures) is in [`fsvv.py`](fsvv.py).
Inputs are read from `../data`, which holds symlinks to the original files. Outputs go to `../outputs`
(figures in `../outputs/figures`). Set `FSVV_OUT=<dir>` to send all outputs somewhere else.

| Notebook | Content | Run |
|---|---|---|
| `00_pipeline.ipynb` | FreeSurfer processing, longitudinal stream, GLMs, `mri_seg_overlap` (as originally run) | reference only, **don't run** |
| `01_cohort.ipynb` | PPMI cohort selection, descriptive statistics → `download_data.csv`, `pd_clinical.csv`, `stage_two_clinical.csv` | ✓ |
| `02_software_variability.ipynb` | measures, paired tests between versions, Dice of significant sets, PD vs HC, MRIQC, segmentation Dice, Figures 1–3 | ✓ |
| `03_clinical.ipynb` | subcortical partial correlations and ANCOVA (Tables 2–3), vertex-wise clusters | ✓ |
| `04_qpn_replication.ipynb` | QPN replication | needs `data/qpn` |
| `11_mindboggle.ipynb` | Mindboggle-101: Dice and volume of the DKT/DK labels of each version vs the manual DKT31 labels, paired version tests (Exp 2) | needs `scripts/exp2` runs |
| `12_crosscluster.ipynb` | Repeat runs (2 replicas, current cluster) vs the paper's runs vs between-version differences on 20 images | needs `data/crosscluster` |
| `freesurfer_variability_mindboggle.ipynb` | Mindboggle subject list (early draft, superseded by `11_mindboggle`) | ✓ |
| `scratch.ipynb` | FS8 processing-completeness checks | ✓ |
| `legacy/` | previous notebooks (outputs cleared), kept for reference | — |

Each analysis notebook ends with a check against the tables used for the submitted paper (`data/tables`).
