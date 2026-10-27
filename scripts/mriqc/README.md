# MRIQC for the revision participants

MRIQC 22.0.1 on the first visit of each participant of the cross-sectional cohort that has no MRIQC output yet
(the paper's outputs are in `data/source_files/clinical/mriqc/output`). Same container (`mriqc_patch.simg`) and
options as the paper. `fsvv.mriqc_file` reads both output folders.

1. TemplateFlow (once, on a login node: compute nodes have no internet). MRIQC reads `$TEMPLATEFLOW_HOME`
   (default `~/.cache/templateflow`); fetch the MNI152NLin2009cAsym files it uses with the container's TemplateFlow:

   ```bash
   module load apptainer
   apptainer exec <repo>/data/source_files/clinical/mriqc/mriqc_patch.simg python -c "
   from templateflow import api
   for res in (1, 2):
       for kw in [dict(suffix='T1w'), dict(desc='brain', suffix='mask'), dict(desc='head', suffix='mask'),
                  dict(desc='carpet', suffix='dseg')] + [dict(label=l, suffix='probseg') for l in ('CSF', 'GM', 'WM')]:
           api.get('MNI152NLin2009cAsym', resolution=res, **kw)"
   ```

2. Inputs: `python scripts/mriqc/make_mriqc_bids.py` writes `outputs/mriqc/bids` (symlinks to the T1w images
   used by FreeSurfer) and `outputs/mriqc/participants.txt`.
3. Run, from the repository root (n = lines of `participants.txt`):
   `sbatch --account=<account> --array=0-<n-1>%40 scripts/mriqc/run_mriqc.sbatch`
