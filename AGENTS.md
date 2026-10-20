# Rules for AI agents working on this repository

## PPMI participant data must never be made public

This project analyses PPMI (Parkinson's Progression Markers Initiative) data under the PPMI
Data Use Agreement. **No information about individual PPMI participants may be released
publicly**, in particular on GitHub (commits, branches, pull requests, issues, comments,
gists, release assets, CI logs) or in any other public place (artifacts, pastes, web services).

This covers, among others:

- participant identifiers: PATNO, `sub-<PATNO>` / `sub-<PATNO>_ses-<visit>` / `*_base` names,
  LONI image IDs (`I<digits>`), series IDs, PPMI file names (`PPMI_<PATNO>_MR_...`);
- per-participant or per-visit records: visits and dates, age, sex, diagnosis group, clinical
  scores (UPDRS, MoCA, ...), years between visits, scanner/field strength;
- per-participant derived data: FreeSurfer outputs, tables with one row per participant or
  image, QC/exclusion/failure lists, qdec files, job lists;
- notebook outputs (printed dataframes, logs, images) that show any of the above.

Only aggregate results (group statistics, figures without identifiable rows) may be public.

### How to apply

- Keep participant-level files out of git: `data/`, `outputs/`, `paper/` and
  `scripts/new_participants/lists/` are in `.gitignore`. Put any new participant-level file
  in one of these locations (or add it to `.gitignore`) — never commit it.
- Never hardcode participant IDs in code, comments, markdown or docs. Read them from a
  gitignored file instead (e.g. `data/tables/excluded_images.txt` via `fsvv.inp(...)`), and
  describe them generically in prose ("one PD-non-MCI participant").
- Notebook outputs are stripped on commit by the `nbstripout` git filter (see
  `.gitattributes`). Do not bypass or disable it.
- Before every commit, push or PR, scan the staged diff for identifiers, e.g.
  `git diff --cached | grep -nE 'sub-[0-9]+|PATNO|PPMI_[0-9]+|\bI[0-9]{6,8}\b'`, and check
  that no new data file (`.csv`, `.tsv`, `.json`, `.dat`, `.txt`, `.nii*`, `.mgz`, ...) is staged.
- Do not paste participant-level data into commit messages, PR descriptions, issues or any
  external service.
- If in doubt, do not publish: ask the user first.
