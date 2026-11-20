# FreeSurfer variability

## Paper figures

From the repository root, define the `figures` command (add the printed line to `~/.bashrc` to keep it):

```bash
alias figures="python $(realpath scripts/paper/figures.py)"
alias figures   # prints the line with the absolute path
```

```bash
figures install                  # once: .venv (Python 3.11, uv), requirements, "fsvv" Jupyter kernel
figures run                      # tuning notebooks 09 (Figures 1-12) and 10 (S1-S6); "Save to paper" in each tab
figures show 7                   # build the paper and open it at Figure 7 (also: S3, fig7_corr_base, or nothing)
figures save                     # copy the saved PDFs from outputs/figures to paper/latex/figures
figures commit -m "message"      # check for participant identifiers, commit and push (--no-push to skip)
```
