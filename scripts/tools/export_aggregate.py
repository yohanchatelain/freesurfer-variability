"""Copy aggregate result files (group statistics, one row per region/test, no participant rows) from outputs/ to
results/, so they can be read for writing without opening the participant-level files of outputs/.

Only the files matched by AGGREGATE are copied, and a file is refused if it contains a participant identifier or a
participant column. results/ is gitignored like outputs/.
Usage: python scripts/tools/export_aggregate.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT, DEST = ROOT / "outputs", ROOT / "results"
AGGREGATE = [
    "qpn_demographics.csv", "qpn_proportion_significant.csv", "qpn_proportion_by_lobe.csv",
    "qpn_ppmi_significant_sets_dice.csv", "qpn_version_paired_ttests.csv", "qpn_pd_vs_ppmi_pd_variability_ttests.csv",
    "qpn_pd_vs_ppmi_pd_mriqc_ttests.csv", "qpn_mriqc_variability_correlations.csv",
    "qpn_pd_vs_ppmi_hc_variability_ttests.csv", "qpn_proportion_significant_all_pd_hc.csv",
    "qpn_pd_vs_hc_variability_ttests.csv", "qpn_pd_vs_hc_mriqc_ttests.csv",
    "version_paired_ttests.csv", "pd_vs_hc_variability_ttests.csv", "pd_vs_hc_variability_qc_covariates.csv",
    "dice_results.csv", "dk_*.csv", "tables/*.csv", "clinical_ancova_hc_pdnonmci.csv", "clinical_partial_corr_updrs.csv", "clinical_*_qc_covariates.csv", "clinical_ancova_hc_pdnonmci_tiv.csv",
    "variability_by_type_*.csv", "asymmetry_ttests.csv", "asymmetry_summary.csv", "crosscluster_*.csv", "exp2/*.csv",
]
ID = re.compile(r"sub-[A-Za-z0-9]+|PPMI_\d{3,}|\bI\d{6,8}\b|QPN[A-Z]*\d{4,}|^PATNO|,PATNO|PATNO_id", re.M)


def main():
    copied, refused = 0, []
    for pattern in AGGREGATE:
        for src in sorted(OUT.glob(pattern)):
            if ID.search(src.read_text(errors="ignore")):
                refused.append(str(src.relative_to(OUT)))
                continue
            dst = DEST / src.relative_to(OUT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"copied {copied} aggregate files to {DEST}")
    if refused:
        print("refused (participant identifiers or columns):", refused)


if __name__ == "__main__":
    main()
