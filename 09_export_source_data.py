"""
09_export_source_data.py

Exports compact, manuscript-facing source-data CSV files from the standardized
outputs. No thermodynamic calculation is performed here.
"""

from pathlib import Path
import shutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "05_manuscript_results_output" / "manuscript_state_table.csv"
REP_DIR = ROOT / "02_representative_linear_response_output"
OUT = ROOT / "09_source_data"
OUT.mkdir(parents=True, exist_ok=True)

if not MASTER.is_file():
    raise FileNotFoundError(MASTER)

m = pd.read_csv(MASTER)

classification_cols = [
    "kappa", "chi_ps", "bulk_class", "surface_class", "final_class",
    "production_class", "working_range"
]
response_cols = [
    "kappa", "chi_ps", "DD_Gamma_nonlinear", "void_response_nonlinear",
    "max_abs_delta_eta", "min_lambda_HII"
]
lr_cols = [
    "kappa", "chi_ps", "LR_status", "DD_Gamma_nonlinear",
    "DD_Gamma_LR", "LR_error_percent"
]

m[[c for c in classification_cols if c in m.columns]].to_csv(
    OUT / "SourceData_EquilibriumClassification.csv", index=False
)
m[[c for c in response_cols if c in m.columns]].to_csv(
    OUT / "SourceData_AdsorptionAndFreeVolumeResponse.csv", index=False
)

regular = m[m["regular_single_phase"].astype(bool)].copy()
regular[[c for c in lr_cols if c in regular.columns]].to_csv(
    OUT / "SourceData_LinearResponse.csv", index=False
)

rep_cols = [
    "kappa", "chi_ps", "working_range", "DD_Gamma_nonlinear",
    "void_response_nonlinear", "max_abs_delta_eta", "min_lambda_HII",
    "LR_status", "DD_Gamma_LR", "LR_error_percent"
]
reps = []
for rep, k, chi in [("A",20.0,0.2),("B",2.0,-0.6),("C",0.5,0.4)]:
    q = m[np.isclose(m["kappa"], k) & np.isclose(m["chi_ps"], chi)].copy()
    if len(q) == 1:
        q.insert(0, "representative", rep)
        reps.append(q)
if reps:
    r = pd.concat(reps, ignore_index=True)
    r[["representative"] + [c for c in rep_cols if c in r.columns]].to_csv(
        OUT / "SourceData_RepresentativeStatesABC.csv", index=False
    )

profile_map = {
    "A": "representative_A_baseline.csv",
    "B": "representative_B_free_volume_depletion.csv",
    "C": "representative_C_free_volume_enrichment.csv",
}
for rep, name in profile_map.items():
    p = REP_DIR / name
    if p.is_file():
        shutil.copy2(p, OUT / f"SourceData_Representative{rep}_Profile.csv")

print(f"Source-data files written to: {OUT}")
