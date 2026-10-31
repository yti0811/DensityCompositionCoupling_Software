"""
08_generate_manuscript_figures.py

Post-processing only. This script does not solve the thermodynamic model.
It reads the standardized CSV outputs from scripts 02 and 05 and generates
manuscript-facing figures with terminology consistent with the revised paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "05_manuscript_results_output" / "manuscript_state_table.csv"
REP_DIR = ROOT / "02_representative_linear_response_output"
OUT = ROOT / "08_manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

def require(path):
    if not path.is_file():
        raise FileNotFoundError(path)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)

require(MASTER)
master = pd.read_csv(MASTER)
regular = master[master["regular_single_phase"].astype(bool)].copy()

# Figure A: nonlinear adsorption response over the regular set.
fig, ax = plt.subplots(figsize=(6.2, 4.6))
sc = ax.scatter(regular["chi_ps"], regular["kappa"],
                c=regular["DD_Gamma_nonlinear"], s=58)
ax.set_xlabel(r"$\chi_{ps}$")
ax.set_ylabel(r"$\kappa$")
ax.set_title("Nonlinear adsorption response")
fig.colorbar(sc, ax=ax, label=r"$\Delta\widetilde{\Gamma}_p$")
save(fig, "manuscript_nonlinear_adsorption_response.png")

# Figure B: free-volume response over the regular set.
fig, ax = plt.subplots(figsize=(6.2, 4.6))
sc = ax.scatter(regular["chi_ps"], regular["kappa"],
                c=regular["void_response_nonlinear"], s=58)
ax.set_xlabel(r"$\chi_{ps}$")
ax.set_ylabel(r"$\kappa$")
ax.set_title("Free-volume response")
fig.colorbar(sc, ax=ax, label=r"$\widetilde{\Gamma}_{void}$")
save(fig, "manuscript_free_volume_response.png")

# Figure C: nonlinear versus LR adsorption response for finite LR states.
lr = regular[np.isfinite(pd.to_numeric(regular["DD_Gamma_LR"], errors="coerce"))].copy()
fig, ax = plt.subplots(figsize=(5.5, 5.0))
x = lr["DD_Gamma_nonlinear"].to_numpy(float)
y = lr["DD_Gamma_LR"].to_numpy(float)
ax.scatter(x, y, s=48)
if len(x):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
ax.set_xlabel(r"Nonlinear $\Delta\widetilde{\Gamma}_p$")
ax.set_ylabel(r"Linear-response $\Delta\widetilde{\Gamma}_p$")
ax.set_title("Linear-response comparison")
save(fig, "manuscript_linear_response_comparison.png")

# Representative profile figures from script 02.
profile_files = {
    "A": REP_DIR / "representative_A_baseline.csv",
    "B": REP_DIR / "representative_B_free_volume_depletion.csv",
    "C": REP_DIR / "representative_C_free_volume_enrichment.csv",
}
for label, path in profile_files.items():
    if not path.is_file():
        continue
    d = pd.read_csv(path)
    if "z_reduced" not in d.columns:
        continue
    # Plot only columns that are actually present; no model quantity is reconstructed.
    candidates = [
        ("delta_phi_p_nonlinear", r"$\Delta\phi_p$ (nonlinear)"),
        ("delta_phi_p_LR", r"$\Delta\phi_p$ (LR)"),
        ("delta_eta_nonlinear", r"$\Delta\eta$ (nonlinear)"),
        ("delta_eta_LR", r"$\Delta\eta$ (LR)"),
    ]
    available = [(c,l) for c,l in candidates if c in d.columns]
    if not available:
        continue
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for c, lab in available:
        ax.plot(d["z_reduced"], d[c], label=lab)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel("Response")
    ax.set_title(f"Representative state {label}")
    ax.legend(frameon=False)
    save(fig, f"representative_{label}_response.png")

print(f"Figures written to: {OUT}")
