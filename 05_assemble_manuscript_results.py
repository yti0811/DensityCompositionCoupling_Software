"""
MANUSCRIPT RESULT ASSEMBLY
Final production-result selection and manuscript audit.

Revision principles
-------------------
1. The nonlinear equilibrium response is taken from Step 14 for all 112 states.
   Step 16 is used only for linear-response (LR) diagnostics.
2. The constitutive working-range classification is independent of LR error:
      core     : max|Delta eta| <= 0.05 and min lambda(HII) >= 0.20
      extended : max|Delta eta| <= 0.10 and min lambda(HII) >= 0.10,
                 excluding core
      stress   : remaining regular BG-I states
3. The two Case-I-reference singular states (kappa, chi_ps) =
   (0.2, 0.6) and (0.1, 0.6) retain their nonlinear equilibrium data but are
   excluded from finite LR-error statistics.
4. Step 17 may supersede Step-16 density-excursion / HII-stiffness metrics
   where an explicit Step-17 certification is available.

Expected audit for the current production data
----------------------------------------------
112 total = 76 BG + 21 BM + 15 BU
76 BG = 60 regular + 9 KKT + 7 bistable
60 regular = 52 core + 4 extended + 4 stress-test
60/60 regular states have positive nonlinear adsorption response
24/60 regular states show void enrichment; 36/60 show void depletion
58/60 regular states have finite LR comparisons; 2/60 are LR singular
"""

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
STEP14_FILE = SCRIPT_DIR / "01_equilibrium_output" / "equilibrium_all_states.csv"
STEP16_FILE = SCRIPT_DIR / "03_global_linear_response_output" / "global_linear_response_regular_states.csv"
STEP17_FILE = SCRIPT_DIR / "04_soft_density_regime_output" / "soft_regime_state_summary.csv"
OUTPUT_DIR = SCRIPT_DIR / "05_manuscript_results_output"


# -----------------------------------------------------------------------------
# Current manuscript criteria: LR-INDEPENDENT constitutive working range
# -----------------------------------------------------------------------------
CORE_MAX_ABS_DETA = 0.05
CORE_MIN_LAMBDA_HII = 0.20
EXT_MAX_ABS_DETA = 0.10
EXT_MIN_LAMBDA_HII = 0.10

# R1 audit: singular Case-I reference Hessian -> no finite LR error assigned.
LR_SINGULAR_CASEI_POINTS = (
    (0.2, 0.6),
    (0.1, 0.6),
)

# Representative states used in the manuscript/SI.
REPRESENTATIVES = (
    ("A", 20.0, 0.2),
    ("B", 2.0, -0.6),
    ("C", 0.5, 0.4),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def is_point(kappa, chi_ps, target_kappa, target_chi, atol=1e-12):
    return np.isclose(kappa, target_kappa, rtol=0.0, atol=atol) & np.isclose(
        chi_ps, target_chi, rtol=0.0, atol=atol
    )


def read_inputs():
    require_file(STEP14_FILE, "Step-14 full results")
    require_file(STEP16_FILE, "Step-16 BG-I+ results")
    require_file(STEP17_FILE, "Step-17 case summary")

    s14 = pd.read_csv(STEP14_FILE)
    s16 = pd.read_csv(STEP16_FILE)
    s17 = pd.read_csv(STEP17_FILE)

    for df, label in ((s14, "equilibrium"), (s16, "global linear response"), (s17, "soft-density certification")):
        for col in ("kappa", "chi_ps"):
            if col not in df.columns:
                raise KeyError(f"{label} is missing required column {col!r}")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return s14, s16, s17


def normalize_step14(s14: pd.DataFrame) -> pd.DataFrame:
    """The equilibrium table is the source of nonlinear equilibrium observables/classes."""
    required = [
        "kappa", "chi_ps", "bulk_class", "surface_class", "final_class",
        "DeltaDelta_Gamma", "void_excess_II", "max_delta_eta",
        "profile_min_eigenvalue",
    ]
    missing = [c for c in required if c not in s14.columns]
    if missing:
        raise KeyError(f"Equilibrium table missing required columns: {missing}")

    out = s14.copy()
    numeric_cols = [
        "DeltaDelta_Gamma", "void_excess_II", "max_delta_eta",
        "profile_min_eigenvalue", "Gamma_I", "Gamma_II", "relative_percent",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Explicit manuscript-facing names.  Do not call these "exact".
    out["DD_Gamma_nonlinear"] = out["DeltaDelta_Gamma"]
    out["void_response_nonlinear"] = out["void_excess_II"]
    out["max_abs_delta_eta"] = out["max_delta_eta"].abs()
    out["min_lambda_HII"] = out["profile_min_eigenvalue"]

    return out


def normalize_step16(s16: pd.DataFrame) -> pd.DataFrame:
    """The global linear-response table contributes LR diagnostics only, not the nonlinear count."""
    required = ["kappa", "chi_ps"]
    missing = [c for c in required if c not in s16.columns]
    if missing:
        raise KeyError(f"Global linear-response table missing required columns: {missing}")

    keep = ["kappa", "chi_ps"]
    mapping = {
        "DD_Gamma_nonlinear": "globalLR_DD_Gamma_nonlinear",
        "DD_Gamma_LR": "DD_Gamma_LR",
        "LR_error_percent": "LR_error_percent_raw",
        "status": "globalLR_status",
        "failure_reason": "globalLR_failure_reason",
        "max_abs_delta_eta_nonlinear": "globalLR_max_abs_delta_eta",
        "min_lambda_CaseII": "globalLR_min_lambda_HII",
        "max_CaseII_residual": "globalLR_max_CaseII_residual",
    }

    out = s16[keep].copy()
    for src, dst in mapping.items():
        if src in s16.columns:
            if src in ("status", "failure_reason"):
                out[dst] = s16[src].astype(str)
            else:
                out[dst] = pd.to_numeric(s16[src], errors="coerce")
        else:
            out[dst] = "" if src in ("status", "failure_reason") else np.nan

    return out


def normalize_step17(s17: pd.DataFrame) -> pd.DataFrame:
    """The soft-density table certifies the low-kappa chi_ps=0.6 edge."""
    out = pd.DataFrame({
        "kappa": s17["kappa"],
        "chi_ps": s17["chi_ps"],
    })

    numeric_map = {
        "max_abs_delta_eta": "softcert_max_abs_delta_eta",
        "min_lambda": "softcert_min_lambda_HII",
        "n_unresolved": "softcert_n_unresolved",
        "n_kkt_global": "softcert_n_kkt",
        "n_multistable": "softcert_n_multistable",
    }
    for src, dst in numeric_map.items():
        out[dst] = (
            pd.to_numeric(s17[src], errors="coerce") if src in s17.columns else np.nan
        )

    out["softcert_contact_type"] = (
        s17["contact_global_type"].astype(str)
        if "contact_global_type" in s17.columns else ""
    )

    out["softcert_certified_regular"] = (
        out["softcert_n_unresolved"].fillna(0).eq(0)
        & out["softcert_n_kkt"].fillna(0).eq(0)
        & out["softcert_n_multistable"].fillna(0).eq(0)
        & out["softcert_contact_type"].fillna("").isin(["", "INTERIOR"])
    )
    return out


def build_master(s14, s16, s17):
    master = normalize_step14(s14)
    master = master.merge(normalize_step16(s16), on=["kappa", "chi_ps"], how="left")
    master = master.merge(normalize_step17(s17), on=["kappa", "chi_ps"], how="left")

    # Step-17 metrics supersede Step-14/16 metrics where Step 17 explicitly
    # audited the state.  This does not alter the Step-14 nonlinear response.
    m = np.isfinite(master["softcert_max_abs_delta_eta"])
    master.loc[m, "max_abs_delta_eta"] = master.loc[m, "softcert_max_abs_delta_eta"]
    m = np.isfinite(master["softcert_min_lambda_HII"])
    master.loc[m, "min_lambda_HII"] = master.loc[m, "softcert_min_lambda_HII"]

    # Thermodynamic/interfacial classes from Step 14.
    master["is_BG"] = master["bulk_class"].eq("BG")
    master["is_BM"] = master["bulk_class"].eq("BM")
    master["is_BU"] = master["bulk_class"].eq("BU")
    master["regular_single_phase"] = master["final_class"].astype(str).str.startswith("BG-I")
    master["is_KKT"] = master["final_class"].astype(str).str.startswith("BG-K")
    master["is_bistable"] = master["final_class"].isin(["BG-BI", "BG-BII"])

    # Constitutive working range: intentionally independent of LR.
    finite_constitutive = (
        np.isfinite(master["max_abs_delta_eta"])
        & np.isfinite(master["min_lambda_HII"])
    )
    core = (
        master["regular_single_phase"]
        & finite_constitutive
        & master["max_abs_delta_eta"].le(CORE_MAX_ABS_DETA)
        & master["min_lambda_HII"].ge(CORE_MIN_LAMBDA_HII)
    )
    extended = (
        master["regular_single_phase"]
        & ~core
        & finite_constitutive
        & master["max_abs_delta_eta"].le(EXT_MAX_ABS_DETA)
        & master["min_lambda_HII"].ge(EXT_MIN_LAMBDA_HII)
    )
    stress = master["regular_single_phase"] & ~core & ~extended

    master["working_range"] = "NOT_REGULAR"
    master.loc[core, "working_range"] = "CORE"
    master.loc[extended, "working_range"] = "EXTENDED"
    master.loc[stress, "working_range"] = "STRESS_TEST"

    # Production class retains the historical names for downstream compatibility,
    # but their definition is now LR-independent.
    master["production_class"] = "UNCLASSIFIED"
    master.loc[master["is_BM"], "production_class"] = "EXCLUDED_BM"
    master.loc[master["is_BU"], "production_class"] = "EXCLUDED_BU"
    master.loc[master["is_bistable"], "production_class"] = "BG_BISTABLE"
    master.loc[master["is_KKT"], "production_class"] = "BG_KKT"
    master.loc[core, "production_class"] = "BG_REGULAR_Q_CORE"
    master.loc[extended, "production_class"] = "BG_REGULAR_Q_EXTENDED"
    master.loc[stress, "production_class"] = "BG_REGULAR_NL"

    # Nonlinear response diagnostics are based on Step 14 for all regular states.
    master["nonlinear_response_finite"] = np.isfinite(master["DD_Gamma_nonlinear"])
    master["nonlinear_response_sign"] = np.select(
        [master["DD_Gamma_nonlinear"].gt(0), master["DD_Gamma_nonlinear"].lt(0),
         master["DD_Gamma_nonlinear"].eq(0)],
        ["ENHANCE", "SUPPRESS", "ZERO"], default="MISSING"
    )
    master["void_response_sign"] = np.select(
        [master["void_response_nonlinear"].gt(0), master["void_response_nonlinear"].lt(0),
         master["void_response_nonlinear"].eq(0)],
        ["VOID_ENRICH", "VOID_DEPLETE", "VOID_ZERO"], default="MISSING"
    )

    # LR availability is a separate diagnostic.  Explicitly invalidate the two
    # singular Case-I-reference states even if an older Step-16 file contains a
    # finite-looking number for one of them.
    singular = np.zeros(len(master), dtype=bool)
    for kappa, chi in LR_SINGULAR_CASEI_POINTS:
        singular |= is_point(master["kappa"], master["chi_ps"], kappa, chi)
    master["LR_caseI_reference_singular"] = singular & master["regular_single_phase"]

    raw_lr_finite = (
        np.isfinite(master["DD_Gamma_LR"])
        & np.isfinite(master["LR_error_percent_raw"])
    )
    master["LR_admissible"] = (
        master["regular_single_phase"]
        & raw_lr_finite
        & ~master["LR_caseI_reference_singular"]
    )

    master["LR_error_percent"] = master["LR_error_percent_raw"]
    master.loc[~master["LR_admissible"], "LR_error_percent"] = np.nan
    master.loc[~master["LR_admissible"], "DD_Gamma_LR"] = np.nan

    master["LR_status"] = np.where(
        master["LR_caseI_reference_singular"], "SINGULAR_CASEI_REFERENCE",
        np.where(master["LR_admissible"], "FINITE", "NOT_EVALUATED")
    )

    return master


def validate_master(master: pd.DataFrame) -> dict:
    reg = master[master["regular_single_phase"]].copy()

    counts = {
        "total": len(master),
        "BG": int(master["is_BG"].sum()),
        "BM": int(master["is_BM"].sum()),
        "BU": int(master["is_BU"].sum()),
        "regular": len(reg),
        "KKT": int(master["is_KKT"].sum()),
        "bistable": int(master["is_bistable"].sum()),
        "core": int(reg["working_range"].eq("CORE").sum()),
        "extended": int(reg["working_range"].eq("EXTENDED").sum()),
        "stress": int(reg["working_range"].eq("STRESS_TEST").sum()),
        "finite_nonlinear": int(reg["nonlinear_response_finite"].sum()),
        "enhance": int(reg["nonlinear_response_sign"].eq("ENHANCE").sum()),
        "suppress": int(reg["nonlinear_response_sign"].eq("SUPPRESS").sum()),
        "void_enrich": int(reg["void_response_sign"].eq("VOID_ENRICH").sum()),
        "void_deplete": int(reg["void_response_sign"].eq("VOID_DEPLETE").sum()),
        "LR_finite": int(reg["LR_admissible"].sum()),
        "LR_singular": int(reg["LR_caseI_reference_singular"].sum()),
    }

    expected = {
        "total": 112, "BG": 76, "BM": 21, "BU": 15,
        "regular": 60, "KKT": 9, "bistable": 7,
        "core": 52, "extended": 4, "stress": 4,
        "finite_nonlinear": 60, "enhance": 60, "suppress": 0,
        "void_enrich": 24, "void_deplete": 36,
        "LR_finite": 58, "LR_singular": 2,
    }

    problems = []
    for key, value in expected.items():
        if counts[key] != value:
            problems.append(f"{key}: got {counts[key]}, expected {value}")

    if problems:
        raise RuntimeError("Manuscript-result audit mismatch:\n  " + "\n  ".join(problems))

    return counts


def representative_table(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep, kappa, chi in REPRESENTATIVES:
        m = is_point(master["kappa"], master["chi_ps"], kappa, chi)
        sub = master.loc[m]
        if len(sub) != 1:
            raise RuntimeError(f"Representative {rep} matched {len(sub)} rows")
        row = sub.iloc[0]
        rows.append({
            "rep": rep,
            "kappa": row["kappa"],
            "chi_ps": row["chi_ps"],
            "working_range": row["working_range"],
            "DD_Gamma_nonlinear": row["DD_Gamma_nonlinear"],
            "void_response_nonlinear": row["void_response_nonlinear"],
            "max_abs_delta_eta": row["max_abs_delta_eta"],
            "min_lambda_HII": row["min_lambda_HII"],
            "LR_status": row["LR_status"],
            "DD_Gamma_LR": row["DD_Gamma_LR"],
            "LR_error_percent": row["LR_error_percent"],
        })
    return pd.DataFrame(rows)


def write_outputs(master: pd.DataFrame, counts: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master.to_csv(OUTPUT_DIR / "manuscript_state_table.csv", index=False)

    reg = master[master["regular_single_phase"]].copy()
    reg.to_csv(OUTPUT_DIR / "regular_states.csv", index=False)

    reg[reg["LR_admissible"]].to_csv(
        OUTPUT_DIR / "linear_response_admissible_states.csv", index=False
    )
    reg[reg["LR_caseI_reference_singular"]].to_csv(
        OUTPUT_DIR / "linear_response_singular_states.csv", index=False
    )

    representative_table(master).to_csv(
        OUTPUT_DIR / "representative_states_ABC.csv", index=False
    )

    # Compact classification matrix useful for manuscript/SI audit.
    matrix_cols = [
        "kappa", "chi_ps", "bulk_class", "surface_class", "final_class",
        "production_class", "working_range", "DD_Gamma_nonlinear",
        "void_response_nonlinear", "max_abs_delta_eta", "min_lambda_HII",
        "LR_status", "DD_Gamma_LR", "LR_error_percent",
    ]
    master[matrix_cols].to_csv(
        OUTPUT_DIR / "manuscript_audit_table.csv", index=False
    )

    lines = [
        "MANUSCRIPT RESULTS -- FINAL AUDIT",
        "",
        f"Total parameter combinations: {counts['total']}",
        f"Bulk classes: BG={counts['BG']}, BM={counts['BM']}, BU={counts['BU']}",
        f"BG interfacial classes: regular={counts['regular']}, KKT={counts['KKT']}, bistable={counts['bistable']}",
        f"Regular working range: core={counts['core']}, extended={counts['extended']}, stress-test={counts['stress']}",
        f"Finite nonlinear regular responses: {counts['finite_nonlinear']}",
        f"Nonlinear adsorption: enhance={counts['enhance']}, suppress={counts['suppress']}",
        f"Nonlinear void response: enrich={counts['void_enrich']}, deplete={counts['void_deplete']}",
        f"LR comparison: finite={counts['LR_finite']}, singular Case-I reference={counts['LR_singular']}",
        "",
        "Constitutive working-range classification is independent of LR error.",
        "Nonlinear equilibrium observables are sourced from Step 14.",
        "Step 16 is used only for LR diagnostics.",
    ]
    (OUTPUT_DIR / "manuscript_results_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def print_summary(counts: dict) -> None:
    print("=" * 72)
    print("MANUSCRIPT RESULTS -- FINAL AUDIT")
    print("=" * 72)
    print(f"Total = {counts['total']}")
    print(f"BG/BM/BU = {counts['BG']}/{counts['BM']}/{counts['BU']}")
    print(
        "BG: regular/KKT/bistable = "
        f"{counts['regular']}/{counts['KKT']}/{counts['bistable']}"
    )
    print(
        "Regular working range: core/extended/stress = "
        f"{counts['core']}/{counts['extended']}/{counts['stress']}"
    )
    print(
        "Regular nonlinear adsorption: finite/enhance/suppress = "
        f"{counts['finite_nonlinear']}/{counts['enhance']}/{counts['suppress']}"
    )
    print(
        "Regular nonlinear void response: enrich/deplete = "
        f"{counts['void_enrich']}/{counts['void_deplete']}"
    )
    print(
        "Regular LR: finite/singular = "
        f"{counts['LR_finite']}/{counts['LR_singular']}"
    )
    print("PASS: all expected revision counts reproduced.")


def main():
    s14, s16, s17 = read_inputs()
    master = build_master(s14, s16, s17)
    counts = validate_master(master)
    write_outputs(master, counts)
    print_summary(counts)


if __name__ == "__main__":
    main()
