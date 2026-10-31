#!/usr/bin/env python3
"""
06_numerical_convergence_validation.py

Numerical validation companion for the Scientific Reports revision.

Purpose
-------
R4: Re-run the 60 regular bulk-global states with the production Step-14
    equilibrium solver and export residual/stability/branch-gap diagnostics.
R5: Re-run representative Cases A-C and the soft regular state (0.1, 0.6)
    on refined quadrature grids and enlarged domains.

Required files (place in the same directory, or pass command-line paths):
  01_equilibrium_regime_map.py
  05_manuscript_results_output/manuscript_state_table.csv

Examples
--------
  python 06_numerical_convergence_validation.py
  python 06_numerical_convergence_validation.py \
      --step14 01_equilibrium_regime_map.py \
      --master 05_manuscript_results_output/manuscript_state_table.csv \
      --out 06_numerical_validation_output

Notes
-----
* This script imports the actual Step-14 production model/solver rather than
  reimplementing the thermodynamics.
* Spatial refinement is a quadrature/profile-sampling test because the model
  is local in z; it is not a PDE mesh-convergence test.
* Domain-size tests use the same dz while changing L.
* No hard-coded expected numerical answers are used as pass criteria.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


CASES = {
    "A": (20.0, 0.2),
    "B": (2.0, -0.6),
    "C": (0.5, 0.4),
    "SOFT": (0.1, 0.6),
}

GRID_DZ = (0.05, 0.025, 0.0125)
DOMAIN_L = (20.0, 30.0, 40.0)
PRODUCTION_L = 20.0
PRODUCTION_DZ = 0.05


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("step14_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import equilibrium solver module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def identify_regular60(master: pd.DataFrame) -> pd.DataFrame:
    """Select the 60 regular BG states without assuming one exact column name."""
    # Preferred explicit revision columns.
    for col in ("production_class", "surface_class", "final_class"):
        if col not in master.columns:
            continue
        s = master[col].astype(str)
        if col == "production_class":
            mask = s.str.lower().str.contains("regular")
        elif col == "surface_class":
            mask = s.str.fullmatch(r"I[+\-0]", na=False)
        else:
            mask = s.str.fullmatch(r"BG-I[+\-0]", na=False)
        sub = master.loc[mask].copy()
        if len(sub) == 60:
            return sub

    # Fallback: explicit regular flag/classification text.
    for col in master.columns:
        if master[col].dtype == object:
            s = master[col].astype(str).str.lower()
            mask = s.str.contains("regular", na=False) & ~s.str.contains("bistable|kkt", regex=True, na=False)
            sub = master.loc[mask].copy()
            if len(sub) == 60:
                return sub

    raise RuntimeError("Could not identify exactly 60 regular BG states from master table.")


def parameter_columns(df: pd.DataFrame) -> Tuple[str, str]:
    kcol = first_existing(df, ["kappa", "Kappa", "κ"])
    ccol = first_existing(df, ["chi_ps", "chi", "chi_bar_ps", "chibar_ps", "χ̄ps"])
    if kcol is None or ccol is None:
        raise RuntimeError(f"Cannot find kappa/chi_ps columns. Columns are: {list(df.columns)}")
    return kcol, ccol


def solve_profiles(m, kappa: float, chi_ps: float, L: float, dz: float):
    """Run Case I and Case II with the imported production solver."""
    par = m.ModelParams(kappa=float(kappa), chi_ps=float(chi_ps))
    wall = m.WallParams()
    z = m.make_grid(float(L), float(dz))
    sq = m.local_state_quantities(z, par, wall)
    log_q = sq["log_q"]
    I = m.solve_caseI_profile(z, log_q, par)
    II = m.solve_caseII_profile(z, log_q, par)
    if not I.get("success", False):
        raise RuntimeError(f"Case I failed at kappa={kappa}, chi={chi_ps}, z={I.get('failed_z')}")
    if not II.get("success", False):
        raise RuntimeError(f"Case II failed at kappa={kappa}, chi={chi_ps}, z={II.get('failed_z')}")

    gamma_I = m.integrate_profile(I["phi_p"] - par.phi_p_b, z)
    gamma_II = m.integrate_profile(II["phi_p"] - par.phi_p_b, z)
    dd = gamma_II - gamma_I
    void = m.integrate_profile(II["phi_v"] - par.phi_v_b, z)
    eta_b = 1.0 - par.phi_v_b

    finite_lam = II["lambda_min"][np.isfinite(II["lambda_min"])]
    finite_gap = II["gap"][np.isfinite(II["gap"])]
    return {
        "par": par,
        "wall": wall,
        "z": z,
        "log_q": log_q,
        "I": I,
        "II": II,
        "Gamma_I": float(gamma_I),
        "Gamma_II": float(gamma_II),
        "DeltaDelta_Gamma": float(dd),
        "void_excess_II": float(void),
        "max_delta_eta": float(np.max(np.abs(II["eta"] - eta_b))),
        "caseI_maxR": float(np.nanmax(I["Rmax"])),
        "caseII_maxR": float(np.nanmax(II["Rmax"])),
        "caseII_min_lambda": float(np.min(finite_lam)) if len(finite_lam) else np.nan,
        "min_competing_gap": float(np.min(finite_gap)) if len(finite_gap) else np.nan,
        "caseI_bistable_z_count": int(I["bistable_z_count"]),
        "caseII_bistable_z_count": int(II["bistable_z_count"]),
        "boundary_z_count": int(II["boundary_z_count"]),
        "boundary_fraction": float(II["boundary_fraction"]),
    }


def direct_local_minimum(m, par, log_q: float, theta0: float, eta0: float) -> Dict[str, float]:
    """Independent box-constrained local minimization of local grand density."""
    eps = 1e-10

    def obj(v):
        theta, eta = map(float, v)
        if not (eps < theta < 1.0 - eps and eps < eta <= 1.0):
            return 1e100
        x = eta * theta
        return m.local_grand_density(x, eta, float(log_q), par)

    sol = minimize(
        obj,
        x0=np.array([np.clip(theta0, 1e-8, 1 - 1e-8), np.clip(eta0, 1e-8, 1.0)]),
        method="L-BFGS-B",
        bounds=[(1e-9, 1 - 1e-9), (1e-9, 1.0)],
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12},
    )
    th, eta = map(float, sol.x)
    x = eta * th
    return {
        "direct_success": bool(sol.success),
        "direct_phi_p": x,
        "direct_eta": eta,
        "direct_g": float(m.local_grand_density(x, eta, float(log_q), par)),
        "direct_nit": int(getattr(sol, "nit", -1)),
    }


def r4_regular_audit(m, regular: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Production re-solve of all 60 regular states plus direct-min checks."""
    kcol, ccol = parameter_columns(regular)
    rows: List[Dict] = []

    for n, (_, src) in enumerate(regular.iterrows(), start=1):
        kappa = float(src[kcol])
        chi = float(src[ccol])
        print(f"Regular-state reaudit {n:02d}/60: kappa={kappa:g}, chi_ps={chi:g}", flush=True)
        sol = solve_profiles(m, kappa, chi, PRODUCTION_L, PRODUCTION_DZ)

        # Independent minimization at wall, approximate wall-field decay point z~1,
        # and outer edge. This tests branch selection without claiming an analytic proof.
        z = sol["z"]
        II = sol["II"]
        check_indices = sorted(set([0, int(np.argmin(np.abs(z - 1.0))), len(z) - 1]))
        direct_records = []
        for idx in check_indices:
            eta = float(II["eta"][idx])
            theta = float(II["phi_p"][idx] / eta)
            d = direct_local_minimum(m, sol["par"], sol["log_q"][idx], theta, eta)
            d.update({
                "z": float(z[idx]),
                "profile_phi_p": float(II["phi_p"][idx]),
                "profile_eta": eta,
                "profile_g": float(II["g"][idx]),
            })
            d["abs_dphi_p"] = abs(d["direct_phi_p"] - d["profile_phi_p"])
            d["abs_deta"] = abs(d["direct_eta"] - d["profile_eta"])
            d["delta_g_direct_minus_profile"] = d["direct_g"] - d["profile_g"]
            direct_records.append(d)

        rows.append({
            "kappa": kappa,
            "chi_ps": chi,
            "Gamma_I": sol["Gamma_I"],
            "Gamma_II": sol["Gamma_II"],
            "DeltaDelta_Gamma": sol["DeltaDelta_Gamma"],
            "void_excess_II": sol["void_excess_II"],
            "max_delta_eta": sol["max_delta_eta"],
            "caseI_maxR": sol["caseI_maxR"],
            "caseII_maxR": sol["caseII_maxR"],
            "caseII_min_lambda": sol["caseII_min_lambda"],
            "min_competing_gap": sol["min_competing_gap"],
            "caseI_bistable_z_count": sol["caseI_bistable_z_count"],
            "caseII_bistable_z_count": sol["caseII_bistable_z_count"],
            "boundary_z_count": sol["boundary_z_count"],
            "direct_max_abs_dphi_p": max(x["abs_dphi_p"] for x in direct_records),
            "direct_max_abs_deta": max(x["abs_deta"] for x in direct_records),
            "direct_min_delta_g": min(x["delta_g_direct_minus_profile"] for x in direct_records),
            "direct_all_success": all(x["direct_success"] for x in direct_records),
            "direct_records_json": json.dumps(direct_records),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "regular_state_numerical_reaudit.csv", index=False)

    summary = {
        "n_regular": int(len(df)),
        "max_caseI_residual": float(df["caseI_maxR"].max()),
        "max_caseII_residual": float(df["caseII_maxR"].max()),
        "min_caseII_lambda": float(df["caseII_min_lambda"].min()),
        "n_caseI_bistable": int((df["caseI_bistable_z_count"] > 0).sum()),
        "n_caseII_bistable": int((df["caseII_bistable_z_count"] > 0).sum()),
        "n_boundary": int((df["boundary_z_count"] > 0).sum()),
        "max_direct_abs_dphi_p": float(df["direct_max_abs_dphi_p"].max()),
        "max_direct_abs_deta": float(df["direct_max_abs_deta"].max()),
        "min_direct_delta_g": float(df["direct_min_delta_g"].min()),
        "all_direct_minimizations_successful": bool(df["direct_all_success"].all()),
    }
    (outdir / "regular_state_reaudit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df


def r5_grid_audit(m, outdir: Path) -> pd.DataFrame:
    rows = []
    for case, (kappa, chi) in CASES.items():
        for dz in GRID_DZ:
            print(f"Quadrature test {case}: L=20, dz={dz}", flush=True)
            sol = solve_profiles(m, kappa, chi, PRODUCTION_L, dz)
            rows.append({
                "case": case, "kappa": kappa, "chi_ps": chi,
                "L": PRODUCTION_L, "dz": dz,
                "Gamma_I": sol["Gamma_I"], "Gamma_II": sol["Gamma_II"],
                "DeltaDelta_Gamma": sol["DeltaDelta_Gamma"],
                "void_excess_II": sol["void_excess_II"],
                "max_delta_eta": sol["max_delta_eta"],
                "caseII_maxR": sol["caseII_maxR"],
                "caseII_min_lambda": sol["caseII_min_lambda"],
            })
    df = pd.DataFrame(rows)

    # Finest dz is the reference for each case.
    refs = (df.sort_values("dz").groupby("case", as_index=False).first()
            [["case", "DeltaDelta_Gamma", "void_excess_II"]]
            .rename(columns={"DeltaDelta_Gamma": "DD_ref", "void_excess_II": "void_ref"}))
    df = df.merge(refs, on="case", how="left")
    df["DD_rel_to_fine_percent"] = 100.0 * (df["DeltaDelta_Gamma"] - df["DD_ref"]) / df["DD_ref"]
    df["void_rel_to_fine_percent"] = np.where(
        np.abs(df["void_ref"]) > 1e-14,
        100.0 * (df["void_excess_II"] - df["void_ref"]) / df["void_ref"],
        np.nan,
    )
    df.to_csv(outdir / "quadrature_convergence.csv", index=False)
    return df


def r5_domain_audit(m, outdir: Path) -> pd.DataFrame:
    rows = []
    for case, (kappa, chi) in CASES.items():
        for L in DOMAIN_L:
            print(f"Domain-size test {case}: L={L:g}, dz=0.05", flush=True)
            sol = solve_profiles(m, kappa, chi, L, PRODUCTION_DZ)
            rows.append({
                "case": case, "kappa": kappa, "chi_ps": chi,
                "L": L, "dz": PRODUCTION_DZ,
                "Gamma_I": sol["Gamma_I"], "Gamma_II": sol["Gamma_II"],
                "DeltaDelta_Gamma": sol["DeltaDelta_Gamma"],
                "void_excess_II": sol["void_excess_II"],
                "max_delta_eta": sol["max_delta_eta"],
                "caseII_maxR": sol["caseII_maxR"],
                "caseII_min_lambda": sol["caseII_min_lambda"],
            })
    df = pd.DataFrame(rows)
    refs = (df.sort_values("L", ascending=False).groupby("case", as_index=False).first()
            [["case", "DeltaDelta_Gamma", "void_excess_II"]]
            .rename(columns={"DeltaDelta_Gamma": "DD_ref_Lmax", "void_excess_II": "void_ref_Lmax"}))
    df = df.merge(refs, on="case", how="left")
    df["DD_rel_to_Lmax_percent"] = 100.0 * (df["DeltaDelta_Gamma"] - df["DD_ref_Lmax"]) / df["DD_ref_Lmax"]
    df["void_rel_to_Lmax_percent"] = np.where(
        np.abs(df["void_ref_Lmax"]) > 1e-14,
        100.0 * (df["void_excess_II"] - df["void_ref_Lmax"]) / df["void_ref_Lmax"],
        np.nan,
    )
    df.to_csv(outdir / "domain_size_convergence.csv", index=False)
    return df


def make_manuscript_tables(grid: pd.DataFrame, domain: pd.DataFrame, outdir: Path):
    prod_grid = grid[np.isclose(grid["dz"], PRODUCTION_DZ)].copy()
    prod_grid = prod_grid[[
        "case", "kappa", "chi_ps", "DeltaDelta_Gamma", "void_excess_II",
        "DD_rel_to_fine_percent", "caseII_maxR", "caseII_min_lambda"
    ]]
    prod_grid.to_csv(outdir / "quadrature_convergence_manuscript_table.csv", index=False)

    L20 = domain[np.isclose(domain["L"], 20.0)].copy()
    L20 = L20[[
        "case", "kappa", "chi_ps", "DeltaDelta_Gamma", "void_excess_II",
        "DD_rel_to_Lmax_percent", "caseII_maxR", "caseII_min_lambda"
    ]]
    L20.to_csv(outdir / "domain_convergence_manuscript_table.csv", index=False)


def write_text_summary(r4: pd.DataFrame, grid: pd.DataFrame, domain: pd.DataFrame, outdir: Path):
    lines = []
    lines.append("NUMERICAL CONVERGENCE VALIDATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Regular-state reaudit states re-audited: {len(r4)}")
    lines.append(f"R4 max Case-I residual:  {r4['caseI_maxR'].max():.6e}")
    lines.append(f"R4 max Case-II residual: {r4['caseII_maxR'].max():.6e}")
    lines.append(f"R4 min Case-II lambda:   {r4['caseII_min_lambda'].min():.6e}")
    lines.append(f"Regular-state reaudit states with Case-I bistability: {(r4['caseI_bistable_z_count']>0).sum()}")
    lines.append(f"Regular-state reaudit states with Case-II bistability: {(r4['caseII_bistable_z_count']>0).sum()}")
    lines.append(f"Regular-state reaudit states with eta=1 boundary: {(r4['boundary_z_count']>0).sum()}")
    lines.append(f"R4 max direct |dphi_p|: {r4['direct_max_abs_dphi_p'].max():.6e}")
    lines.append(f"R4 max direct |deta|:   {r4['direct_max_abs_deta'].max():.6e}")
    lines.append("")
    lines.append("R5 production-grid relative differences versus dz=0.0125:")
    for _, x in grid[np.isclose(grid['dz'], 0.05)].iterrows():
        lines.append(f"  {x['case']}: {x['DD_rel_to_fine_percent']:+.6f}%")
    lines.append("")
    lines.append("R5 L=20 relative differences versus L=40 at dz=0.05:")
    for _, x in domain[np.isclose(domain['L'], 20.0)].iterrows():
        lines.append(f"  {x['case']}: {x['DD_rel_to_Lmax_percent']:+.6e}%")
    (outdir / "numerical_validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step14", type=Path, default=SCRIPT_DIR / "01_equilibrium_regime_map.py")
    ap.add_argument("--master", type=Path, default=SCRIPT_DIR / "05_manuscript_results_output" / "manuscript_state_table.csv")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "06_numerical_validation_output")
    ap.add_argument("--skip-r4", action="store_true", help="Skip the 60-state R4 re-audit.")
    args = ap.parse_args()

    if not args.step14.exists():
        raise FileNotFoundError(f"Missing equilibrium solver source: {args.step14}")
    if not args.master.exists():
        raise FileNotFoundError(f"Missing manuscript master table: {args.master}")
    args.out.mkdir(parents=True, exist_ok=True)

    m = load_module(args.step14.resolve())
    master = pd.read_csv(args.master)
    regular = identify_regular60(master)
    regular.to_csv(args.out / "regular_state_input_selection.csv", index=False)

    if args.skip_r4:
        r4_path = args.out / "regular_state_numerical_reaudit.csv"
        if not r4_path.exists():
            raise FileNotFoundError("--skip-r4 requested but prior R4 output is absent.")
        r4 = pd.read_csv(r4_path)
    else:
        r4 = r4_regular_audit(m, regular, args.out)

    grid = r5_grid_audit(m, args.out)
    domain = r5_domain_audit(m, args.out)
    make_manuscript_tables(grid, domain, args.out)
    write_text_summary(r4, grid, domain, args.out)

    print("\nCompleted numerical convergence validation.")
    print(f"Outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()
