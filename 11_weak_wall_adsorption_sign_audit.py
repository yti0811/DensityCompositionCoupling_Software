#!/usr/bin/env python3
"""Focused s=0.1 adsorption-difference audit near the void crossover.

Uses the original 01 equilibrium model through the existing refined Step 10
implementation. No change to the underlying thermodynamic model.

This is a convergence / numerical-sign diagnostic, NOT a proof that a tiny
negative difference is either numerical noise or a physical nonlinear effect.
"""
import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STEP10 = HERE / '10_finite_wall_crossover_refined.py'
if not STEP10.exists():
    raise SystemExit(f'Missing {STEP10}; place scripts 01, 10, 11 together.')
spec = importlib.util.spec_from_file_location('step10_refined', STEP10)
step10 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step10
spec.loader.exec_module(step10)
model = step10.model


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(key for rec in records for key in rec))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots', type=Path, default=HERE/'10_finite_wall_crossover_refined_output'/'finite_wall_crossover_roots.csv')
    ap.add_argument('--mode', choices=['quick','certified'], default='quick')
    ap.add_argument('--kappas', default='5,20,50')
    ap.add_argument('--wall-scale', type=float, default=0.1)
    ap.add_argument('--offsets', default='-0.002,-0.0005,0,0.0005,0.002')
    ap.add_argument('--dzs', default='0.1,0.05,0.025')
    ap.add_argument('--Ls', default='20,30')
    ap.add_argument('--zero-tolerance', type=float, default=1e-9,
                    help='Absolute reporting threshold only; NOT an error bound.')
    ap.add_argument('--output', type=Path, default=HERE/'11_weak_wall_sign_audit_output')
    args = ap.parse_args()
    kappas = [float(v) for v in args.kappas.split(',')]
    offsets = [float(v) for v in args.offsets.split(',')]
    dzs = [float(v) for v in args.dzs.split(',')]
    Ls = [float(v) for v in args.Ls.split(',')]
    if min(dzs) <= 0 or min(Ls) <= 0 or args.wall_scale <= 0 or args.zero_tolerance <= 0:
        ap.error('dz, L, wall-scale, zero-tolerance must be positive')
    if not args.roots.is_file():
        ap.error(f'Missing roots CSV: {args.roots}')
    with args.roots.open(newline='', encoding='utf-8-sig') as f:
        roots = list(csv.DictReader(f))
    centers = {}
    for r in roots:
        if r.get('status') == 'root_found':
            k = float(r['kappa']); s = float(r['wall_scale'])
            if k in kappas and math.isclose(s, args.wall_scale, abs_tol=1e-12):
                centers[k] = float(r['chi_cross'])
    missing = set(kappas)-set(centers)
    if missing:
        ap.error(f'Missing crossover roots for kappa={sorted(missing)} and s={args.wall_scale}')

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = model.ModelParams()
    wallbase = model.WallParams()
    wall = replace(wallbase, eps_N=wallbase.eps_N*args.wall_scale,
                   eps_D=wallbase.eps_D*args.wall_scale)
    records = []
    audits = {}
    for k in kappas:
        for offset in offsets:
            chi = centers[k] + offset
            par = replace(base, kappa=k, chi_ps=chi)
            audit = None
            if args.mode == 'certified':
                audit = model.audit_bulk_global(par)
                audits[f'{k:g}|{chi:.12g}'] = audit.get('bulk_class')
            for L in Ls:
                for dz in dzs:
                    z = model.make_grid(L, dz)
                    r = step10.evaluate(par, wall, z, args.mode, audit)
                    row = {'kappa':k, 'wall_scale':args.wall_scale,
                           'chi_ps':chi,'offset_from_crossover':offset,
                           'reference_crossover':centers[k], 'mode':args.mode,
                           'L':float(z[-1]),'dz':float(z[1]-z[0]), **r}
                    if r['ok']:
                        diff = r['delta_gamma']
                        row['difference_of_integrals'] = r['gamma_II']-r['gamma_I']
                        row['cancellation_discrepancy'] = diff-row['difference_of_integrals']
                        row['sign_with_reporting_threshold'] = (
                            'positive' if diff > args.zero_tolerance else
                            'negative' if diff < -args.zero_tolerance else 'unresolved_near_zero')
                        row['relative_difference_percent'] = (
                            100*diff/r['gamma_I'] if r['gamma_I'] != 0 else math.nan)
                    records.append(row)
                    print(f'k={k:g} chi={chi:.9f} L={L:g} dz={dz:g}: '
                          f'{r.get("delta_gamma",float("nan")):+.5e} [{r["reason"]}]', flush=True)
    write_csv(out/'adsorption_sign_convergence_points.csv',records)
    comparisons = []
    for k in kappas:
        for offset in offsets:
            subset = [r for r in records if r['kappa']==k and r['offset_from_crossover']==offset and r['ok']]
            if not subset:
                comparisons.append({'kappa':k,'offset':offset,'status':'no_successful_evaluations'})
                continue
            vals = [r['delta_gamma'] for r in subset]
            spread = max(vals)-min(vals)
            signs = set(np.sign(vals))
            max_abs = max(abs(v) for v in vals)
            comparisons.append({'kappa':k,'offset':offset,'chi_ps':subset[0]['chi_ps'],
                'n_success':len(subset),'n_requested':len(Ls)*len(dzs),
                'min_delta_gamma':min(vals),'max_delta_gamma':max(vals),
                'range_across_grids_and_domains':spread,
                'max_abs_delta_gamma':max_abs,
                'all_abs_below_reporting_threshold':max_abs <= args.zero_tolerance,
                'sign_consistent_across_grids':len(signs)==1,
                'spread_exceeds_signal':spread >= max_abs,
                'status': ('insufficient_successful_evaluations' if len(subset)!=len(Ls)*len(dzs)
                           else 'computed_not_certified')})
    write_csv(out/'adsorption_sign_convergence_summary.csv',comparisons)
    print(f'Wrote {out} ({len(records)} evaluations).')
    print('Interpretation: inspect convergence and cancellation discrepancy; do not label near-zero values physical without further verification.')

if __name__ == '__main__':
    main()
