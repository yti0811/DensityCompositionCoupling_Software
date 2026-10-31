#!/usr/bin/env python3
"""Reviewer 1 comment 2: limited one-factor-at-a-time sensitivity audit.

Requires the unmodified 01_equilibrium_regime_map.py and
10_finite_wall_crossover_refined.py in the same directory.

Only bulk-global, non-bistable, non-boundary solutions are counted as regular.
'certified' means the production multi-start candidate search, NOT a rigorous
proof of global optimality. Weak-wall predictions are first-order diagnostics.
"""
import argparse
import csv
import json
import math
import sys
import importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P10 = HERE / '10_finite_wall_crossover_refined.py'
if not P10.exists():
    raise SystemExit('Missing 10_finite_wall_crossover_refined.py next to script 12')
spec = importlib.util.spec_from_file_location('sensitivity_step10', P10)
step10 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step10
spec.loader.exec_module(step10)
model = step10.model

# Case B/C values transcribed from the uploaded revised manuscript (p. 6).
STATES = {'A': (20.0, 0.2), 'B': (2.0, -0.6), 'C': (0.5, 0.4),
          'X': (20.0, 0.127900396)}


def experiments(base, wall, include_cross):
    cases = [('baseline', base, wall, 'baseline')]
    for name, vals in [('eps_N', [0.5, 1.5]), ('eps_D', [1.5, 4.5])]:
        for val in vals:
            cases.append((f'{name}_{val:g}', base, replace(wall, **{name:val}), name))
    for scale in [0.1, 0.5, 1.5]:
        cases.append((f'wall_scale_{scale:g}', base,
                      replace(wall, eps_N=wall.eps_N*scale, eps_D=wall.eps_D*scale), 'wall_scale'))
    for val in [3.0, 7.0]:
        cases.append((f'delta_u_int_{val:g}', replace(base, delta_u_int=val), wall, 'delta_u_int'))
    for val in [0.01, 0.04]:
        cases.append((f'phi_p_b_{val:g}', replace(base, phi_p_b=val), wall, 'phi_p_b'))
    return cases


def save_csv(path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def classify_error(exact, approx, floor):
    absolute=abs(exact-approx)
    return (absolute, absolute/abs(exact) if abs(exact)>floor else float('nan'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['quick','certified'], default='quick')
    ap.add_argument('--states', default='A,B,C,X', help='Comma-separated subset of A,B,C,X')
    ap.add_argument('--L', type=float, default=20)
    ap.add_argument('--dz', type=float, default=0.05)
    ap.add_argument('--error-floor', type=float, default=1e-9,
                    help='Suppress relative errors below this absolute response; not an uncertainty bound')
    ap.add_argument('--output', type=Path, default=HERE/'12_parameter_sensitivity_output')
    ap.add_argument('--smoke', action='store_true', help='Only baseline and eps_N=0.5 for selected states')
    args=ap.parse_args()
    names=[s.strip() for s in args.states.split(',') if s.strip()]
    if not names or any(s not in STATES for s in names):
        ap.error('states must be a subset of A,B,C,X')
    if args.L<=0 or args.dz<=0 or args.error_floor<=0:
        ap.error('L, dz and error-floor must be positive')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    z=model.make_grid(args.L,args.dz)
    base=model.ModelParams(); wallbase=model.WallParams()
    rows=[]; audits={}
    for state in names:
        kappa,chi=STATES[state]
        statebase=replace(base,kappa=kappa,chi_ps=chi)
        tests=experiments(statebase,wallbase,state=='X')
        if args.smoke:
            tests=tests[:2]
        for label,par,wall,factor in tests:
            rec={'state':state,'test':label,'factor':factor,'mode':args.mode,
                 'kappa':par.kappa,'chi_ps':par.chi_ps,'phi_p_b':par.phi_p_b,
                 'phi_v_b':par.phi_v_b,'delta_u_int':par.delta_u_int,
                 'eps_N':wall.eps_N,'eps_D':wall.eps_D,'L':float(z[-1]),
                 'dz':float(z[1]-z[0])}
            try:
                xb,_,eta_b=model.bulk_quantities(par)
                hb=model.local_hessian(xb,eta_b,par)
                rec['bulk_lambda_min']=float(hb['lambda_min'])
                rec['bulk_det']=float(hb['det'])
                if args.mode=='certified':
                    key=(par.kappa,par.chi_ps,par.phi_p_b,par.delta_u_int)
                    if key not in audits:
                        audits[key]=model.audit_bulk_global(par)
                    audit=audits[key]
                    rec['bulk_class']=audit['bulk_class']
                else:
                    audit=None
                    rec['bulk_class']='NOT_AUDITED'
                res=step10.evaluate(par,wall,z,args.mode,audit)
                rec.update(res)
                rec['regular_eligible']=bool(res.get('ok') and
                                             (args.mode!='certified' or rec['bulk_class']=='BG'))
                if res.get('ok'):
                    gamma=res['gamma_I']
                    rec['relative_enhancement_percent']=(100*res['delta_gamma']/gamma
                                                         if abs(gamma)>1e-14 else math.nan)
                    for prefix,actual,approx in [('adsorption',res['delta_gamma'],res['weak_delta_gamma']),
                                                  ('void',res['void_excess_II'],res['weak_void_excess'])]:
                        ae,re=classify_error(actual,approx,args.error_floor)
                        rec[f'{prefix}_absolute_error']=ae
                        rec[f'{prefix}_relative_error']=re
                        rec[f'{prefix}_weak_sign_agrees']=(
                            math.copysign(1,actual)==math.copysign(1,approx)
                            if abs(actual)>args.error_floor and abs(approx)>args.error_floor
                            else 'UNRESOLVED_NEAR_ZERO')
                    rec['adsorption_10pct_agreement']=(
                        bool(rec['adsorption_relative_error']<=0.1)
                        if math.isfinite(rec['adsorption_relative_error']) else 'UNRESOLVED_NEAR_ZERO')
                    rec['void_10pct_agreement']=(
                        bool(rec['void_relative_error']<=0.1)
                        if math.isfinite(rec['void_relative_error']) else 'UNRESOLVED_NEAR_ZERO')
                print(f'{state:1s} {label:19s} {res.get("reason", "UNKNOWN"):25s} '
                      f'DGamma={res.get("delta_gamma", float("nan")):+.5e}',flush=True)
            except Exception as exc:
                rec.update({'ok':False,'reason':f'{type(exc).__name__}: {exc}',
                            'regular_eligible':False})
                print(f'{state:1s} {label:19s} ERROR: {exc}',flush=True)
            rows.append(rec)
    save_csv(out/'parameter_sensitivity_all_states.csv',rows)
    regular=[r for r in rows if r.get('regular_eligible')]
    save_csv(out/'parameter_sensitivity_regular_states.csv',regular)
    save_csv(out/'weak_wall_validity.csv',regular)
    summary=[]
    for state in names:
        subset=[r for r in rows if r['state']==state]
        good=[r for r in subset if r.get('regular_eligible')]
        baseline=next((r for r in good if r['test']=='baseline'),None)
        summary.append({'state':state,'n_tests':len(subset),'n_regular':len(good),
                        'n_failed_or_excluded':len(subset)-len(good),
                        'n_positive_delta_gamma':sum(r['delta_gamma']>args.error_floor for r in good),
                        'n_negative_delta_gamma':sum(r['delta_gamma']< -args.error_floor for r in good),
                        'n_near_zero_delta_gamma':sum(abs(r['delta_gamma'])<=args.error_floor for r in good),
                        'baseline_delta_gamma':baseline['delta_gamma'] if baseline else math.nan,
                        'min_delta_gamma':min((r['delta_gamma'] for r in good),default=math.nan),
                        'max_delta_gamma':max((r['delta_gamma'] for r in good),default=math.nan)})
    save_csv(out/'parameter_sensitivity_summary.csv',summary)
    print(f'Wrote {out}: {len(rows)} rows; {len(regular)} eligible ({args.mode}).')

if __name__=='__main__':
    main()
