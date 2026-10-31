#!/usr/bin/env python3
"""Refined finite-wall void-excess crossover with root bracketing and audits.

Requires 01_equilibrium_regime_map.py alongside this file.
quick: local stationary continuation (exploratory only).
certified: production multi-minimum profile solvers + bulk-global audit;
           certification is conditional on their search completeness.
Outputs pointwise and root CSV files; optional grid/domain checks.
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
from scipy.optimize import brentq, root

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / '01_equilibrium_regime_map.py'
if not MODEL_PATH.exists():
    raise SystemExit(f'Missing required model: {MODEL_PATH}')
spec = importlib.util.spec_from_file_location('equilibrium_model', MODEL_PATH)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def evaluate(par, wall, z, mode, bulk_audit=None):
    lq = model.local_state_quantities(z, par, wall)['log_q']
    xb, _, eb = model.bulk_quantities(par)
    hb = model.local_hessian(xb, eb, par)
    if hb['lambda_min'] <= 1e-9:
        return {'ok': False, 'reason': 'bulk_locally_unstable'}
    if mode == 'certified':
        audit = bulk_audit if bulk_audit is not None else model.audit_bulk_global(par)
        if audit['bulk_class'] != 'BG':
            return {'ok': False, 'reason': 'bulk_' + audit['bulk_class']}
        a = model.solve_caseI_profile(z, lq, par)
        b = model.solve_caseII_profile(z, lq, par)
        if not a['success'] or not b['success']:
            return {'ok': False, 'reason': 'profile_solver_failure'}
        if a['bistable'] or b['bistable'] or b['boundary']:
            return {'ok': False, 'reason': 'nonregular_surface_state'}
        x1, x2, eta2 = a['phi_p'], b['phi_p'], b['eta']
        min_eig = float(np.nanmin(b['lambda_min']))
        max_res = float(max(np.nanmax(a['Rmax']), np.nanmax(b['Rmax'])))
    else:
        x1 = np.empty(len(z)); x2 = np.empty(len(z)); eta2 = np.empty(len(z))
        last_x, last_eta = xb, eb
        min_eig, max_res = float('inf'), 0.0
        for i in range(len(z)-1, -1, -1):
            logq = float(lq[i])
            try:
                x1[i] = brentq(lambda x: model.residual_x(x, eb, logq, par),
                                1e-12, eb-1e-12, xtol=1e-13)
            except ValueError:
                return {'ok': False, 'reason': 'caseI_root_failure'}
            def fun(v):
                x, e = v
                if not (1e-12 < x < e-1e-12 and e < 1):
                    return np.array([1e4 + abs(x), 1e4 + abs(e)])
                return np.array([model.residual_x(x,e,logq,par),
                                 model.residual_eta(x,e,par)])
            sol = root(fun, [last_x, last_eta], tol=1e-11)
            residual = float(np.max(np.abs(fun(sol.x))))
            if residual > 1e-8:
                return {'ok': False, 'reason': 'caseII_root_failure'}
            x,e = map(float,sol.x)
            if not (0 < x < e < 1):
                return {'ok': False, 'reason': 'out_of_bounds'}
            lam = float(model.local_hessian(x,e,par)['lambda_min'])
            if lam <= 1e-9:
                return {'ok': False, 'reason': 'caseII_local_instability'}
            x2[i], eta2[i] = x,e
            last_x,last_eta=x,e
            min_eig=min(min_eig,lam); max_res=max(max_res,residual)
    B = (1-1/par.r_p)/eb - 1/(eb-xb) + par.chi_ps - par.chi_ss
    D = hb['det']; A=hb['Hxx']
    w = (lq-model.bulk_log_q(par))/par.r_p
    return {'ok': True, 'reason':'OK',
            'gamma_I':float(model.integrate_profile(x1-xb,z)),
            'gamma_II':float(model.integrate_profile(x2-xb,z)),
            'delta_gamma':float(model.integrate_profile(x2-x1,z)),
            'void_excess_II':float(model.integrate_profile(eb-eta2,z)),
            'weak_void_excess':float((B/D)*model.integrate_profile(w,z)),
            'weak_delta_gamma':float((B*B/(A*D))*model.integrate_profile(w,z)),
            'H_xeta_bulk':float(B),'min_eigenvalue':min_eig,
            'max_residual':max_res}


def write_csv(path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('quick','certified'),default='quick')
    p.add_argument('--kappas',default='5,20,50')
    p.add_argument('--scales',default='0.1,0.5,1,1.5')
    p.add_argument('--chi-min',type=float,default=0.10)
    p.add_argument('--chi-max',type=float,default=0.17)
    p.add_argument('--chi-count',type=int,default=15)
    p.add_argument('--root-xtol',type=float,default=1e-7)
    p.add_argument('--L',type=float,default=12.)
    p.add_argument('--dz',type=float,default=.2)
    p.add_argument('--convergence',action='store_true',help='re-evaluate roots at dz/2 and 1.5L')
    p.add_argument('--output',type=Path,default=HERE/'10_finite_wall_crossover_refined_output')
    args=p.parse_args()
    if args.chi_count < 2 or args.chi_min >= args.chi_max or args.dz<=0 or args.L<=0 or args.root_xtol<=0:
        p.error('Invalid chi grid, root tolerance, L or dz')
    kappas=[float(v) for v in args.kappas.split(',')]
    scales=[float(v) for v in args.scales.split(',')]
    if any(k<=0 for k in kappas) or any(s<=0 for s in scales):
        p.error('kappas and scales must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    z=model.make_grid(args.L,args.dz)
    base=model.ModelParams(); wallbase=model.WallParams()
    weak_chi=1/(1-base.phi_v_b-base.phi_p_b)-(1-1/base.r_p)/(1-base.phi_v_b)+base.chi_ss
    rows=[]; roots=[]; audit_cache={}
    for k in kappas:
        for s in scales:
            wall=replace(wallbase,eps_N=wallbase.eps_N*s,eps_D=wallbase.eps_D*s)
            cache={}
            def sample(chi, grid=z, save=True):
                chi=float(chi)
                key=(round(chi,14),len(grid),float(grid[-1]))
                if key in cache:
                    return cache[key]
                par=replace(base,kappa=k,chi_ps=chi)
                audit=None
                if args.mode=='certified':
                    ak=(k,round(chi,14))
                    if ak not in audit_cache:
                        audit_cache[ak]=model.audit_bulk_global(par)
                    audit=audit_cache[ak]
                result=evaluate(par,wall,grid,args.mode,audit)
                record={'kappa':k,'wall_scale':s,'eps_N':wall.eps_N,'eps_D':wall.eps_D,
                        'chi_ps':chi,'mode':args.mode,'L':float(grid[-1]),
                        'dz':float(grid[1]-grid[0]),**result}
                cache[key]=record
                if save: rows.append(record)
                return record
            coarse=[sample(chi) for chi in np.linspace(args.chi_min,args.chi_max,args.chi_count)]
            for left,right in zip(coarse,coarse[1:]):
                if not(left['ok'] and right['ok']):
                    continue
                f0=left['void_excess_II'];f1=right['void_excess_II']
                if f0*f1>0 and f0!=0 and f1!=0:
                    continue
                a,b=left['chi_ps'],right['chi_ps']
                try:
                    def f(chi):
                        result=sample(chi)
                        if not result['ok']:
                            raise RuntimeError(f'Invalid intermediate state at chi={chi:.9g}: {result["reason"]}')
                        return result['void_excess_II']
                    cross=a if f0==0 else b if f1==0 else brentq(f,a,b,xtol=args.root_xtol,rtol=1e-12)
                    rr=sample(cross)
                    item={'kappa':k,'wall_scale':s,'mode':args.mode,'chi_cross':cross,
                          'chi_weak_wall':weak_chi,'shift':cross-weak_chi,
                          'bracket_left':a,'bracket_right':b,
                          'void_at_root':rr['void_excess_II'],
                          'delta_gamma_at_root':rr['delta_gamma'],
                          'min_eigenvalue_at_root':rr['min_eigenvalue'],
                          'max_residual_at_root':rr['max_residual'],
                          'status':'root_found'}
                    if args.convergence:
                        for label,grid in [('fine_grid',model.make_grid(args.L,args.dz/2)),
                                           ('long_domain',model.make_grid(1.5*args.L,args.dz))]:
                            chk=sample(cross,grid)
                            item[label+'_ok']=chk['ok']
                            item[label+'_void_at_base_root']=chk.get('void_excess_II',math.nan)
                            if chk['ok']:
                                # Rebracket/re-solve with refined grid; failure is recorded, not hidden.
                                def fgrid(c):
                                    v=sample(c,grid)
                                    if not v['ok']:
                                        raise RuntimeError(v['reason'])
                                    return v['void_excess_II']
                                try:
                                    fa,fb=fgrid(a),fgrid(b)
                                    if fa*fb>0:
                                        raise RuntimeError('No sign change on refined grid')
                                    cnew=brentq(fgrid,a,b,xtol=args.root_xtol,rtol=1e-12)
                                    item[label+'_chi_cross']=cnew
                                    item[label+'_chi_shift']=cnew-cross
                                except (RuntimeError,ValueError) as exc:
                                    item[label+'_error']=str(exc)
                    roots.append(item)
                    print(f'kappa={k:g} scale={s:g}: root={cross:.9f} shift={cross-weak_chi:+.7f}',flush=True)
                except (RuntimeError,ValueError) as exc:
                    roots.append({'kappa':k,'wall_scale':s,'mode':args.mode,
                                  'bracket_left':a,'bracket_right':b,
                                  'status':'root_failed','error':str(exc)})
                    print(f'kappa={k:g} scale={s:g}: root failed: {exc}',flush=True)
            if not any(r['kappa']==k and r['wall_scale']==s for r in roots):
                roots.append({'kappa':k,'wall_scale':s,'mode':args.mode,
                              'status':'no_bracket_in_range'})
    write_csv(out/'finite_wall_crossover_points_refined.csv',rows)
    write_csv(out/'finite_wall_crossover_roots.csv',roots)
    print(f'Wrote {out}; {sum(r.get("status")=="root_found" for r in roots)} roots. Mode={args.mode}')
    if args.mode=='quick':
        print('WARNING: QUICK mode is NOT a global-equilibrium certification.')
    else:
        print('NOTE: Certified mode uses production candidate searches; independent convergence remains important.')

if __name__=='__main__':
    main()
