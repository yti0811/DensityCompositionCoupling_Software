#!/usr/bin/env python3
"""Compute Eq.(30) excess grand potential for Cases I/II using the original production solver."""
import argparse, importlib.util, csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('production', HERE/'01_equilibrium_regime_map.py')
m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def run(name, kappa, chi, L, dz):
    par=m.ModelParams(kappa=kappa,chi_ps=chi)
    wall=m.WallParams()
    bulk=m.audit_bulk_global(par)
    if bulk['bulk_class']!='BG':
        return {'state':name,'status':'excluded_bulk_'+str(bulk['bulk_class'])}
    z=m.make_grid(L,dz)
    q=m.local_state_quantities(z,par,wall)['log_q']
    I=m.solve_caseI_profile(z,q,par)
    II=m.solve_caseII_profile(z,q,par)
    if not I['success'] or not II['success']:
        return {'state':name,'status':'solver_failed'}
    xb,_,etab=m.bulk_quantities(par)
    gb=m.local_grand_density(xb,etab,m.bulk_log_q(par),par)
    gi=np.array([m.local_grand_density(float(x),float(e),float(lq),par) for x,e,lq in zip(I['phi_p'],I['eta'],q)])
    gii=np.array([m.local_grand_density(float(x),float(e),float(lq),par) for x,e,lq in zip(II['phi_p'],II['eta'],q)])
    exI=m.integrate_profile(gi-gb,z)
    exII=m.integrate_profile(gii-gb,z)
    delta=m.integrate_profile(gii-gi,z)
    gap=gii-gi
    tol=1e-9
    return dict(state=name,kappa=kappa,chi_ps=chi,L=L,dz=dz,status='ok',bulk_class=bulk['bulk_class'],
                gamma_ex_I=exI,gamma_ex_II=exII,delta_gamma_ex=delta,
                difference_of_excesses=exII-exI,arithmetic_discrepancy=delta-(exII-exI),
                max_local_positive_gap=float(np.max(gap)),min_local_gap=float(np.min(gap)),
                num_local_gap_above_tolerance=int(np.sum(gap>tol)),
                variational_sign_ok=bool(delta<=tol),
                adsorption_difference=m.integrate_profile(II['phi_p']-I['phi_p'],z),
                caseI_bistable=bool(I['bistable']),caseII_bistable=bool(II['bistable']),
                caseII_boundary=bool(II['boundary']),max_caseII_residual=float(np.max(II['Rmax'])),
                min_caseII_eigenvalue=float(np.nanmin(II['lambda_min'])))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--L',type=float,default=20)
    p.add_argument('--dz',type=float,default=.05)
    p.add_argument('--states',default='A,B,C')
    p.add_argument('--output',default=str(HERE/'14_grand_potential_output'))
    a=p.parse_args()
    states={'A':(20,.2),'B':(2,-.6),'C':(.5,.4),'X':(20,.1279003961516696)}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    names=[k.strip() for k in a.states.split(',') if k.strip()]
    if not names or any(k not in states for k in names):
        p.error('states must be a subset of A,B,C,X')
    if a.L<=0 or a.dz<=0:
        p.error('L and dz must be positive')
    rows=[run(k,*states[k],a.L,a.dz) for k in names]
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'excess_grand_potential_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote',out/'excess_grand_potential_comparison.csv')
if __name__=='__main__':main()
