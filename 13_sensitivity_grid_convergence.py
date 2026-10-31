#!/usr/bin/env python3
"""Run step-12 sensitivity at a finer grid and compare with existing certified CSV.

Default: ALL 48 A/B/C/X tests, certified, dz=0.025, L=20.
Optional --subset nine: A/B/C baseline, wall_scale_0.1, wall_scale_1.5.
No physical model or solver routines are modified.
"""
import argparse, csv, json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NINE = {'baseline', 'wall_scale_0.1', 'wall_scale_1.5'}

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def number(value):
    try: return float(value)
    except (TypeError, ValueError): return math.nan

def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=HERE/'12_parameter_sensitivity_output'/'parameter_sensitivity_all_states.csv')
    p.add_argument('--output',type=Path,default=HERE/'13_sensitivity_convergence_output')
    p.add_argument('--dz',type=float,default=0.025)
    p.add_argument('--L',type=float,default=20.0)
    p.add_argument('--subset',choices=['all','nine'],default='all')
    p.add_argument('--mode',choices=['certified','quick'],default='quick')
    p.add_argument('--reuse-fine',action='store_true',help='Only compare already generated fine-grid CSV')
    args=p.parse_args()
    if args.dz<=0 or args.L<=0: p.error('dz and L must be positive')
    if not args.reference.is_file(): p.error(f'Missing reference CSV: {args.reference}')
    original=read_csv(args.reference)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    fine_dir=out/'fine_grid'
    fine_csv=fine_dir/'parameter_sensitivity_all_states.csv'
    if not args.reuse_fine:
        script=HERE/'12_parameter_sensitivity_analysis.py'
        if not script.is_file(): p.error(f'Missing {script}')
        cmd=[sys.executable,str(script),'--mode',args.mode,'--states',
             'A,B,C,X' if args.subset=='all' else 'A,B,C',
             '--L',str(args.L),'--dz',str(args.dz),'--output',str(fine_dir)]
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run(cmd,check=True)
    if not fine_csv.is_file(): p.error(f'Missing fine-grid CSV: {fine_csv}')
    fine=read_csv(fine_csv)
    old={(r['state'],r['test']):r for r in original}
    chosen=[r for r in fine if args.subset=='all' or
            (r['state'] in 'ABC' and r['test'] in NINE)]
    metrics=['gamma_I','gamma_II','delta_gamma','relative_enhancement_percent',
             'void_excess_II','weak_delta_gamma','weak_void_excess']
    results=[]
    for f in chosen:
        key=(f['state'],f['test']); b=old.get(key)
        r={'state':key[0],'test':key[1], 'reference_found':b is not None,
           'reference_ok':b.get('ok','') if b else '', 'fine_ok':f.get('ok',''),
           'reference_mode':b.get('mode','') if b else '', 'fine_mode':f.get('mode',''),
           'reference_dz':b.get('dz','') if b else '', 'fine_dz':f.get('dz',''),
           'reference_L':b.get('L','') if b else '', 'fine_L':f.get('L',''),
           'reference_bulk_class':b.get('bulk_class','') if b else '',
           'fine_bulk_class':f.get('bulk_class',''),
           'reference_regular':b.get('regular_eligible','') if b else '',
           'fine_regular':f.get('regular_eligible',''),
           'fine_reason':f.get('reason',''),
           'fine_min_eigenvalue':f.get('min_eigenvalue',''),
           'fine_max_residual':f.get('max_residual','')}
        for m in metrics:
            x=number(b.get(m)) if b else math.nan
            y=number(f.get(m))
            r[f'{m}_reference']=x;r[f'{m}_fine']=y
            r[f'{m}_absolute_change']=abs(y-x)
            # Use reference signal, not an arbitrary small epsilon, as denominator.
            r[f'{m}_relative_change']=abs(y-x)/abs(x) if abs(x)>1e-12 else math.nan
        x=number(b.get('delta_gamma')) if b else math.nan
        y=number(f.get('delta_gamma'))
        r['delta_gamma_sign_consistent']=(x*y>0) if abs(x)>1e-9 and abs(y)>1e-9 else 'UNRESOLVED_NEAR_ZERO'
        r['delta_gamma_resolved']=abs(x)>1e-9 and abs(y)>1e-9
        results.append(r)
    write_csv(out/'grid_convergence_comparison.csv',results)
    summary=[]
    for state in sorted(set(r['state'] for r in results)):
        subset=[r for r in results if r['state']==state]
        valid=[r for r in subset if r['fine_ok']=='True' and r['reference_ok']=='True']
        summary.append({'state':state,'n_compared':len(subset),'n_success_both':len(valid),
          'n_unresolved_sign':sum(r['delta_gamma_sign_consistent']=='UNRESOLVED_NEAR_ZERO' for r in valid),
          'max_abs_delta_gamma_change':max((r['delta_gamma_absolute_change'] for r in valid),default=math.nan),
          'max_relative_delta_gamma_change_resolved':max((r['delta_gamma_relative_change'] for r in valid if r['delta_gamma_resolved']),default=math.nan),
          'max_relative_enhancement_percent_change':max((r['relative_enhancement_percent_absolute_change'] for r in valid),default=math.nan),
          'min_fine_hessian_eigenvalue':min((number(r['fine_min_eigenvalue']) for r in valid),default=math.nan)})
    write_csv(out/'grid_convergence_summary.csv',summary)
    print('Comparison written:',out/'grid_convergence_comparison.csv',flush=True)
    for r in summary: print(r,flush=True)

if __name__=='__main__':main()
