import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from scipy.optimize import least_squares, minimize, brentq
from pathlib import Path

# ============================================================
# EQUILIBRIUM REGIME MAP
# FINAL ADMISSIBLE REGIME MAP
# ============================================================
# Combines:
#   - Step 12 interfacial classification
#   - Step 13 / 13R bulk global-equilibrium certification
#
# Final hierarchy:
#   BU      : prescribed bulk locally unstable
#   BM      : prescribed bulk locally stable but metastable
#   BR      : bulk global status unresolved
#   BG-BI   : bulk-global, but Case-I surface/local bistability
#   BG-BII  : bulk-global, Case-II-only local bistability
#   BG-K+/-/0 : bulk-global, KKT eta=1 boundary profile
#   BG-I+/-/0 : bulk-global, unique interior profile
#
# Files written to OUTPUT_DIR:
#   equilibrium_all_states.csv
#   equilibrium_bulk_global_states.csv
#   equilibrium_classification_matrix.csv
#   equilibrium_regime_map.png
#   adsorption_response_map.png
#   free_volume_response_map.png
#   equilibrium_summary.txt
# ============================================================

@dataclass
class ModelParams:
    r_p: float = 10.0
    phi_p_b: float = 0.02
    phi_v_b: float = 0.05
    chi_pp: float = -0.5
    chi_ps: float = 0.2
    chi_ss: float = 0.0
    kappa: float = 20.0
    delta_u_int: float = 5.0

@dataclass
class WallParams:
    eps_N: float = 1.0
    eps_D: float = 3.0
    lambda_w: float = 1.0

# ---------------- numerical tolerances ----------------
EPS = 1e-10
ROOT_TOL = 1e-9
HESS_TOL = 1e-9
MIN_DUP_TOL = 2e-6
EFFECT_ABS_TOL = 1e-8
DIRECT_LOWER_TRIGGER = 1e-6
GLOBAL_G_TOL = 1e-8
BOUNDARY_DETECT_TOL = 2e-5


def bulk_quantities(par):
    x_b = par.phi_p_b
    eta_b = 1.0 - par.phi_v_b
    s_b = eta_b - x_b
    if not (0 < x_b < eta_b < 1):
        raise ValueError('Require 0 < phi_p_b < eta_b < 1')
    return x_b, s_b, eta_b


def bulk_log_q(par):
    return np.logaddexp(0.0, -par.delta_u_int)


def make_grid(L=20.0, dz=0.05):
    return np.arange(int(np.floor(L/dz)) + 1) * dz


def wall_potentials(z, wall):
    decay = np.exp(-np.asarray(z, float) / wall.lambda_w)
    return -wall.eps_N * decay, -wall.eps_D * decay


def local_state_quantities(z, par, wall):
    U_N, U_Dw = wall_potentials(z, wall)
    E_N = U_N
    E_D = par.delta_u_int + U_Dw
    log_q = np.logaddexp(-E_N, -E_D)
    P_N = np.exp(-E_N - log_q)
    P_D = np.exp(-E_D - log_q)
    return {'log_q': log_q, 'P_N': P_N, 'P_D': P_D}


def bulk_chemical_potentials(par):
    x_b, s_b, eta_b = bulk_quantities(par)
    lqb = bulk_log_q(par)
    r = par.r_p
    mu_s = (np.log(s_b/eta_b)
            + (1 - 1/r) * x_b/eta_b
            + par.chi_ps*x_b + par.chi_ss*s_b)
    mu_p = (np.log(x_b/eta_b) + 1
            - r*(np.log(s_b/eta_b) + 1)
            + r*(par.chi_pp*x_b + par.chi_ps*(s_b-x_b) - par.chi_ss*s_b)
            - lqb + r*mu_s)
    return mu_p, mu_s


def interaction_I(x, s, par):
    return par.chi_pp*x + par.chi_ps*(s-x) - par.chi_ss*s


def residual_x(x, eta, log_q, par):
    if not (0 < x < eta <= 1):
        return np.nan
    s = eta - x
    x_b, s_b, eta_b = bulk_quantities(par)
    return ((1/par.r_p)*np.log((x/eta)/(x_b/eta_b))
            - np.log((s/eta)/(s_b/eta_b))
            + interaction_I(x, s, par) - interaction_I(x_b, s_b, par)
            - (1/par.r_p)*(log_q - bulk_log_q(par)))


def residual_eta(x, eta, par):
    if not (0 < x < eta <= 1):
        return np.nan
    s = eta - x
    x_b, s_b, eta_b = bulk_quantities(par)
    return (np.log((s/eta)/(s_b/eta_b))
            + (1 - 1/par.r_p)*(x/eta - x_b/eta_b)
            + par.kappa*(eta-eta_b)
            + par.chi_ps*(x-x_b)
            + par.chi_ss*(s-s_b))


def local_grand_density(x, eta, log_q, par):
    if not (0 < x < eta <= 1):
        return np.inf
    s = eta - x
    mu_p, mu_s = bulk_chemical_potentials(par)
    _, _, eta_b = bulk_quantities(par)
    r = par.r_p
    g_mix = (x/r)*np.log(x/eta) + s*np.log(s/eta)
    g_eos = 0.5*par.kappa*(eta-eta_b)**2
    g_int = 0.5*par.chi_pp*x*x + par.chi_ps*x*s + 0.5*par.chi_ss*s*s
    g_state = -(x/r)*log_q
    g_mu = -(x/r)*mu_p - s*mu_s
    return g_mix + g_eos + g_int + g_state + g_mu


def local_hessian(x, eta, par):
    s = eta - x
    r = par.r_p
    Hxx = 1/(r*x) + 1/s + par.chi_pp - 2*par.chi_ps + par.chi_ss
    Hxe = (1 - 1/r)/eta - 1/s + par.chi_ps - par.chi_ss
    Hee = (1/s - 1/eta - (1 - 1/r)*x/eta**2 + par.kappa + par.chi_ss)
    H = np.array([[Hxx, Hxe], [Hxe, Hee]], float)
    eig = np.linalg.eigvalsh(H)
    return {'Hxx': Hxx, 'det': np.linalg.det(H), 'lambda_min': eig[0]}


def sigmoid(a):
    if a >= 0:
        e = np.exp(-a)
        return 1/(1+e)
    e = np.exp(a)
    return e/(1+e)


def logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p/(1-p))


def fields_from_u(u):
    theta = sigmoid(u[0])
    eta = sigmoid(u[1])
    x = eta*theta
    return x, eta-x, eta, theta


def transformed_EL(u, log_q, par):
    x, s, eta, theta = fields_from_u(u)
    Rx = residual_x(x, eta, log_q, par)
    Re = residual_eta(x, eta, par)
    if not (np.isfinite(Rx) and np.isfinite(Re)):
        return np.array([1e6, 1e6])
    return np.array([Rx, Re])


def integrate_profile(y, z):
    return np.trapezoid(y, z) if hasattr(np, 'trapezoid') else np.trapz(y, z)

# ============================================================
# CASE I
# ============================================================

def caseI_Hxx(x, par):
    _, _, eta_b = bulk_quantities(par)
    s = eta_b - x
    return 1/(par.r_p*x) + 1/s + par.chi_pp - 2*par.chi_ps + par.chi_ss


def find_caseI_minima(log_q, par, nscan=2200):
    _, _, eta_b = bulk_quantities(par)
    xs = np.linspace(1e-7, eta_b-1e-7, nscan)
    vals = np.array([residual_x(x, eta_b, log_q, par) for x in xs])
    roots = []
    for i in range(len(xs)-1):
        fa, fb = vals[i], vals[i+1]
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa == 0:
            root = xs[i]
        elif fa*fb < 0:
            try:
                root = brentq(lambda x: residual_x(x, eta_b, log_q, par), xs[i], xs[i+1], xtol=1e-13, rtol=1e-13)
            except Exception:
                continue
        else:
            continue
        if not any(abs(root-r) < 2e-7 for r in roots):
            roots.append(root)
    minima = []
    for x in roots:
        R = residual_x(x, eta_b, log_q, par)
        Hxx = caseI_Hxx(x, par)
        if abs(R) <= ROOT_TOL and Hxx > HESS_TOL:
            minima.append({'phi_p': x, 'eta': eta_b, 'phi_v': 1-eta_b,
                           'g': local_grand_density(x, eta_b, log_q, par),
                           'Rmax': abs(R), 'type': 'CASEI'})
    minima.sort(key=lambda q: q['g'])
    return minima

# ============================================================
# CASE II interior + eta=1 KKT boundary
# ============================================================

def interior_seed_list(par, continuation=None):
    x_b, _, eta_b = bulk_quantities(par)
    theta_b = x_b/eta_b
    seeds = []
    if continuation is not None:
        seeds.append(continuation)
    for eta in [0.55, 0.75, 0.90, eta_b, 0.98]:
        for th in [theta_b, 0.01, 0.03, 0.08, 0.15, 0.30, 0.50, 0.72, 0.90]:
            seeds.append((float(np.clip(th,1e-6,1-1e-6)), float(np.clip(eta,1e-6,1-1e-6))))
    unique = []
    for s in seeds:
        if s not in unique:
            unique.append(s)
    return unique


def find_interior_minima(log_q, par, continuation=None):
    roots = []
    for th0, eta0 in interior_seed_list(par, continuation):
        u0 = np.array([logit(th0), logit(eta0)])
        try:
            sol = least_squares(transformed_EL, u0, args=(log_q, par),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1500)
        except Exception:
            continue
        x, s, eta, theta = fields_from_u(sol.x)
        if not (0 < x < eta < 1):
            continue
        Rx = residual_x(x, eta, log_q, par)
        Re = residual_eta(x, eta, par)
        Rmax = max(abs(Rx), abs(Re))
        if Rmax > ROOT_TOL:
            continue
        H = local_hessian(x, eta, par)
        if not (H['Hxx'] > HESS_TOL and H['det'] > HESS_TOL and H['lambda_min'] > HESS_TOL):
            continue
        c = {'phi_p': x, 'eta': eta, 'phi_v': 1-eta, 'theta': theta,
             'g': local_grand_density(x, eta, log_q, par), 'Rmax': Rmax,
             'lambda_min': H['lambda_min'], 'type': 'INTERIOR'}
        if not any(np.linalg.norm(np.array([c['theta'],c['eta']]) - np.array([o['theta'],o['eta']])) < MIN_DUP_TOL for o in roots):
            roots.append(c)
    roots.sort(key=lambda q: q['g'])
    return roots


def find_boundary_minima(log_q, par, nscan=2500):
    eta = 1.0
    xs = np.linspace(1e-7, 1-1e-7, nscan)
    vals = np.array([residual_x(x, eta, log_q, par) for x in xs])
    roots = []
    for i in range(len(xs)-1):
        fa, fb = vals[i], vals[i+1]
        if not (np.isfinite(fa) and np.isfinite(fb)) or fa*fb >= 0:
            continue
        try:
            root = brentq(lambda x: residual_x(x, 1.0, log_q, par), xs[i], xs[i+1], xtol=1e-13, rtol=1e-13)
        except Exception:
            continue
        if not any(abs(root-r) < 2e-7 for r in roots):
            roots.append(root)
    out = []
    for x in roots:
        Rx = residual_x(x, 1.0, log_q, par)
        Re = residual_eta(x, 1.0, par)
        Hxx = 1/(par.r_p*x) + 1/(1-x) + par.chi_pp - 2*par.chi_ps + par.chi_ss
        if abs(Rx) <= ROOT_TOL and Re <= ROOT_TOL and Hxx > HESS_TOL:
            out.append({'phi_p': x, 'eta': 1.0, 'phi_v': 0.0,
                        'g': local_grand_density(x, 1.0, log_q, par),
                        'Rmax': max(abs(Rx), max(Re,0.0)), 'R_eta': Re,
                        'type': 'BOUNDARY'})
    out.sort(key=lambda q: q['g'])
    return out


def solve_caseI_profile(z, log_q, par):
    N = len(z)
    p = np.full(N,np.nan); eta = np.full(N,np.nan); g = np.full(N,np.nan)
    nmin = np.zeros(N,int); Rmax = np.full(N,np.nan)
    for i in range(N):
        mins = find_caseI_minima(log_q[i], par)
        if not mins:
            return {'success': False, 'failed_z': z[i]}
        nmin[i] = len(mins)
        b = mins[0]
        p[i], eta[i], g[i], Rmax[i] = b['phi_p'], b['eta'], b['g'], b['Rmax']
    return {'success': True, 'phi_p': p, 'eta': eta, 'phi_v': 1-eta, 'g': g,
            'Rmax': Rmax, 'nmin': nmin, 'bistable': bool(np.any(nmin>=2)),
            'bistable_z_count': int(np.sum(nmin>=2))}


def solve_caseII_profile(z, log_q, par):
    N = len(z)
    p = np.full(N,np.nan); eta = np.full(N,np.nan); g = np.full(N,np.nan)
    state_type = np.empty(N,object); n_total=np.zeros(N,int); n_in=np.zeros(N,int); n_b=np.zeros(N,int)
    Rmax=np.full(N,np.nan); lam=np.full(N,np.nan); gap=np.full(N,np.inf)
    continuation=None
    for i in range(N-1,-1,-1):
        interior = find_interior_minima(log_q[i], par, continuation)
        boundary = find_boundary_minima(log_q[i], par)
        candidates = sorted(interior + boundary, key=lambda q:q['g'])
        n_in[i], n_b[i], n_total[i] = len(interior), len(boundary), len(candidates)
        if not candidates:
            return {'success': False, 'failed_z': z[i]}
        b = candidates[0]
        p[i], eta[i], g[i], Rmax[i], state_type[i] = b['phi_p'], b['eta'], b['g'], b['Rmax'], b['type']
        if b['type'] == 'INTERIOR':
            continuation=(b['theta'],b['eta']); lam[i]=b['lambda_min']
        else:
            continuation=(b['phi_p'],0.999)
        if len(candidates)>=2:
            gap[i]=candidates[1]['g']-candidates[0]['g']
    bmask = state_type == 'BOUNDARY'
    return {'success': True, 'phi_p':p,'eta':eta,'phi_v':1-eta,'g':g,
            'state_type':state_type,'n_total':n_total,'n_interior':n_in,'n_boundary':n_b,
            'Rmax':Rmax,'lambda_min':lam,'gap':gap,
            'bistable':bool(np.any(n_total>=2)),'bistable_z_count':int(np.sum(n_total>=2)),
            'boundary':bool(np.any(bmask)),'boundary_fraction':float(np.mean(bmask)),
            'boundary_z_count':int(np.sum(bmask)),'wall_state_type':state_type[0]}

# ============================================================
# Robust bulk global-equilibrium audit (Step 13 + 13R logic)
# ============================================================

def bulk_residual_x(x, eta, par):
    return residual_x(x, eta, bulk_log_q(par), par)


def direct_bulk_seeds(par):
    x_b, _, eta_b = bulk_quantities(par)
    theta_b=x_b/eta_b
    ths=[theta_b,0.005,0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.65,0.8,0.9,0.97]
    etas=[0.40,0.55,0.70,0.82,0.90,eta_b,0.98,0.995,1.0]
    return [(float(np.clip(t,1e-7,1-1e-7)),float(np.clip(e,1e-7,1.0))) for t in ths for e in etas]


def direct_bulk_multistart(par):
    lqb=bulk_log_q(par)
    def obj(y):
        th=float(y[0]); eta=float(y[1])
        if not (EPS < th < 1-EPS and EPS < eta <= 1):
            return 1e50
        return local_grand_density(eta*th, eta, lqb, par)
    cand=[]
    for th0,e0 in direct_bulk_seeds(par):
        try:
            sol=minimize(obj,np.array([th0,e0]),method='L-BFGS-B',
                         bounds=[(1e-8,1-1e-8),(1e-8,1.0)],
                         options={'maxiter':4000,'ftol':1e-15,'gtol':1e-12})
        except Exception:
            continue
        th=float(sol.x[0]); eta=float(sol.x[1]); x=eta*th
        if not (0<x<eta<=1):
            continue
        g=local_grand_density(x,eta,lqb,par)
        if not np.isfinite(g):
            continue
        c={'phi_p':x,'eta':eta,'phi_v':1-eta,'theta':th,'g':g}
        if not any(np.linalg.norm(np.array([th,eta])-np.array([o['theta'],o['eta']]))<MIN_DUP_TOL for o in cand):
            cand.append(c)
    cand.sort(key=lambda q:q['g'])
    return cand


def certify_bulk_direct_candidate(c, par):
    lqb=bulk_log_q(par)
    # Boundary-like candidate first
    if 1-c['eta'] <= BOUNDARY_DETECT_TOL:
        bmins=find_boundary_minima(lqb,par,nscan=6000)
        if bmins:
            # choose boundary solution closest in x among low-g candidates
            b=sorted(bmins,key=lambda q:(abs(q['phi_p']-c['phi_p']),q['g']))[0]
            return b
    # Interior refine
    eta0=min(c['eta'],1-1e-8); th0=c['phi_p']/c['eta']
    try:
        sol=least_squares(transformed_EL,np.array([logit(th0),logit(eta0)]),args=(lqb,par),
                          xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=5000)
        x,s,eta,th=fields_from_u(sol.x)
        if 0<x<eta<1:
            Rx=residual_x(x,eta,lqb,par); Re=residual_eta(x,eta,par)
            H=local_hessian(x,eta,par)
            if max(abs(Rx),abs(Re))<=ROOT_TOL and H['Hxx']>HESS_TOL and H['det']>HESS_TOL and H['lambda_min']>HESS_TOL:
                return {'phi_p':x,'eta':eta,'phi_v':1-eta,'theta':th,
                        'g':local_grand_density(x,eta,lqb,par),'Rmax':max(abs(Rx),abs(Re)),
                        'lambda_min':H['lambda_min'],'type':'INTERIOR'}
    except Exception:
        pass
    # fallback full boundary scan
    bmins=find_boundary_minima(lqb,par,nscan=6000)
    if bmins:
        return min(bmins,key=lambda q:q['g'])
    return None


def audit_bulk_global(par):
    x_b,_,eta_b=bulk_quantities(par)
    lqb=bulk_log_q(par)
    g_b=local_grand_density(x_b,eta_b,lqb,par)
    H=local_hessian(x_b,eta_b,par)
    if not (H['Hxx']>HESS_TOL and H['det']>HESS_TOL and H['lambda_min']>HESS_TOL):
        return {'bulk_class':'BU','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':np.nan,
                'bulk_competing_type':'UNSTABLE','bulk_competing_phi_p':np.nan,'bulk_competing_eta':np.nan}
    direct=direct_bulk_multistart(par)
    if not direct:
        return {'bulk_class':'BR','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':np.nan,
                'bulk_competing_type':'UNRESOLVED','bulk_competing_phi_p':np.nan,'bulk_competing_eta':np.nan}
    d=direct[0]
    dg=d['g']-g_b
    if dg >= -DIRECT_LOWER_TRIGGER:
        return {'bulk_class':'BG','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':0.0,
                'bulk_competing_type':'NONE','bulk_competing_phi_p':np.nan,'bulk_competing_eta':np.nan}
    cert=certify_bulk_direct_candidate(d,par)
    if cert is None:
        return {'bulk_class':'BR','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':dg,
                'bulk_competing_type':'UNRESOLVED','bulk_competing_phi_p':d['phi_p'],'bulk_competing_eta':d['eta']}
    dg_cert=cert['g']-g_b
    if dg_cert < -GLOBAL_G_TOL:
        return {'bulk_class':'BM','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':dg_cert,
                'bulk_competing_type':cert['type'],'bulk_competing_phi_p':cert['phi_p'],'bulk_competing_eta':cert['eta']}
    return {'bulk_class':'BG','bulk_lambda_min':H['lambda_min'],'bulk_delta_g':0.0,
            'bulk_competing_type':'NONE','bulk_competing_phi_p':np.nan,'bulk_competing_eta':np.nan}

# ============================================================
# Combined parameter-point evaluation
# ============================================================

def evaluate_point(par, wall, L=20.0, dz=0.05):
    bulk=audit_bulk_global(par)
    if bulk['bulk_class'] in ('BU','BM','BR'):
        return {**bulk,
                'surface_class':'NOT_EVALUATED',
                'final_class':bulk['bulk_class'],
                'Gamma_I':np.nan,'Gamma_II':np.nan,'DeltaDelta_Gamma':np.nan,
                'relative_percent':np.nan,'void_excess_II':np.nan,'boundary_fraction':np.nan,
                'caseI_bistable_z_count':np.nan,'caseII_bistable_z_count':np.nan,
                'profile_min_eigenvalue':np.nan,'max_delta_eta':np.nan,'maxR':np.nan}
    z=make_grid(L,dz)
    log_q=local_state_quantities(z,par,wall)['log_q']
    I=solve_caseI_profile(z,log_q,par)
    II=solve_caseII_profile(z,log_q,par)
    if not I['success'] or not II['success']:
        return {**bulk,'surface_class':'SOLVER_FAIL','final_class':'BG-SF',
                'Gamma_I':np.nan,'Gamma_II':np.nan,'DeltaDelta_Gamma':np.nan,
                'relative_percent':np.nan,'void_excess_II':np.nan,'boundary_fraction':np.nan,
                'caseI_bistable_z_count':np.nan,'caseII_bistable_z_count':np.nan,
                'profile_min_eigenvalue':np.nan,'max_delta_eta':np.nan,'maxR':np.nan}
    Gamma_I=integrate_profile(I['phi_p']-par.phi_p_b,z)
    Gamma_II=integrate_profile(II['phi_p']-par.phi_p_b,z)
    DD=Gamma_II-Gamma_I
    rel=100*DD/Gamma_I if abs(Gamma_I)>1e-14 else np.nan
    void=integrate_profile(II['phi_v']-par.phi_v_b,z)
    _,_,eta_b=bulk_quantities(par)
    maxdeta=float(np.max(np.abs(II['eta']-eta_b)))
    finite=II['lambda_min'][np.isfinite(II['lambda_min'])]
    mineig=float(np.min(finite)) if len(finite) else np.nan
    maxR=float(np.nanmax(II['Rmax']))
    if I['bistable']:
        surface='BI'; final='BG-BI'
    elif II['bistable']:
        surface='BII'; final='BG-BII'
    else:
        sign='+' if DD>EFFECT_ABS_TOL else '-' if DD<-EFFECT_ABS_TOL else '0'
        if II['boundary']:
            surface='K'+sign; final='BG-K'+sign
        else:
            surface='I'+sign; final='BG-I'+sign
    return {**bulk,'surface_class':surface,'final_class':final,
            'Gamma_I':Gamma_I,'Gamma_II':Gamma_II,'DeltaDelta_Gamma':DD,
            'relative_percent':rel,'void_excess_II':void,'boundary_fraction':II['boundary_fraction'],
            'caseI_bistable_z_count':I['bistable_z_count'],'caseII_bistable_z_count':II['bistable_z_count'],
            'profile_min_eigenvalue':mineig,'max_delta_eta':maxdeta,'maxR':maxR,
            'phi_p_wall':II['phi_p'][0],'eta_wall':II['eta'][0],'phi_v_wall':II['phi_v'][0]}

# ============================================================
# Plotting helpers
# ============================================================

CLASS_ORDER = ['BG-I+','BG-I0','BG-I-','BG-K+','BG-K0','BG-K-','BG-BII','BG-BI','BM','BU','BR','BG-SF']
CLASS_CODE = {c:i for i,c in enumerate(CLASS_ORDER)}


def save_classification_map(df, kappas, chis, outpath):
    matrix=np.full((len(chis),len(kappas)),np.nan)
    labels=np.empty((len(chis),len(kappas)),object)
    for i,chi in enumerate(chis):
        for j,k in enumerate(kappas):
            row=df[(np.isclose(df.chi_ps,chi)) & (np.isclose(df.kappa,k))].iloc[0]
            labels[i,j]=row.final_class
            matrix[i,j]=CLASS_CODE.get(row.final_class,len(CLASS_ORDER))
    fig,ax=plt.subplots(figsize=(11,8))
    im=ax.imshow(matrix,aspect='auto',origin='lower')
    ax.set_xticks(range(len(kappas)),[str(k) for k in kappas])
    ax.set_yticks(range(len(chis)),[str(c) for c in chis])
    ax.set_xlabel('kappa')
    ax.set_ylabel('chi_ps')
    ax.set_title('equilibrium-regime calculation final admissible regime map')
    for i in range(len(chis)):
        for j in range(len(kappas)):
            ax.text(j,i,labels[i,j],ha='center',va='center',fontsize=8)
    cbar=fig.colorbar(im,ax=ax)
    ticks=np.arange(len(CLASS_ORDER))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(CLASS_ORDER)
    fig.tight_layout()
    fig.savefig(outpath,dpi=300,bbox_inches='tight')
    plt.close(fig)


def save_numeric_map(df, kappas, chis, column, title, outpath):
    matrix=np.full((len(chis),len(kappas)),np.nan)
    for i,chi in enumerate(chis):
        for j,k in enumerate(kappas):
            row=df[(np.isclose(df.chi_ps,chi)) & (np.isclose(df.kappa,k))].iloc[0]
            if str(row.final_class).startswith('BG-I') or str(row.final_class).startswith('BG-K'):
                matrix[i,j]=row[column]
    fig,ax=plt.subplots(figsize=(10,7))
    im=ax.imshow(matrix,aspect='auto',origin='lower')
    ax.set_xticks(range(len(kappas)),[str(k) for k in kappas])
    ax.set_yticks(range(len(chis)),[str(c) for c in chis])
    ax.set_xlabel('kappa')
    ax.set_ylabel('chi_ps')
    ax.set_title(title)
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    fig.savefig(outpath,dpi=300,bbox_inches='tight')
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    OUTPUT_DIR=Path('01_equilibrium_output')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    base=ModelParams()
    wall=WallParams()

    kappas=[20.0,10.0,5.0,2.0,1.0,0.5,0.2,0.1]
    chi_values=[-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0]

    records=[]
    total=len(kappas)*len(chi_values); n=0
    print('='*140)
    print('EQUILIBRIUM REGIME MAP: FINAL ADMISSIBLE REGIME MAP')
    print('='*140)
    for chi in chi_values:
        for kappa in kappas:
            n+=1
            par=replace(base,chi_ps=chi,kappa=kappa)
            out=evaluate_point(par,wall,L=20.0,dz=0.05)
            out.update({'chi_ps':chi,'kappa':kappa})
            records.append(out)
            print(f'[{n:3d}/{total}] chi_ps={chi:5.2f} kappa={kappa:5.2f}  -> {out["final_class"]}')

    df=pd.DataFrame(records)
    cols=['chi_ps','kappa','bulk_class','surface_class','final_class',
          'Gamma_I','Gamma_II','DeltaDelta_Gamma','relative_percent','void_excess_II',
          'boundary_fraction','caseI_bistable_z_count','caseII_bistable_z_count',
          'bulk_lambda_min','bulk_delta_g','bulk_competing_type','bulk_competing_phi_p','bulk_competing_eta',
          'profile_min_eigenvalue','max_delta_eta','maxR','phi_p_wall','eta_wall','phi_v_wall']
    for c in cols:
        if c not in df.columns:
            df[c]=np.nan
    df=df[cols]
    df.to_csv(OUTPUT_DIR/'equilibrium_all_states.csv',index=False)

    admissible=df[df.final_class.str.startswith('BG-')].copy()
    admissible=admissible[~admissible.final_class.isin(['BG-SF'])]
    admissible.to_csv(OUTPUT_DIR/'equilibrium_bulk_global_states.csv',index=False)

    matrix=[]
    for chi in chi_values:
        row={'chi_ps':chi}
        for k in kappas:
            row[str(k)]=df[(np.isclose(df.chi_ps,chi))&(np.isclose(df.kappa,k))].iloc[0].final_class
        matrix.append(row)
    pd.DataFrame(matrix).to_csv(OUTPUT_DIR/'equilibrium_classification_matrix.csv',index=False)

    save_classification_map(df,kappas,chi_values,OUTPUT_DIR/'equilibrium_regime_map.png')
    save_numeric_map(df,kappas,chi_values,'relative_percent',
                     'Compressibility modulation of adsorption (%)',
                     OUTPUT_DIR/'adsorption_response_map.png')
    save_numeric_map(df,kappas,chi_values,'void_excess_II',
                     'Integrated void excess in admissible single-phase states',
                     OUTPUT_DIR/'free_volume_response_map.png')

    counts=df.final_class.value_counts().sort_index()
    with open(OUTPUT_DIR/'equilibrium_summary.txt','w') as f:
        f.write('EQUILIBRIUM REGIME MAP FINAL ADMISSIBLE REGIME MAP\n')
        f.write('='*70+'\n\n')
        f.write('Final-class counts:\n')
        for key,val in counts.items():
            f.write(f'{key:12s} {val:4d}\n')
        f.write('\nAdmissible bulk-global classes are BG-* only.\n')
        f.write('BM/BU/BR are excluded from ordinary adsorption interpretation.\n')
        f.write('BG-BI/BG-BII are bulk-admissible but locally bistable; treat separately from the regular single-phase response.\n')
        f.write('BG-K* contains eta=1 KKT boundary portions; use as limiting/saturation regimes.\n')
        f.write('BG-I* is the clean interior single-phase regime.\n')

    print('\n'+'='*140)
    print('FINAL CLASS COUNTS')
    print('='*140)
    print(counts.to_string())
    print('\nSaved files:')
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(' ',p)
