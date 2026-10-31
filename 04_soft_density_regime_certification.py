
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, replace
from scipy.optimize import (
    brentq,
    least_squares,
    minimize,
)


# ============================================================
# SOFT-DENSITY REGIME CERTIFICATION
#
# LOW-KAPPA / HIGH-CHI BOUNDARY CERTIFICATION
#
# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
#
# Target the four Step-16 edge points:
#
#   (kappa, chi_ps)
#       (1.0, 0.6)
#       (0.5, 0.6)
#       (0.2, 0.6)
#       (0.1, 0.6)
#
# and determine, as a function of z:
#
#   1. whether the equilibrium local state is an
#      interior stationary minimum,
#
#   2. whether eta = 1 is the constrained KKT minimum,
#
#   3. whether multiple locally stable minima coexist,
#
#   4. whether Step-16 failure at (0.1,0.6,z=0)
#      was numerical or physical,
#
#   5. where nonlinear density relaxation becomes large.
#
#
# The code uses three independent checks:
#
#   A. direct constrained minimization in (theta, eta)
#   B. exact interior EL refinement
#   C. exact eta=1 boundary root/KKT certification
#
#
# OUTPUT
# ------------------------------------------------------------
#
# 04_soft_density_regime_output/
#
#   soft_regime_pointwise_certification.csv
#   soft_regime_state_summary.csv
#   soft_regime_certification_summary.txt
#
#   step17_k1_chi06_profiles.png
#   step17_k05_chi06_profiles.png
#   step17_k02_chi06_profiles.png
#   step17_k01_chi06_profiles.png
#
#   step17_k1_chi06_free_energy_gap.png
#   ...
#
#   soft_regime_k01_chi06_contact_landscape.csv
#   soft_regime_k01_chi06_contact_landscape.png
#
#
# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# This step does NOT replace Step 14 bulk admissibility.
# It resolves the interfacial/local response boundary inside
# the already-selected BG region.
#
# ============================================================


# ============================================================
# 1. PATH SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STEP14_FILE = os.path.join(
    SCRIPT_DIR,
    "01_equilibrium_output",
    "equilibrium_all_states.csv"
)

STEP16_FILE = os.path.join(
    SCRIPT_DIR,
    "03_global_linear_response_output",
    "global_linear_response_regular_states.csv"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "04_soft_density_regime_output"
)


# ============================================================
# 2. MODEL PARAMETERS
# ============================================================

@dataclass
class ModelParams:

    r_p: float

    phi_p_b: float
    phi_v_b: float

    chi_pp: float
    chi_ps: float
    chi_ss: float

    kappa: float

    delta_u_int: float


@dataclass
class WallParams:

    eps_N: float
    eps_D: float
    lambda_w: float


# ============================================================
# 3. NUMERICAL SETTINGS
# ============================================================

ROOT_TOL = 1.0e-9
CERT_TOL = 1.0e-8
HESS_TOL = 1.0e-9

DUP_TOL = 1.0e-7
GAP_TOL = 1.0e-8

ETA_BOUNDARY_TOL = 2.0e-6

MAX_NFEV = 4000

DIRECT_THETA_EPS = 1.0e-8
DIRECT_ETA_EPS = 1.0e-6

BOUNDARY_SCAN_POINTS = 5000

# Near-wall fine grid, then coarser tail.
Z_FINE_MAX = 2.0
DZ_FINE = 0.025

Z_TAIL_MAX = 8.0
DZ_TAIL = 0.10

# Contact landscape only for the most problematic point.
LANDSCAPE_THETA_POINTS = 240
LANDSCAPE_ETA_POINTS = 180


# ============================================================
# 4. TARGET POINTS
# ============================================================

TARGETS = [

    {
        "label": "kappa=1.0, chi_ps=0.6",
        "slug": "k1_chi06",
        "kappa": 1.0,
        "chi_ps": 0.6,
    },

    {
        "label": "kappa=0.5, chi_ps=0.6",
        "slug": "k05_chi06",
        "kappa": 0.5,
        "chi_ps": 0.6,
    },

    {
        "label": "kappa=0.2, chi_ps=0.6",
        "slug": "k02_chi06",
        "kappa": 0.2,
        "chi_ps": 0.6,
    },

    {
        "label": "kappa=0.1, chi_ps=0.6",
        "slug": "k01_chi06",
        "kappa": 0.1,
        "chi_ps": 0.6,
    },
]


# ============================================================
# 5. GRID
# ============================================================

def make_z_grid():

    z1 = np.arange(
        0.0,
        Z_FINE_MAX + 0.5 * DZ_FINE,
        DZ_FINE
    )

    z2 = np.arange(
        Z_FINE_MAX + DZ_TAIL,
        Z_TAIL_MAX + 0.5 * DZ_TAIL,
        DZ_TAIL
    )

    return np.concatenate(
        [
            z1,
            z2
        ]
    )


# ============================================================
# 6. BASIC MODEL FUNCTIONS
# ============================================================

def bulk_quantities(par):

    x_b = par.phi_p_b
    eta_b = 1.0 - par.phi_v_b
    s_b = eta_b - x_b

    if not (
        0.0 < x_b < eta_b < 1.0
    ):

        raise ValueError(
            "Require 0 < phi_p_b < eta_b < 1."
        )

    return (
        x_b,
        s_b,
        eta_b
    )


def state_energies(
    z,
    par,
    wall
):

    decay = np.exp(
        -z / wall.lambda_w
    )

    E_N = (
        -wall.eps_N
        * decay
    )

    E_D = (
        par.delta_u_int
        -
        wall.eps_D
        * decay
    )

    return (
        E_N,
        E_D
    )


def log_q(
    z,
    par,
    wall
):

    E_N, E_D = state_energies(
        z,
        par,
        wall
    )

    return np.logaddexp(
        -E_N,
        -E_D
    )


def log_q_bulk(par):

    return np.logaddexp(
        0.0,
        -par.delta_u_int
    )


def interaction_I(
    x,
    s,
    par
):

    return (

        par.chi_pp * x

        +

        par.chi_ps
        * (
            s - x
        )

        -

        par.chi_ss * s
    )


# ============================================================
# 7. BULK-SUBTRACTED EL RESIDUALS
# ============================================================

def residual_x(
    x,
    eta,
    z,
    par,
    wall
):

    if not (
        0.0 < x < eta <= 1.0
    ):

        return np.nan

    s = eta - x

    if s <= 0.0:

        return np.nan

    x_b, s_b, eta_b = bulk_quantities(
        par
    )

    return (

        (
            1.0
            / par.r_p
        )
        * np.log(

            (x / eta)

            /

            (x_b / eta_b)
        )

        -

        np.log(

            (s / eta)

            /

            (s_b / eta_b)
        )

        +

        interaction_I(
            x,
            s,
            par
        )

        -

        interaction_I(
            x_b,
            s_b,
            par
        )

        -

        (
            1.0
            / par.r_p
        )
        * (

            log_q(
                z,
                par,
                wall
            )

            -

            log_q_bulk(
                par
            )
        )
    )


def residual_eta(
    x,
    eta,
    par
):

    if not (
        0.0 < x < eta <= 1.0
    ):

        return np.nan

    s = eta - x

    if s <= 0.0:

        return np.nan

    x_b, s_b, eta_b = bulk_quantities(
        par
    )

    return (

        np.log(

            (s / eta)

            /

            (s_b / eta_b)
        )

        +

        (
            1.0
            - 1.0 / par.r_p
        )
        * (

            x / eta

            -

            x_b / eta_b
        )

        +

        par.kappa
        * (
            eta - eta_b
        )

        +

        par.chi_ps
        * (
            x - x_b
        )

        +

        par.chi_ss
        * (
            s - s_b
        )
    )


# ============================================================
# 8. HESSIAN
# ============================================================

def local_hessian(
    x,
    eta,
    par
):

    s = eta - x
    r = par.r_p

    Hxx = (

        1.0
        / (
            r * x
        )

        +

        1.0 / s

        +

        par.chi_pp

        -

        2.0
        * par.chi_ps

        +

        par.chi_ss
    )

    Hxe = (

        (
            1.0
            - 1.0 / r
        )
        / eta

        -

        1.0 / s

        +

        par.chi_ps

        -

        par.chi_ss
    )

    Hee = (

        1.0 / s

        -

        1.0 / eta

        -

        (
            1.0
            - 1.0 / r
        )
        * x
        / eta**2

        +

        par.kappa

        +

        par.chi_ss
    )

    detH = (
        Hxx * Hee
        -
        Hxe**2
    )

    eig = np.linalg.eigvalsh(
        np.array(
            [
                [
                    Hxx,
                    Hxe
                ],
                [
                    Hxe,
                    Hee
                ]
            ]
        )
    )

    return {
        "Hxx": Hxx,
        "Hxeta": Hxe,
        "Hetaeta": Hee,
        "det": detH,
        "lambda_min": eig[0],
        "lambda_max": eig[1],
    }


# ============================================================
# 9. CHEMICAL POTENTIALS AND GRAND DENSITY
# ============================================================

def bulk_chemical_potentials(par):

    x_b, s_b, eta_b = bulk_quantities(
        par
    )

    r = par.r_p
    lqb = log_q_bulk(
        par
    )

    mu_s = (

        np.log(
            s_b / eta_b
        )

        +

        (
            1.0
            - 1.0 / r
        )
        * x_b / eta_b

        +

        par.chi_ps
        * x_b

        +

        par.chi_ss
        * s_b
    )

    mu_p = (

        np.log(
            x_b / eta_b
        )

        + 1.0

        -

        r
        * (
            np.log(
                s_b / eta_b
            )
            + 1.0
        )

        +

        r
        * (

            par.chi_pp
            * x_b

            +

            par.chi_ps
            * (
                s_b - x_b
            )

            -

            par.chi_ss
            * s_b
        )

        -

        lqb

        +

        r * mu_s
    )

    return (
        mu_p,
        mu_s
    )


def grand_density(
    x,
    eta,
    z,
    par,
    wall
):

    if not (
        0.0 < x < eta <= 1.0
    ):

        return np.inf

    s = eta - x

    if s <= 0.0:

        return np.inf

    _, _, eta_b = bulk_quantities(
        par
    )

    mu_p, mu_s = bulk_chemical_potentials(
        par
    )

    r = par.r_p

    g_mix = (

        (
            x / r
        )
        * np.log(
            x / eta
        )

        +

        s
        * np.log(
            s / eta
        )
    )

    g_eos = (

        0.5
        * par.kappa
        * (
            eta - eta_b
        )**2
    )

    g_int = (

        0.5
        * par.chi_pp
        * x**2

        +

        par.chi_ps
        * x * s

        +

        0.5
        * par.chi_ss
        * s**2
    )

    g_state = (

        -(
            x / r
        )
        * log_q(
            z,
            par,
            wall
        )
    )

    g_mu = (

        -(
            x / r
        )
        * mu_p

        -

        s * mu_s
    )

    return (

        g_mix
        + g_eos
        + g_int
        + g_state
        + g_mu
    )


# ============================================================
# 10. DIRECT PARAMETERIZATION
#
# theta = x/eta
# eta in (0,1]
# ============================================================

def direct_objective(
    v,
    z,
    par,
    wall
):

    theta = v[0]
    eta = v[1]

    if not (
        DIRECT_THETA_EPS
        <= theta
        <= 1.0 - DIRECT_THETA_EPS
    ):

        return 1.0e8

    if not (
        DIRECT_ETA_EPS
        <= eta
        <= 1.0
    ):

        return 1.0e8

    x = (
        theta * eta
    )

    return grand_density(
        x,
        eta,
        z,
        par,
        wall
    )


def direct_multistart(
    z,
    par,
    wall,
    previous_candidates=None
):

    x_b, _, eta_b = bulk_quantities(
        par
    )

    theta_b = (
        x_b / eta_b
    )

    seeds = [

        (
            theta_b,
            eta_b
        ),

        (
            min(
                0.95,
                max(
                    0.02,
                    2.0 * theta_b
                )
            ),
            eta_b
        ),

        (
            0.05,
            0.99
        ),

        (
            0.10,
            0.99
        ),

        (
            0.25,
            0.99
        ),

        (
            0.50,
            0.99
        ),

        (
            0.80,
            0.99
        ),

        (
            0.95,
            0.999999
        ),
    ]

    if previous_candidates is not None:

        for c in previous_candidates:

            if (
                np.isfinite(
                    c.get(
                        "x",
                        np.nan
                    )
                )
                and
                np.isfinite(
                    c.get(
                        "eta",
                        np.nan
                    )
                )
            ):

                eta0 = c["eta"]
                x0 = c["x"]

                if (
                    0.0
                    < x0
                    < eta0
                    <= 1.0
                ):

                    seeds.insert(
                        0,
                        (
                            x0 / eta0,
                            eta0
                        )
                    )

    bounds = [

        (
            DIRECT_THETA_EPS,
            1.0
            - DIRECT_THETA_EPS
        ),

        (
            DIRECT_ETA_EPS,
            1.0
        )
    ]

    minima = []

    for seed in seeds:

        try:

            sol = minimize(

                direct_objective,

                x0=np.array(
                    seed,
                    dtype=float
                ),

                args=(
                    z,
                    par,
                    wall
                ),

                method="L-BFGS-B",

                bounds=bounds,

                options={
                    "ftol": 1.0e-13,
                    "gtol": 1.0e-10,
                    "maxiter": 2000,
                }
            )

        except Exception:

            continue

        theta = float(
            sol.x[0]
        )

        eta = float(
            sol.x[1]
        )

        x = theta * eta

        g = grand_density(
            x,
            eta,
            z,
            par,
            wall
        )

        if not np.isfinite(
            g
        ):

            continue

        candidate = {
            "x": x,
            "eta": eta,
            "theta": theta,
            "g": g,
            "success": bool(
                sol.success
            ),
        }

        duplicate = False

        for old in minima:

            d = np.linalg.norm(
                np.array(
                    [
                        x,
                        eta
                    ]
                )
                -
                np.array(
                    [
                        old["x"],
                        old["eta"]
                    ]
                )
            )

            if d < 5.0e-6:

                duplicate = True

                if g < old["g"]:

                    old.update(
                        candidate
                    )

                break

        if not duplicate:

            minima.append(
                candidate
            )

    minima.sort(
        key=lambda q:
            q["g"]
    )

    return minima


# ============================================================
# 11. INTERIOR EXACT REFINEMENT
# ============================================================

def sigmoid(a):

    if a >= 0.0:

        e = np.exp(
            -a
        )

        return (
            1.0
            / (
                1.0 + e
            )
        )

    e = np.exp(
        a
    )

    return (
        e
        / (
            1.0 + e
        )
    )


def logit(p):

    p = np.clip(
        p,
        1.0e-12,
        1.0 - 1.0e-12
    )

    return np.log(
        p
        /
        (
            1.0 - p
        )
    )


def unpack_interior(u):

    theta = sigmoid(
        u[0]
    )

    eta = sigmoid(
        u[1]
    )

    x = (
        eta * theta
    )

    return (
        x,
        eta
    )


def interior_residual_u(
    u,
    z,
    par,
    wall
):

    x, eta = unpack_interior(
        u
    )

    Rx = residual_x(
        x,
        eta,
        z,
        par,
        wall
    )

    Re = residual_eta(
        x,
        eta,
        par
    )

    if not (
        np.isfinite(Rx)
        and
        np.isfinite(Re)
    ):

        return np.array(
            [
                1.0e6,
                1.0e6
            ]
        )

    return np.array(
        [
            Rx,
            Re
        ]
    )


def refine_interior_candidate(
    candidate,
    z,
    par,
    wall
):

    x0 = candidate["x"]
    eta0 = candidate["eta"]

    if not (
        0.0
        < x0
        < eta0
        < 1.0
    ):

        return None

    if eta0 > (
        1.0
        - ETA_BOUNDARY_TOL
    ):

        return None

    theta0 = (
        x0 / eta0
    )

    u0 = np.array(
        [
            logit(
                theta0
            ),
            logit(
                eta0
            )
        ]
    )

    try:

        sol = least_squares(

            interior_residual_u,

            u0,

            args=(
                z,
                par,
                wall
            ),

            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,

            max_nfev=
                MAX_NFEV
        )

    except Exception:

        return None

    x, eta = unpack_interior(
        sol.x
    )

    if not (
        0.0
        < x
        < eta
        < 1.0
    ):

        return None

    Rx = residual_x(
        x,
        eta,
        z,
        par,
        wall
    )

    Re = residual_eta(
        x,
        eta,
        par
    )

    Rmax = max(
        abs(Rx),
        abs(Re)
    )

    if Rmax > CERT_TOL:

        return None

    H = local_hessian(
        x,
        eta,
        par
    )

    if not (
        H["Hxx"]
        > HESS_TOL

        and

        H["det"]
        > HESS_TOL

        and

        H["lambda_min"]
        > HESS_TOL
    ):

        return None

    return {
        "type": "INTERIOR",
        "x": x,
        "eta": eta,
        "phi_v": 1.0 - eta,
        "g": grand_density(
            x,
            eta,
            z,
            par,
            wall
        ),
        "Rmax": Rmax,
        "Reta": Re,
        "Hxx": H["Hxx"],
        "detH": H["det"],
        "lambda_min": H["lambda_min"],
    }


# ============================================================
# 12. ETA=1 BOUNDARY ROOTS + KKT
#
# Upper constraint eta <= 1:
#
#   R_x = 0
#   R_eta <= 0
#   H_xx > 0
#
# ============================================================

def find_boundary_roots(
    z,
    par,
    wall
):

    eta = 1.0

    x_grid = np.linspace(
        1.0e-8,
        1.0 - 1.0e-8,
        BOUNDARY_SCAN_POINTS
    )

    vals = np.array(
        [
            residual_x(
                x,
                eta,
                z,
                par,
                wall
            )
            for x in x_grid
        ]
    )

    roots = []

    for i in range(
        len(x_grid)
        - 1
    ):

        fa = vals[i]
        fb = vals[i + 1]

        if not (
            np.isfinite(fa)
            and
            np.isfinite(fb)
        ):

            continue

        root = None

        if abs(fa) < ROOT_TOL:

            root = x_grid[i]

        elif fa * fb < 0.0:

            try:

                root = brentq(

                    lambda xx:
                        residual_x(
                            xx,
                            1.0,
                            z,
                            par,
                            wall
                        ),

                    x_grid[i],
                    x_grid[i + 1],

                    xtol=1.0e-14,
                    rtol=1.0e-14
                )

            except Exception:

                continue

        if root is None:

            continue

        if any(
            abs(
                root - old
            )
            < DUP_TOL
            for old in roots
        ):

            continue

        roots.append(
            root
        )

    candidates = []

    for x in roots:

        Re = residual_eta(
            x,
            1.0,
            par
        )

        H = local_hessian(
            x,
            1.0,
            par
        )

        kkt = (
            Re
            <= CERT_TOL

            and

            H["Hxx"]
            > HESS_TOL
        )

        candidates.append(
            {
                "type":
                    "BOUNDARY",

                "x":
                    x,

                "eta":
                    1.0,

                "phi_v":
                    0.0,

                "g":
                    grand_density(
                        x,
                        1.0,
                        z,
                        par,
                        wall
                    ),

                "Rmax":
                    abs(
                        residual_x(
                            x,
                            1.0,
                            z,
                            par,
                            wall
                        )
                    ),

                "Reta":
                    Re,

                "Hxx":
                    H["Hxx"],

                "detH":
                    H["det"],

                "lambda_min":
                    H["lambda_min"],

                "KKT":
                    kkt,
            }
        )

    candidates.sort(
        key=lambda q:
            q["g"]
    )

    return candidates


# ============================================================
# 13. CERTIFY ALL DIRECT MINIMA
# ============================================================

def certify_local_states(
    z,
    par,
    wall,
    direct_minima
):

    certified = []

    # --------------------------------------------------------
    # A. Refine all distinct direct interior minima
    # --------------------------------------------------------

    for d in direct_minima:

        c = refine_interior_candidate(
            d,
            z,
            par,
            wall
        )

        if c is None:

            continue

        duplicate = False

        for old in certified:

            dist = np.linalg.norm(
                np.array(
                    [
                        c["x"],
                        c["eta"]
                    ]
                )
                -
                np.array(
                    [
                        old["x"],
                        old["eta"]
                    ]
                )
            )

            if dist < DUP_TOL:

                duplicate = True
                break

        if not duplicate:

            certified.append(
                c
            )

    # --------------------------------------------------------
    # B. Add ALL exact stable KKT boundary roots
    # --------------------------------------------------------

    boundary = find_boundary_roots(
        z,
        par,
        wall
    )

    for c in boundary:

        if not c["KKT"]:

            continue

        duplicate = False

        for old in certified:

            dist = np.linalg.norm(
                np.array(
                    [
                        c["x"],
                        c["eta"]
                    ]
                )
                -
                np.array(
                    [
                        old["x"],
                        old["eta"]
                    ]
                )
            )

            if dist < DUP_TOL:

                duplicate = True
                break

        if not duplicate:

            certified.append(
                c
            )

    certified.sort(
        key=lambda q:
            q["g"]
    )

    return certified


# ============================================================
# 14. POINT CLASSIFICATION
# ============================================================

def classify_point(
    certified
):

    if len(certified) == 0:

        return (
            "UNRESOLVED",
            None,
            np.nan
        )

    global_state = certified[0]

    if len(certified) == 1:

        if global_state["type"] == "INTERIOR":

            return (
                "INTERIOR",
                global_state,
                np.nan
            )

        return (
            "KKT",
            global_state,
            np.nan
        )

    gap = (
        certified[1]["g"]
        -
        certified[0]["g"]
    )

    if abs(gap) <= GAP_TOL:

        return (
            "COEXIST",
            global_state,
            gap
        )

    # More than one stable local state exists even if one is
    # clearly global.
    if global_state["type"] == "INTERIOR":

        return (
            "MULTI_INTERIOR",
            global_state,
            gap
        )

    return (
        "MULTI_KKT",
        global_state,
        gap
    )


# ============================================================
# 15. CASE-I REFERENCE ROOT
# ============================================================

def solve_caseI_reference(
    z,
    par,
    wall
):

    _, _, eta_b = bulk_quantities(
        par
    )

    x_grid = np.linspace(
        1.0e-8,
        eta_b - 1.0e-8,
        4000
    )

    vals = np.array(
        [
            residual_x(
                x,
                eta_b,
                z,
                par,
                wall
            )
            for x in x_grid
        ]
    )

    roots = []

    for i in range(
        len(x_grid) - 1
    ):

        fa = vals[i]
        fb = vals[i + 1]

        if not (
            np.isfinite(fa)
            and
            np.isfinite(fb)
        ):

            continue

        if fa * fb < 0.0:

            try:

                root = brentq(

                    lambda xx:
                        residual_x(
                            xx,
                            eta_b,
                            z,
                            par,
                            wall
                        ),

                    x_grid[i],
                    x_grid[i + 1],

                    xtol=1.0e-14,
                    rtol=1.0e-14
                )

            except Exception:

                continue

            if any(
                abs(
                    root - old
                )
                < DUP_TOL
                for old in roots
            ):

                continue

            H = local_hessian(
                root,
                eta_b,
                par
            )

            if H["Hxx"] > HESS_TOL:

                roots.append(
                    root
                )

    if len(roots) == 0:

        return np.nan

    gs = np.array(
        [
            grand_density(
                x,
                eta_b,
                z,
                par,
                wall
            )
            for x in roots
        ]
    )

    return roots[
        int(
            np.argmin(
                gs
            )
        )
    ]


# ============================================================
# 16. ONE TARGET CASE
# ============================================================

def run_target(
    target,
    base_par,
    wall,
    z_grid
):

    par = replace(

        base_par,

        kappa=target[
            "kappa"
        ],

        chi_ps=target[
            "chi_ps"
        ]
    )

    print()
    print("=" * 120)

    print(
        target[
            "label"
        ]
    )

    print("=" * 120)

    rows = []

    previous = None

    for iz, z in enumerate(
        z_grid
    ):

        direct = direct_multistart(

            z,
            par,
            wall,

            previous_candidates=
                previous
        )

        certified = certify_local_states(
            z,
            par,
            wall,
            direct
        )

        classification, global_state, gap = (
            classify_point(
                certified
            )
        )

        xI = solve_caseI_reference(
            z,
            par,
            wall
        )

        _, _, eta_b = bulk_quantities(
            par
        )

        if global_state is None:

            row = {

                "label":
                    target["label"],

                "slug":
                    target["slug"],

                "kappa":
                    par.kappa,

                "chi_ps":
                    par.chi_ps,

                "z":
                    z,

                "classification":
                    classification,

                "n_direct_minima":
                    len(direct),

                "n_certified_states":
                    0,

                "global_type":
                    "",

                "phi_p_CaseI":
                    xI,

                "eta_CaseI":
                    eta_b,

                "phi_p":
                    np.nan,

                "eta":
                    np.nan,

                "phi_v":
                    np.nan,

                "delta_phi_p":
                    np.nan,

                "delta_eta":
                    np.nan,

                "g_global":
                    np.nan,

                "gap_to_second":
                    np.nan,

                "Reta":
                    np.nan,

                "Hxx":
                    np.nan,

                "detH":
                    np.nan,

                "lambda_min":
                    np.nan,
            }

        else:

            row = {

                "label":
                    target["label"],

                "slug":
                    target["slug"],

                "kappa":
                    par.kappa,

                "chi_ps":
                    par.chi_ps,

                "z":
                    z,

                "classification":
                    classification,

                "n_direct_minima":
                    len(direct),

                "n_certified_states":
                    len(certified),

                "global_type":
                    global_state["type"],

                "phi_p_CaseI":
                    xI,

                "eta_CaseI":
                    eta_b,

                "phi_p":
                    global_state["x"],

                "eta":
                    global_state["eta"],

                "phi_v":
                    global_state["phi_v"],

                "delta_phi_p":
                    (
                        global_state["x"]
                        - xI
                    )
                    if np.isfinite(xI)
                    else np.nan,

                "delta_eta":
                    global_state["eta"]
                    - eta_b,

                "g_global":
                    global_state["g"],

                "gap_to_second":
                    gap,

                "Reta":
                    global_state["Reta"],

                "Hxx":
                    global_state["Hxx"],

                "detH":
                    global_state["detH"],

                "lambda_min":
                    global_state[
                        "lambda_min"
                    ],
            }

        if len(certified) >= 2:

            row[
                "second_type"
            ] = certified[1][
                "type"
            ]

            row[
                "second_phi_p"
            ] = certified[1][
                "x"
            ]

            row[
                "second_eta"
            ] = certified[1][
                "eta"
            ]

            row[
                "second_g"
            ] = certified[1][
                "g"
            ]

        else:

            row[
                "second_type"
            ] = ""

            row[
                "second_phi_p"
            ] = np.nan

            row[
                "second_eta"
            ] = np.nan

            row[
                "second_g"
            ] = np.nan

        rows.append(
            row
        )

        previous = certified[:3]

        print(
            f"z={z:6.3f}  "
            f"{classification:14s}  "
            f"ncert={len(certified):2d}  "
            f"x="
            f"{row['phi_p'] if np.isfinite(row['phi_p']) else np.nan: .7f}  "
            f"eta="
            f"{row['eta'] if np.isfinite(row['eta']) else np.nan: .7f}"
        )

    df = pd.DataFrame(
        rows
    )

    return (
        par,
        df
    )


# ============================================================
# 17. SUMMARY FOR ONE CASE
# ============================================================

def summarize_case(
    target,
    df,
    par
):

    valid = df[
        np.isfinite(
            df["eta"]
        )
    ].copy()

    class_counts = (
        df[
            "classification"
        ]
        .value_counts()
        .to_dict()
    )

    kkt_mask = (
        df[
            "global_type"
        ]
        == "BOUNDARY"
    )

    multi_mask = (
        df[
            "n_certified_states"
        ]
        >= 2
    )

    unresolved_mask = (
        df[
            "classification"
        ]
        == "UNRESOLVED"
    )

    if np.any(
        kkt_mask
    ):

        z_kkt_min = float(
            df.loc[
                kkt_mask,
                "z"
            ].min()
        )

        z_kkt_max = float(
            df.loc[
                kkt_mask,
                "z"
            ].max()
        )

    else:

        z_kkt_min = np.nan
        z_kkt_max = np.nan

    if np.any(
        multi_mask
    ):

        z_multi_min = float(
            df.loc[
                multi_mask,
                "z"
            ].min()
        )

        z_multi_max = float(
            df.loc[
                multi_mask,
                "z"
            ].max()
        )

    else:

        z_multi_min = np.nan
        z_multi_max = np.nan

    if len(valid) > 0:

        max_abs_deta = float(
            np.nanmax(
                np.abs(
                    valid[
                        "delta_eta"
                    ]
                )
            )
        )

        min_lambda = float(
            np.nanmin(
                valid[
                    "lambda_min"
                ]
            )
        )

        max_phi_p = float(
            np.nanmax(
                valid[
                    "phi_p"
                ]
            )
        )

        min_phi_v = float(
            np.nanmin(
                valid[
                    "phi_v"
                ]
            )
        )

    else:

        max_abs_deta = np.nan
        min_lambda = np.nan
        max_phi_p = np.nan
        min_phi_v = np.nan

    contact = df.iloc[
        0
    ]

    return {

        "label":
            target["label"],

        "slug":
            target["slug"],

        "kappa":
            par.kappa,

        "chi_ps":
            par.chi_ps,

        "n_z":
            len(df),

        "n_unresolved":
            int(
                np.sum(
                    unresolved_mask
                )
            ),

        "n_kkt_global":
            int(
                np.sum(
                    kkt_mask
                )
            ),

        "n_multistable":
            int(
                np.sum(
                    multi_mask
                )
            ),

        "z_kkt_min":
            z_kkt_min,

        "z_kkt_max":
            z_kkt_max,

        "z_multistable_min":
            z_multi_min,

        "z_multistable_max":
            z_multi_max,

        "max_abs_delta_eta":
            max_abs_deta,

        "min_lambda":
            min_lambda,

        "max_phi_p":
            max_phi_p,

        "min_phi_v":
            min_phi_v,

        "contact_classification":
            contact[
                "classification"
            ],

        "contact_global_type":
            contact[
                "global_type"
            ],

        "contact_phi_p":
            contact[
                "phi_p"
            ],

        "contact_eta":
            contact[
                "eta"
            ],

        "contact_phi_v":
            contact[
                "phi_v"
            ],

        "contact_delta_eta":
            contact[
                "delta_eta"
            ],

        "contact_lambda_min":
            contact[
                "lambda_min"
            ],

        "class_counts":
            str(
                class_counts
            ),
    }


# ============================================================
# 18. PLOTS FOR ONE CASE
# ============================================================

def plot_case(
    target,
    df
):

    z = df[
        "z"
    ].to_numpy()

    # --------------------------------------------------------
    # Profile plot
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            8.0,
            5.5
        )
    )

    plt.plot(
        z,
        df[
            "phi_p"
        ],
        label=r"$\phi_p$ nonlinear"
    )

    plt.plot(
        z,
        df[
            "eta"
        ],
        label=r"$\eta$ nonlinear"
    )

    plt.plot(
        z,
        df[
            "phi_v"
        ],
        label=r"$\phi_v$ nonlinear"
    )

    plt.plot(
        z,
        df[
            "phi_p_CaseI"
        ],
        "--",
        label=r"$\phi_p$ Case I"
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.ylabel(
        "Volume fraction / occupied fraction"
    )

    plt.title(
        target[
            "label"
        ]
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"soft_regime_{target['slug']}_profiles.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Delta eta / softness plot
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            8.0,
            5.5
        )
    )

    plt.plot(
        z,
        df[
            "delta_eta"
        ],
        label=r"$\Delta\eta$"
    )

    plt.plot(
        z,
        df[
            "lambda_min"
        ],
        label=r"$\lambda_{\min}$"
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.title(
        target[
            "label"
        ]
        + ": nonlinearity / softness"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"soft_regime_{target['slug']}_curvature.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Free-energy gap plot
    # --------------------------------------------------------

    gap = df[
        "gap_to_second"
    ].to_numpy()

    fig = plt.figure(
        figsize=(
            8.0,
            5.5
        )
    )

    plt.plot(
        z,
        gap
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.ylabel(
        r"$g_2-g_1$"
    )

    plt.title(
        target[
            "label"
        ]
        + ": gap to second certified state"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"soft_regime_{target['slug']}_free_energy_gap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 19. CONTACT LANDSCAPE
#
# Only for kappa=0.1, chi_ps=0.6 at z=0.
# This is a diagnostic grid, not the production solver.
# ============================================================

def contact_landscape(
    par,
    wall,
    slug
):

    z = 0.0

    theta_grid = np.linspace(
        0.002,
        0.995,
        LANDSCAPE_THETA_POINTS
    )

    eta_grid = np.linspace(
        0.70,
        1.0,
        LANDSCAPE_ETA_POINTS
    )

    rows = []

    g_matrix = np.full(
        (
            len(eta_grid),
            len(theta_grid)
        ),
        np.nan
    )

    g_min = np.inf

    for ie, eta in enumerate(
        eta_grid
    ):

        for it, theta in enumerate(
            theta_grid
        ):

            x = (
                eta * theta
            )

            g = grand_density(
                x,
                eta,
                z,
                par,
                wall
            )

            g_matrix[
                ie,
                it
            ] = g

            g_min = min(
                g_min,
                g
            )

    dg_matrix = (
        g_matrix
        -
        g_min
    )

    for ie, eta in enumerate(
        eta_grid
    ):

        for it, theta in enumerate(
            theta_grid
        ):

            rows.append(
                {
                    "theta":
                        theta,

                    "eta":
                        eta,

                    "phi_p":
                        theta * eta,

                    "phi_v":
                        1.0 - eta,

                    "delta_g_from_grid_min":
                        dg_matrix[
                            ie,
                            it
                        ],
                }
            )

    df = pd.DataFrame(
        rows
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"soft_regime_{slug}_contact_landscape.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # Contour plot.
    fig = plt.figure(
        figsize=(
            8.0,
            6.0
        )
    )

    positive = dg_matrix[
        np.isfinite(
            dg_matrix
        )
    ]

    if len(positive) > 0:

        vmax = np.nanpercentile(
            positive,
            95.0
        )

        if vmax <= 0.0:

            vmax = np.nanmax(
                positive
            )

        levels = np.linspace(
            0.0,
            vmax,
            30
        )

        contour = plt.contourf(
            theta_grid,
            eta_grid,
            dg_matrix,
            levels=levels
        )

        plt.colorbar(
            contour,
            label=r"$g-g_{\min}$"
        )

    plt.xlabel(
        r"$\theta_p=\phi_p/\eta$"
    )

    plt.ylabel(
        r"$\eta$"
    )

    plt.title(
        r"Contact landscape: "
        r"$\kappa=0.1,\ \chi_{ps}=0.6,\ z=0$"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"soft_regime_{slug}_contact_landscape.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 20. OPTIONAL INPUT AUDIT
# ============================================================

def print_previous_step_status():

    print()
    print("=" * 120)
    print(
        "PREVIOUS-STEP INPUT AUDIT"
    )
    print("=" * 120)

    if os.path.exists(
        STEP14_FILE
    ):

        try:

            df14 = pd.read_csv(
                STEP14_FILE
            )

            for target in TARGETS:

                sub = df14[
                    (
                        np.isclose(
                            df14[
                                "kappa"
                            ],
                            target[
                                "kappa"
                            ]
                        )
                    )
                    &
                    (
                        np.isclose(
                            df14[
                                "chi_ps"
                            ],
                            target[
                                "chi_ps"
                            ]
                        )
                    )
                ]

                if len(sub) > 0:

                    print(
                        target[
                            "label"
                        ],
                        "Equilibrium final_class =",
                        sub.iloc[0].get(
                            "final_class",
                            "N/A"
                        )
                    )

        except Exception as exc:

            print(
                "Could not read equilibrium table:",
                exc
            )

    else:

        print(
            "Equilibrium table not found:",
            STEP14_FILE
        )

    if os.path.exists(
        STEP16_FILE
    ):

        try:

            df16 = pd.read_csv(
                STEP16_FILE
            )

            for target in TARGETS:

                sub = df16[
                    (
                        np.isclose(
                            df16[
                                "kappa"
                            ],
                            target[
                                "kappa"
                            ]
                        )
                    )
                    &
                    (
                        np.isclose(
                            df16[
                                "chi_ps"
                            ],
                            target[
                                "chi_ps"
                            ]
                        )
                    )
                ]

                if len(sub) > 0:

                    print(
                        target[
                            "label"
                        ],
                        "Global LR status =",
                        sub.iloc[0].get(
                            "status",
                            "N/A"
                        ),
                        "LR quality =",
                        sub.iloc[0].get(
                            "LR_quality",
                            "N/A"
                        )
                    )

        except Exception as exc:

            print(
                "Could not read global LR table:",
                exc
            )

    else:

        print(
            "Global LR table not found:",
            STEP16_FILE
        )


# ============================================================
# 21. MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print_previous_step_status()

    base_par = ModelParams(

        r_p=10.0,

        phi_p_b=0.02,

        phi_v_b=0.05,

        chi_pp=-0.5,

        chi_ps=0.2,

        chi_ss=0.0,

        kappa=20.0,

        delta_u_int=5.0
    )

    wall = WallParams(

        eps_N=1.0,

        eps_D=3.0,

        lambda_w=1.0
    )

    z_grid = make_z_grid()

    all_dfs = []
    summaries = []

    for target in TARGETS:

        par, df = run_target(

            target,
            base_par,
            wall,
            z_grid
        )

        all_dfs.append(
            df
        )

        summary = summarize_case(

            target,
            df,
            par
        )

        summaries.append(
            summary
        )

        plot_case(
            target,
            df
        )

        case_csv = os.path.join(

            OUTPUT_DIR,

            f"soft_regime_{target['slug']}_pointwise.csv"
        )

        df.to_csv(
            case_csv,
            index=False
        )

        # Contact landscape for the problematic point.
        if (
            np.isclose(
                par.kappa,
                0.1
            )
            and
            np.isclose(
                par.chi_ps,
                0.6
            )
        ):

            contact_landscape(
                par,
                wall,
                target["slug"]
            )

    # ========================================================
    # COMBINED POINTWISE CSV
    # ========================================================

    full_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    full_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "soft_regime_pointwise_certification.csv"
        ),

        index=False
    )

    # ========================================================
    # CASE SUMMARY CSV
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "soft_regime_state_summary.csv"
        ),

        index=False
    )

    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    txt_path = os.path.join(
        OUTPUT_DIR,
        "soft_regime_certification_summary.txt"
    )

    with open(
        txt_path,
        "w"
    ) as f:

        f.write(
            "SOFT-DENSITY REGIME CERTIFICATION LOW-KAPPA / HIGH-CHI BOUNDARY CERTIFICATION\n"
        )

        f.write(
            "=" * 90
            + "\n\n"
        )

        f.write(
            "Target points:\n"
        )

        for target in TARGETS:

            f.write(
                "  "
                + target[
                    "label"
                ]
                + "\n"
            )

        f.write(
            "\nClassification meanings:\n\n"
        )

        f.write(
            "INTERIOR       : one certified stable interior minimum\n"
        )

        f.write(
            "KKT            : one certified eta=1 constrained minimum\n"
        )

        f.write(
            "MULTI_INTERIOR : multiple stable minima, interior global\n"
        )

        f.write(
            "MULTI_KKT      : multiple stable minima, boundary global\n"
        )

        f.write(
            "COEXIST        : two lowest certified minima within GAP_TOL\n"
        )

        f.write(
            "UNRESOLVED     : no certified nonlinear local state\n\n"
        )

        for s in summaries:

            f.write(
                "-" * 90
                + "\n"
            )

            f.write(
                s[
                    "label"
                ]
                + "\n"
            )

            for key, value in s.items():

                if key in (
                    "label",
                    "slug"
                ):

                    continue

                f.write(
                    f"  {key} = {value}\n"
                )

            f.write(
                "\n"
            )

        f.write(
            "\nInterpretation guide:\n\n"
        )

        f.write(
            "1. If kappa=0.1, chi_ps=0.6 is KKT at z=0, "
            "the Step-16 interior solver failure was physical boundary saturation.\n"
        )

        f.write(
            "2. If it is INTERIOR with a certified solution, "
            "the Step-16 failure was numerical.\n"
        )

        f.write(
            "3. If MULTI_* or COEXIST appears, "
            "the local LDA enters a bistable branch regime.\n"
        )

        f.write(
            "4. Large |delta eta| and small lambda_min mark "
            "strongly nonlinear/soft response where linear response and "
            "the harmonic EOS lose quantitative reliability.\n"
        )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 120)
    print(
        "SOFT-DENSITY REGIME CERTIFICATION CASE SUMMARY"
    )
    print("=" * 120)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Files written to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Primary files:"
    )

    print(
        "  soft_regime_pointwise_certification.csv"
    )

    print(
        "  soft_regime_state_summary.csv"
    )

    print(
        "  soft_regime_certification_summary.txt"
    )

    print()

    print(
        "Most important diagnostic for the failed Step-16 point:"
    )

    print(
        "  soft_regime_k01_chi06_contact_landscape.csv"
    )

    print(
        "  soft_regime_k01_chi06_contact_landscape.png"
    )

    print()
    print("=" * 120)
