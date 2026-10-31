import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, replace
from scipy.optimize import brentq, least_squares


# ============================================================
# STEP 15
#
# LINEAR-RESPONSE MECHANISM AUDIT
#
# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
#
# Use clean BG-I+ representative states from Step 14:
#
#   A: kappa=20,  chi_ps= 0.2
#      baseline / very weak compressibility response
#
#   B: kappa=2,   chi_ps=-0.6
#      adsorption enhancement with void depletion
#
#   C: kappa=0.5, chi_ps= 0.4
#      adsorption enhancement with void enrichment
#
#
# At each z:
#
#   Case I:
#
#       eta = eta_b
#       R_x(x_I, eta_b) = 0
#
#
#   Case II:
#
#       R_x(x_II, eta_II)   = 0
#       R_eta(x_II,eta_II)  = 0
#
#
# Linear response around Case I:
#
#       H [delta x, delta eta]^T
#       =
#       -[0, R_eta^I]^T
#
#
# therefore
#
#       delta x_LR
#         = H_xeta / det(H) * R_eta^I
#
#       delta eta_LR
#         = -H_xx / det(H) * R_eta^I
#
#
# Compare:
#
#       delta x_exact
#         = x_II - x_I
#
#       delta eta_exact
#         = eta_II - eta_b
#
#
# Integrated:
#
#       DD_Gamma_nonlinear
#         = integral dz (x_II - x_I)
#
#       DD_Gamma_LR
#         = integral dz delta x_LR
#
#
#       Delta Void_exact
#         = - integral dz delta eta_exact
#
#       Delta Void_LR
#         = - integral dz delta eta_LR
#
#
# OUTPUT
# ------------------------------------------------------------
#
# 02_representative_linear_response_output/
#
#   representative_linear_response_summary.csv
#   representative_linear_response_summary.txt
#
#   representative_A_baseline.csv
#   representative_B_free_volume_depletion.csv
#   representative_C_free_volume_enrichment.csv
#
#   step15_A_delta_phi_p.png
#   step15_A_delta_eta.png
#   ...
#
# Spatial coordinate is treated as reduced z.
#
# ============================================================


# ============================================================
# 1. MODEL PARAMETERS
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
# 2. NUMERICAL SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "02_representative_linear_response_output")

L = 20.0
DZ = 0.05

EPS = 1.0e-12

ROOT_TOL = 1.0e-9
HESS_TOL = 1.0e-9

CASEI_SCAN_POINTS = 3000

MAX_NFEV = 3000


# ============================================================
# 3. GRID
# ============================================================

def make_grid():

    n = int(
        round(
            L / DZ
        )
    )

    return np.linspace(
        0.0,
        L,
        n + 1
    )


# ============================================================
# 4. BULK QUANTITIES
# ============================================================

def bulk_quantities(par):

    x_b = par.phi_p_b

    eta_b = (
        1.0
        - par.phi_v_b
    )

    s_b = eta_b - x_b

    if not (
        0.0
        < x_b
        < eta_b
        < 1.0
    ):

        raise ValueError(
            "Require "
            "0 < phi_p_b < eta_b < 1."
        )

    return (
        x_b,
        s_b,
        eta_b
    )


# ============================================================
# 5. WALL / STATE PART
# ============================================================

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


def state_probabilities(
    z,
    par,
    wall
):

    E_N, E_D = state_energies(
        z,
        par,
        wall
    )

    lq = np.logaddexp(
        -E_N,
        -E_D
    )

    P_N = np.exp(
        -E_N - lq
    )

    P_D = np.exp(
        -E_D - lq
    )

    return (
        P_N,
        P_D
    )


# ============================================================
# 6. INTERACTION DERIVATIVE
# ============================================================

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
# 7. EXACT BULK-SUBTRACTED RESIDUALS
# ============================================================

def residual_x(
    x,
    eta,
    z,
    par,
    wall
):

    if not (
        0.0
        < x
        < eta
        <= 1.0
    ):

        return np.nan

    s = eta - x

    if s <= 0.0:

        return np.nan

    x_b, s_b, eta_b = (
        bulk_quantities(par)
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
        0.0
        < x
        < eta
        <= 1.0
    ):

        return np.nan

    s = eta - x

    if s <= 0.0:

        return np.nan

    x_b, s_b, eta_b = (
        bulk_quantities(par)
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

    det = (
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
        "Hxx":
            Hxx,

        "Hxeta":
            Hxe,

        "Hetaeta":
            Hee,

        "det":
            det,

        "lambda_min":
            eig[0],

        "lambda_max":
            eig[1]
    }


# ============================================================
# 9. CHEMICAL POTENTIALS FOR GRAND DENSITY
# ============================================================

def bulk_chemical_potentials(par):

    x_b, s_b, eta_b = (
        bulk_quantities(par)
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


# ============================================================
# 10. LOCAL GRAND DENSITY
#
# Used only to select the lowest Case-I stationary minimum
# if more than one root happens to be detected.
# ============================================================

def grand_density(
    x,
    eta,
    z,
    par,
    wall
):

    if not (
        0.0
        < x
        < eta
        <= 1.0
    ):

        return np.inf

    s = eta - x

    _, _, eta_b = (
        bulk_quantities(par)
    )

    mu_p, mu_s = (
        bulk_chemical_potentials(
            par
        )
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
# 11. CASE I SOLVER
# ============================================================

def solve_caseI_point(
    z,
    par,
    wall
):

    _, _, eta_b = (
        bulk_quantities(par)
    )

    theta_grid = np.linspace(
        1.0e-8,
        1.0 - 1.0e-8,
        CASEI_SCAN_POINTS
    )

    x_grid = (
        eta_b
        * theta_grid
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

        if abs(fa) < ROOT_TOL:

            root = x_grid[i]

        elif fa * fb < 0.0:

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

        else:

            continue

        if any(
            abs(
                root - old
            )
            < 1.0e-8
            for old in roots
        ):

            continue

        roots.append(
            root
        )

    candidates = []

    for x in roots:

        H = local_hessian(
            x,
            eta_b,
            par
        )

        # Case I only varies x.
        if H["Hxx"] <= HESS_TOL:

            continue

        candidates.append(
            {
                "x":
                    x,

                "eta":
                    eta_b,

                "g":
                    grand_density(
                        x,
                        eta_b,
                        z,
                        par,
                        wall
                    ),

                "Hxx":
                    H["Hxx"]
            }
        )

    if len(candidates) == 0:

        raise RuntimeError(
            f"No stable Case-I root at z={z}."
        )

    candidates.sort(
        key=lambda q:
            q["g"]
    )

    return candidates[0]


def solve_caseI_profile(
    z_grid,
    par,
    wall
):

    x = np.zeros_like(
        z_grid
    )

    eta = np.zeros_like(
        z_grid
    )

    for i, z in enumerate(
        z_grid
    ):

        sol = solve_caseI_point(
            z,
            par,
            wall
        )

        x[i] = sol["x"]
        eta[i] = sol["eta"]

    return {
        "x":
            x,

        "eta":
            eta,

        "phi_v":
            1.0 - eta
    }


# ============================================================
# 12. TRANSFORM FOR CASE II
# ============================================================

def sigmoid(a):

    if a >= 0.0:

        e = np.exp(-a)

        return (
            1.0
            / (
                1.0 + e
            )
        )

    e = np.exp(a)

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


def unpack_u(u):

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
        eta,
        theta
    )


def caseII_residual_u(
    u,
    z,
    par,
    wall
):

    x, eta, theta = (
        unpack_u(u)
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


# ============================================================
# 13. CASE II INTERIOR SOLVER
# ============================================================

def solve_caseII_point(
    z,
    par,
    wall,
    guess_list
):

    candidates = []

    for x0, eta0 in guess_list:

        if not (
            0.0
            < x0
            < eta0
            < 1.0
        ):

            continue

        theta0 = (
            x0 / eta0
        )

        u0 = np.array(
            [
                logit(theta0),
                logit(eta0)
            ]
        )

        try:

            sol = least_squares(

                caseII_residual_u,

                u0,

                args=(
                    z,
                    par,
                    wall
                ),

                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,

                max_nfev=MAX_NFEV
            )

        except Exception:

            continue

        x, eta, theta = (
            unpack_u(
                sol.x
            )
        )

        if not (
            0.0
            < x
            < eta
            < 1.0
        ):

            continue

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

        if Rmax > ROOT_TOL:

            continue

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

            continue

        candidate = {
            "x":
                x,

            "eta":
                eta,

            "g":
                grand_density(
                    x,
                    eta,
                    z,
                    par,
                    wall
                ),

            "Rmax":
                Rmax,

            "lambda_min":
                H["lambda_min"]
        }

        duplicate = False

        for old in candidates:

            distance = np.linalg.norm(
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

            if distance < 1.0e-8:

                duplicate = True
                break

        if not duplicate:

            candidates.append(
                candidate
            )

    if len(candidates) == 0:

        raise RuntimeError(
            f"No stable interior Case-II solution at z={z}."
        )

    candidates.sort(
        key=lambda q:
            q["g"]
    )

    return candidates[0]


def solve_caseII_profile(
    z_grid,
    par,
    wall,
    caseI
):

    x_b, _, eta_b = (
        bulk_quantities(par)
    )

    x = np.zeros_like(
        z_grid
    )

    eta = np.zeros_like(
        z_grid
    )

    Rmax = np.zeros_like(
        z_grid
    )

    lambda_min = np.zeros_like(
        z_grid
    )

    previous = None

    for i, z in enumerate(
        z_grid
    ):

        x_I = caseI["x"][i]

        guesses = []

        if previous is not None:

            guesses.append(
                previous
            )

        guesses.extend(
            [
                (
                    x_I,
                    eta_b
                ),
                (
                    x_b,
                    eta_b
                ),
                (
                    max(
                        1.0e-5,
                        0.8 * x_I
                    ),
                    min(
                        0.995,
                        eta_b
                    )
                ),
                (
                    min(
                        0.9 * eta_b,
                        1.2 * x_I
                    ),
                    min(
                        0.995,
                        eta_b
                    )
                )
            ]
        )

        sol = solve_caseII_point(
            z,
            par,
            wall,
            guesses
        )

        x[i] = sol["x"]
        eta[i] = sol["eta"]

        Rmax[i] = sol["Rmax"]

        lambda_min[i] = (
            sol["lambda_min"]
        )

        previous = (
            sol["x"],
            sol["eta"]
        )

    return {
        "x":
            x,

        "eta":
            eta,

        "phi_v":
            1.0 - eta,

        "Rmax":
            Rmax,

        "lambda_min":
            lambda_min
    }


# ============================================================
# 14. LINEAR-RESPONSE PROFILE
# ============================================================

def linear_response_profile(
    z_grid,
    par,
    caseI
):

    _, _, eta_b = (
        bulk_quantities(par)
    )

    n = len(
        z_grid
    )

    Reta_I = np.zeros(
        n
    )

    Hxx = np.zeros(
        n
    )

    Hxeta = np.zeros(
        n
    )

    Hetaeta = np.zeros(
        n
    )

    detH = np.zeros(
        n
    )

    lambda_min = np.zeros(
        n
    )

    delta_x = np.zeros(
        n
    )

    delta_eta = np.zeros(
        n
    )

    for i in range(n):

        x_I = caseI["x"][i]

        Reta = residual_eta(
            x_I,
            eta_b,
            par
        )

        H = local_hessian(
            x_I,
            eta_b,
            par
        )

        if H["det"] <= 0.0:

            raise RuntimeError(
                "Non-positive Hessian determinant "
                f"in LR calculation at z={z_grid[i]}."
            )

        Reta_I[i] = Reta

        Hxx[i] = H["Hxx"]
        Hxeta[i] = H["Hxeta"]
        Hetaeta[i] = H["Hetaeta"]
        detH[i] = H["det"]

        lambda_min[i] = (
            H["lambda_min"]
        )

        delta_x[i] = (

            H["Hxeta"]
            /
            H["det"]
            *
            Reta
        )

        delta_eta[i] = (

            -H["Hxx"]
            /
            H["det"]
            *
            Reta
        )

    return {
        "Reta_I":
            Reta_I,

        "Hxx":
            Hxx,

        "Hxeta":
            Hxeta,

        "Hetaeta":
            Hetaeta,

        "detH":
            detH,

        "lambda_min":
            lambda_min,

        "delta_x":
            delta_x,

        "delta_eta":
            delta_eta,

        "x_pred":
            caseI["x"]
            + delta_x,

        "eta_pred":
            eta_b
            + delta_eta
    }


# ============================================================
# 15. NUMERICAL INTEGRATION
# ============================================================

def integrate(
    y,
    z
):

    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            y,
            z
        )

    return np.trapz(
        y,
        z
    )


# ============================================================
# 16. ERROR METRICS
# ============================================================

def rmse(
    exact,
    pred
):

    return np.sqrt(
        np.mean(
            (
                exact - pred
            )**2
        )
    )


def normalized_rmse(
    exact,
    pred
):

    scale = np.max(
        np.abs(
            exact
        )
    )

    if scale < 1.0e-15:

        return np.nan

    return (
        rmse(
            exact,
            pred
        )
        /
        scale
    )


def correlation(
    exact,
    pred
):

    if (
        np.std(exact)
        < 1.0e-15

        or

        np.std(pred)
        < 1.0e-15
    ):

        return np.nan

    return np.corrcoef(
        exact,
        pred
    )[0, 1]


def relative_error(
    exact,
    pred
):

    if abs(exact) < 1.0e-15:

        return np.nan

    return (
        100.0
        * (
            pred - exact
        )
        / exact
    )


# ============================================================
# 17. OPTIONAL EQUILIBRIUM CLASS CHECK
# ============================================================

def check_step14_class(
    chi_ps,
    kappa
):

    filename = os.path.join(
        SCRIPT_DIR,
        "01_equilibrium_output",
        "equilibrium_all_states.csv"
    )

    if not os.path.exists(
        filename
    ):

        return None

    try:

        df = pd.read_csv(
            filename
        )

    except Exception:

        return None

    match = df[
        (
            np.isclose(
                df["chi_ps"],
                chi_ps
            )
        )
        &
        (
            np.isclose(
                df["kappa"],
                kappa
            )
        )
    ]

    if len(match) == 0:

        return None

    return str(
        match.iloc[0][
            "final_class"
        ]
    )


# ============================================================
# 18. SINGLE REPRESENTATIVE CASE
# ============================================================

def run_representative(
    label,
    slug,
    par,
    wall,
    z_grid
):

    print()
    print("=" * 110)

    print(
        f"{label}: "
        f"kappa={par.kappa}, "
        f"chi_ps={par.chi_ps}"
    )

    print("=" * 110)

    step14_class = (
        check_step14_class(
            par.chi_ps,
            par.kappa
        )
    )

    if step14_class is not None:

        print(
            "equilibrium class =",
            step14_class
        )

        if step14_class != "BG-I+":

            print(
                "WARNING: representative point "
                "is not BG-I+ according to Step 14."
            )

    # --------------------------------------------------------
    # Exact profiles
    # --------------------------------------------------------

    caseI = solve_caseI_profile(
        z_grid,
        par,
        wall
    )

    caseII = solve_caseII_profile(
        z_grid,
        par,
        wall,
        caseI
    )

    # --------------------------------------------------------
    # LR
    # --------------------------------------------------------

    lr = linear_response_profile(
        z_grid,
        par,
        caseI
    )

    _, _, eta_b = (
        bulk_quantities(par)
    )

    dx_exact = (
        caseII["x"]
        -
        caseI["x"]
    )

    deta_exact = (
        caseII["eta"]
        -
        eta_b
    )

    dx_lr = lr[
        "delta_x"
    ]

    deta_lr = lr[
        "delta_eta"
    ]

    # --------------------------------------------------------
    # Integrated response
    # --------------------------------------------------------

    DD_Gamma_exact = integrate(
        dx_exact,
        z_grid
    )

    DD_Gamma_LR = integrate(
        dx_lr,
        z_grid
    )

    void_exact = (

        -integrate(
            deta_exact,
            z_grid
        )
    )

    void_LR = (

        -integrate(
            deta_lr,
            z_grid
        )
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    summary = {

        "label":
            label,

        "slug":
            slug,

        "kappa":
            par.kappa,

        "chi_ps":
            par.chi_ps,

        "DD_Gamma_nonlinear":
            DD_Gamma_exact,

        "DD_Gamma_LR":
            DD_Gamma_LR,

        "DD_Gamma_LR_error_percent":
            relative_error(
                DD_Gamma_exact,
                DD_Gamma_LR
            ),

        "void_response_nonlinear":
            void_exact,

        "void_LR":
            void_LR,

        "void_LR_error_percent":
            relative_error(
                void_exact,
                void_LR
            ),

        "dx_RMSE":
            rmse(
                dx_exact,
                dx_lr
            ),

        "dx_NRMSE":
            normalized_rmse(
                dx_exact,
                dx_lr
            ),

        "dx_correlation":
            correlation(
                dx_exact,
                dx_lr
            ),

        "deta_RMSE":
            rmse(
                deta_exact,
                deta_lr
            ),

        "deta_NRMSE":
            normalized_rmse(
                deta_exact,
                deta_lr
            ),

        "deta_correlation":
            correlation(
                deta_exact,
                deta_lr
            ),

        "max_abs_dx_nonlinear":
            np.max(
                np.abs(
                    dx_exact
                )
            ),

        "max_abs_dx_LR":
            np.max(
                np.abs(
                    dx_lr
                )
            ),

        "max_abs_delta_eta_nonlinear":
            np.max(
                np.abs(
                    deta_exact
                )
            ),

        "max_abs_deta_LR":
            np.max(
                np.abs(
                    deta_lr
                )
            ),

        "max_abs_Reta_I":
            np.max(
                np.abs(
                    lr["Reta_I"]
                )
            ),

        "min_detH_CaseI":
            np.min(
                lr["detH"]
            ),

        "min_lambda_CaseI":
            np.min(
                lr["lambda_min"]
            ),

        "min_lambda_CaseII":
            np.min(
                caseII[
                    "lambda_min"
                ]
            ),

        "max_nonlinear_residual_CaseII":
            np.max(
                caseII[
                    "Rmax"
                ]
            )
    }

    # --------------------------------------------------------
    # State probabilities
    # --------------------------------------------------------

    P_N = np.zeros_like(
        z_grid
    )

    P_D = np.zeros_like(
        z_grid
    )

    for i, z in enumerate(
        z_grid
    ):

        P_N[i], P_D[i] = (
            state_probabilities(
                z,
                par,
                wall
            )
        )

    # --------------------------------------------------------
    # Profile CSV
    # --------------------------------------------------------

    profile_df = pd.DataFrame(
        {
            "z_reduced":
                z_grid,

            "phi_p_CaseI":
                caseI["x"],

            "phi_p_CaseII":
                caseII["x"],

            "eta_CaseI":
                caseI["eta"],

            "eta_CaseII":
                caseII["eta"],

            "phi_v_CaseI":
                caseI["phi_v"],

            "phi_v_CaseII":
                caseII["phi_v"],

            "delta_phi_p_nonlinear":
                dx_exact,

            "delta_phi_p_LR":
                dx_lr,

            "delta_eta_nonlinear":
                deta_exact,

            "delta_eta_LR":
                deta_lr,

            "delta_phi_v_nonlinear":
                -deta_exact,

            "delta_phi_v_LR":
                -deta_lr,

            "Reta_CaseI":
                lr["Reta_I"],

            "Hxx_CaseI":
                lr["Hxx"],

            "Hxeta_CaseI":
                lr["Hxeta"],

            "Hetaeta_CaseI":
                lr["Hetaeta"],

            "detH_CaseI":
                lr["detH"],

            "lambda_min_CaseI":
                lr["lambda_min"],

            "lambda_min_CaseII":
                caseII[
                    "lambda_min"
                ],

            "CaseII_residual":
                caseII[
                    "Rmax"
                ],

            "P_N":
                P_N,

            "P_D":
                P_D
        }
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"representative_{slug}.csv"
    )

    profile_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Plot 1: delta phi_p
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.plot(
        z_grid,
        dx_exact,
        label="Nonlinear Case II - Case I",
        linewidth=2.0
    )

    plt.plot(
        z_grid,
        dx_lr,
        "--",
        label="Linear response",
        linewidth=2.0
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.ylabel(
        r"$\Delta\phi_p$"
    )

    plt.title(
        f"{label}: protein-density response"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"representative_{slug}_composition_response.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Plot 2: delta eta
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.plot(
        z_grid,
        deta_exact,
        label="Nonlinear Case II - Case I",
        linewidth=2.0
    )

    plt.plot(
        z_grid,
        deta_lr,
        "--",
        label="Linear response",
        linewidth=2.0
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.ylabel(
        r"$\Delta\eta$"
    )

    plt.title(
        f"{label}: occupied-density response"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"representative_{slug}_density_response.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Plot 3: LR driving residual
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.plot(
        z_grid,
        lr["Reta_I"],
        linewidth=2.0
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        "Reduced distance from wall, z"
    )

    plt.ylabel(
        r"$R_\eta^{I}$"
    )

    plt.title(
        f"{label}: density-relaxation driving residual"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"representative_{slug}_density_driver.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        "DD_Gamma nonlinear =",
        DD_Gamma_exact
    )

    print(
        "DD_Gamma LR    =",
        DD_Gamma_LR
    )

    print(
        "LR error (%)   =",
        summary[
            "DD_Gamma_LR_error_percent"
        ]
    )

    print()

    print(
        "Free-volume response nonlinear =",
        void_exact
    )

    print(
        "Void response LR    =",
        void_LR
    )

    print(
        "Void LR error (%)   =",
        summary[
            "void_LR_error_percent"
        ]
    )

    print()

    print(
        "max |delta eta| nonlinear =",
        summary[
            "max_abs_delta_eta_nonlinear"
        ]
    )

    print(
        "min Case-I eigenvalue =",
        summary[
            "min_lambda_CaseI"
        ]
    )

    print(
        "min Case-II eigenvalue =",
        summary[
            "min_lambda_CaseII"
        ]
    )

    print(
        "max Case-II residual =",
        summary[
            "max_nonlinear_residual_CaseII"
        ]
    )

    return summary


# ============================================================
# 19. MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    z_grid = make_grid()

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

    representatives = [

        (
            "A baseline",
            "A_baseline",
            replace(
                base_par,
                kappa=20.0,
                chi_ps=0.2
            )
        ),

        (
            "B free-volume depletion",
            "B_free_volume_depletion",
            replace(
                base_par,
                kappa=2.0,
                chi_ps=-0.6
            )
        ),

        (
            "C free-volume enrichment",
            "C_free_volume_enrichment",
            replace(
                base_par,
                kappa=0.5,
                chi_ps=0.4
            )
        )
    ]

    summaries = []

    print()
    print("=" * 110)
    print(
        "REPRESENTATIVE LINEAR-RESPONSE ANALYSIS"
    )
    print("=" * 110)

    for (
        label,
        slug,
        par
    ) in representatives:

        summary = run_representative(

            label,
            slug,
            par,
            wall,
            z_grid
        )

        summaries.append(
            summary
        )

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "representative_linear_response_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    summary_txt = os.path.join(
        OUTPUT_DIR,
        "representative_linear_response_summary.txt"
    )

    with open(
        summary_txt,
        "w"
    ) as f:

        f.write(
            "REPRESENTATIVE LINEAR-RESPONSE ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Linear-response equations:\n\n"
        )

        f.write(
            "delta x_LR "
            "= (H_xeta/detH) R_eta^I\n"
        )

        f.write(
            "delta eta_LR "
            "= -(H_xx/detH) R_eta^I\n\n"
        )

        for s in summaries:

            f.write(
                f"{s['label']}\n"
            )

            f.write(
                f"  kappa = {s['kappa']}\n"
            )

            f.write(
                f"  chi_ps = {s['chi_ps']}\n"
            )

            f.write(
                "  DD_Gamma nonlinear = "
                f"{s['DD_Gamma_nonlinear']:.12e}\n"
            )

            f.write(
                "  DD_Gamma LR = "
                f"{s['DD_Gamma_LR']:.12e}\n"
            )

            f.write(
                "  adsorption LR error (%) = "
                f"{s['DD_Gamma_LR_error_percent']:.6f}\n"
            )

            f.write(
                "  void exact = "
                f"{s['void_response_nonlinear']:.12e}\n"
            )

            f.write(
                "  void LR = "
                f"{s['void_LR']:.12e}\n"
            )

            f.write(
                "  void LR error (%) = "
                f"{s['void_LR_error_percent']:.6f}\n"
            )

            f.write(
                "  max |delta eta| = "
                f"{s['max_abs_delta_eta_nonlinear']:.12e}\n"
            )

            f.write(
                "  min Case-I eigenvalue = "
                f"{s['min_lambda_CaseI']:.12e}\n"
            )

            f.write(
                "  min Case-II eigenvalue = "
                f"{s['min_lambda_CaseII']:.12e}\n"
            )

            f.write(
                "  max nonlinear residual = "
                f"{s['max_nonlinear_residual_CaseII']:.12e}\n"
            )

            f.write(
                "\n"
            )

    # ========================================================
    # COMPARISON PLOT:
    # integrated exact vs LR adsorption response
    # ========================================================

    labels = [
        s["label"]
        for s in summaries
    ]

    exact_values = np.array(
        [
            s["DD_Gamma_nonlinear"]
            for s in summaries
        ]
    )

    lr_values = np.array(
        [
            s["DD_Gamma_LR"]
            for s in summaries
        ]
    )

    x_positions = np.arange(
        len(labels)
    )

    width = 0.35

    fig = plt.figure(
        figsize=(
            8.0,
            5.0
        )
    )

    plt.bar(
        x_positions
        - width / 2.0,
        exact_values,
        width,
        label="Nonlinear"
    )

    plt.bar(
        x_positions
        + width / 2.0,
        lr_values,
        width,
        label="Linear response"
    )

    plt.xticks(
        x_positions,
        labels,
        rotation=15
    )

    plt.ylabel(
        r"$\Delta\Delta\tilde{\Gamma}_p$"
    )

    plt.title(
        "Integrated compressibility-induced adsorption response"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "representative_adsorption_nonlinear_vs_LR.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 110)
    print(
        "STEP 15 SUMMARY TABLE"
    )
    print("=" * 110)

    columns_to_print = [

        "label",
        "kappa",
        "chi_ps",

        "DD_Gamma_nonlinear",
        "DD_Gamma_LR",
        "DD_Gamma_LR_error_percent",

        "void_response_nonlinear",
        "void_LR",
        "void_LR_error_percent",

        "max_abs_delta_eta_nonlinear",
        "min_lambda_CaseII"
    ]

    print(
        summary_df[
            columns_to_print
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Files written to:",
        OUTPUT_DIR
    )

    print()

    print(
        "Expected files:"
    )

    print(
        "  representative_linear_response_summary.csv"
    )

    print(
        "  representative_linear_response_summary.txt"
    )

    print(
        "  3 profile CSV files"
    )

    print(
        "  9 representative profile/driver plots"
    )

    print(
        "  representative_adsorption_nonlinear_vs_LR.png"
    )

    print()
    print("=" * 110)