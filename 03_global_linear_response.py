import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, replace
from scipy.optimize import brentq, least_squares


# ============================================================
# GLOBAL LINEAR-RESPONSE ANALYSIS
#
# GLOBAL LINEAR-RESPONSE MECHANISM MAP
#
# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
#
# Use Step-14 final admissible map and analyze ONLY:
#
#       final_class == "BG-I+"
#
# For every clean single-phase bulk-global point:
#
#   1. Solve exact Case I profile.
#
#   2. Solve exact Case II profile.
#
#   3. Compute linear response around Case I:
#
#          dx_LR
#          = (Hxeta / detH) * Reta_I
#
#          deta_LR
#          = -(Hxx / detH) * Reta_I
#
#   4. Compare exact and LR integrated adsorption:
#
#          DD_Gamma_nonlinear
#          DD_Gamma_LR
#
#   5. Analyze mechanism sign structure:
#
#          sign(Reta_I)
#          sign(Hxeta)
#          sign(Hxeta * Reta_I)
#
#   6. Save:
#
#       03_global_linear_response_output/
#
#       global_linear_response_regular_states.csv
#       global_linear_response_classification_matrix.csv
#       global_linear_response_summary.txt
#
#       linear_response_error_map.png
#       nonlinear_adsorption_response_map.png
#       free_volume_response_map.png
#       density_driver_integral_map.png
#       density_composition_coupling_map.png
#       coupling_driver_sign_map.png
#
#
# ------------------------------------------------------------
# NOTE
# ------------------------------------------------------------
#
# This step is a mechanism audit, not a new thermodynamic
# admissibility calculation.
#
# Step 14 remains the authority for BG/BM/BU/BI/KKT classes.
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
# 2. SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP14_FILE = os.path.join(SCRIPT_DIR, "01_equilibrium_output", "equilibrium_all_states.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "03_global_linear_response_output")

L = 20.0
DZ = 0.05

ROOT_TOL = 1.0e-9
HESS_TOL = 1.0e-9

CASEI_SCAN_POINTS = 2500

MAX_NFEV = 3000

LR_GOOD_ERROR_PERCENT = 5.0
LR_MODERATE_ERROR_PERCENT = 15.0


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
# 4. BULK
# ============================================================

def bulk_quantities(par):

    x_b = par.phi_p_b

    eta_b = (
        1.0
        - par.phi_v_b
    )

    s_b = eta_b - x_b

    return (
        x_b,
        s_b,
        eta_b
    )


# ============================================================
# 5. STATE / WALL
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


# ============================================================
# 6. INTERACTION
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
# 7. RESIDUALS
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

    x_b, s_b, eta_b = (
        bulk_quantities(par)
    )

    return (

        (
            1.0 / par.r_p
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
            1.0 / par.r_p
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
        "Hxx":
            Hxx,

        "Hxeta":
            Hxe,

        "Hetaeta":
            Hee,

        "det":
            detH,

        "lambda_min":
            eig[0]
    }


# ============================================================
# 9. BULK CHEMICAL POTENTIALS
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
# 10. GRAND DENSITY
# ============================================================

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
# 11. CASE I
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

        root = None

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

        if root is None:

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

        if H["Hxx"] <= HESS_TOL:

            continue

        candidates.append(
            {
                "x":
                    x,

                "g":
                    grand_density(
                        x,
                        eta_b,
                        z,
                        par,
                        wall
                    )
            }
        )

    if len(candidates) == 0:

        raise RuntimeError(
            f"Case-I failure at z={z}"
        )

    candidates.sort(
        key=lambda q:
            q["g"]
    )

    return candidates[0]["x"]


def solve_caseI_profile(
    z_grid,
    par,
    wall
):

    _, _, eta_b = (
        bulk_quantities(par)
    )

    x = np.zeros_like(
        z_grid
    )

    for i, z in enumerate(
        z_grid
    ):

        x[i] = solve_caseI_point(
            z,
            par,
            wall
        )

    return {
        "x":
            x,

        "eta":
            np.full_like(
                z_grid,
                eta_b
            )
    }


# ============================================================
# 12. CASE II TRANSFORM
# ============================================================

def sigmoid(a):

    if a >= 0.0:

        e = np.exp(-a)

        return (
            1.0
            /
            (
                1.0 + e
            )
        )

    e = np.exp(a)

    return (
        e
        /
        (
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
        eta
    )


def residual_u(
    u,
    z,
    par,
    wall
):

    x, eta = unpack_u(
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


# ============================================================
# 13. CASE II
# ============================================================

def solve_caseII_point(
    z,
    par,
    wall,
    guesses
):

    candidates = []

    for x0, eta0 in guesses:

        if not (
            0.0
            < x0
            < eta0
            < 1.0
        ):

            continue

        u0 = np.array(
            [
                logit(
                    x0 / eta0
                ),

                logit(
                    eta0
                )
            ]
        )

        try:

            sol = least_squares(

                residual_u,

                u0,

                args=(
                    z,
                    par,
                    wall
                ),

                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,

                max_nfev=
                    MAX_NFEV
            )

        except Exception:

            continue

        x, eta = unpack_u(
            sol.x
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

        c = {
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

            d = np.linalg.norm(
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

            if d < 1.0e-8:

                duplicate = True
                break

        if not duplicate:

            candidates.append(
                c
            )

    if len(candidates) == 0:

        raise RuntimeError(
            f"Case-II failure at z={z}"
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

    residual_max = np.zeros_like(
        z_grid
    )

    lambda_min = np.zeros_like(
        z_grid
    )

    previous = None

    for i, z in enumerate(
        z_grid
    ):

        xI = caseI["x"][i]

        guesses = []

        if previous is not None:

            guesses.append(
                previous
            )

        guesses.extend(
            [
                (
                    xI,
                    eta_b
                ),

                (
                    x_b,
                    eta_b
                ),

                (
                    max(
                        1.0e-6,
                        0.8 * xI
                    ),
                    eta_b
                ),

                (
                    min(
                        0.9 * eta_b,
                        1.2 * xI
                    ),
                    eta_b
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

        eta[i] = sol[
            "eta"
        ]

        residual_max[i] = (
            sol["Rmax"]
        )

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

        "Rmax":
            residual_max,

        "lambda_min":
            lambda_min
    }


# ============================================================
# 14. INTEGRATION
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
# 15. LR CLASS
# ============================================================

def classify_lr_error(
    error_percent
):

    a = abs(
        error_percent
    )

    if a <= LR_GOOD_ERROR_PERCENT:

        return "LR_GOOD"

    if a <= LR_MODERATE_ERROR_PERCENT:

        return "LR_MODERATE"

    return "LR_POOR"


# ============================================================
# 16. SINGLE POINT
# ============================================================

def analyze_point(
    par,
    wall,
    z_grid
):

    _, _, eta_b = (
        bulk_quantities(par)
    )

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

    n = len(
        z_grid
    )

    Reta = np.zeros(
        n
    )

    Hxeta = np.zeros(
        n
    )

    Hxx = np.zeros(
        n
    )

    detH = np.zeros(
        n
    )

    dx_lr = np.zeros(
        n
    )

    deta_lr = np.zeros(
        n
    )

    for i in range(n):

        xI = caseI["x"][i]

        Reta[i] = residual_eta(
            xI,
            eta_b,
            par
        )

        H = local_hessian(
            xI,
            eta_b,
            par
        )

        Hxeta[i] = (
            H["Hxeta"]
        )

        Hxx[i] = (
            H["Hxx"]
        )

        detH[i] = (
            H["det"]
        )

        dx_lr[i] = (

            Hxeta[i]
            /
            detH[i]
            *
            Reta[i]
        )

        deta_lr[i] = (

            -Hxx[i]
            /
            detH[i]
            *
            Reta[i]
        )

    DD_exact = integrate(
        dx_exact,
        z_grid
    )

    DD_lr = integrate(
        dx_lr,
        z_grid
    )

    void_exact = (

        -integrate(
            deta_exact,
            z_grid
        )
    )

    void_lr = (

        -integrate(
            deta_lr,
            z_grid
        )
    )

    if abs(DD_exact) > 1.0e-15:

        lr_error_percent = (

            100.0
            * (
                DD_lr
                -
                DD_exact
            )
            /
            DD_exact
        )

    else:

        lr_error_percent = (
            np.nan
        )

    # --------------------------------------------------------
    # Mechanistic integrals
    # --------------------------------------------------------

    driver_integral = integrate(
        Reta,
        z_grid
    )

    coupling_integral = integrate(
        Hxeta,
        z_grid
    )

    sign_product_integral = integrate(
        Hxeta * Reta,
        z_grid
    )

    response_kernel_integral = integrate(
        (
            Hxeta
            /
            detH
        )
        *
        Reta,
        z_grid
    )

    # --------------------------------------------------------
    # Sign fractions across profile
    # --------------------------------------------------------

    positive_driver_fraction = (
        np.mean(
            Reta > 0.0
        )
    )

    negative_driver_fraction = (
        np.mean(
            Reta < 0.0
        )
    )

    positive_Hxeta_fraction = (
        np.mean(
            Hxeta > 0.0
        )
    )

    negative_Hxeta_fraction = (
        np.mean(
            Hxeta < 0.0
        )
    )

    positive_product_fraction = (
        np.mean(
            Hxeta * Reta
            > 0.0
        )
    )

    negative_product_fraction = (
        np.mean(
            Hxeta * Reta
            < 0.0
        )
    )

    # --------------------------------------------------------
    # Wall values
    # --------------------------------------------------------

    wall_Reta = (
        Reta[0]
    )

    wall_Hxeta = (
        Hxeta[0]
    )

    wall_product = (
        wall_Reta
        *
        wall_Hxeta
    )

    wall_dx_exact = (
        dx_exact[0]
    )

    wall_dx_lr = (
        dx_lr[0]
    )

    wall_deta_exact = (
        deta_exact[0]
    )

    # --------------------------------------------------------
    # Profile errors
    # --------------------------------------------------------

    dx_rmse = np.sqrt(
        np.mean(
            (
                dx_exact
                -
                dx_lr
            )**2
        )
    )

    deta_rmse = np.sqrt(
        np.mean(
            (
                deta_exact
                -
                deta_lr
            )**2
        )
    )

    max_dx = np.max(
        np.abs(
            dx_exact
        )
    )

    max_deta = np.max(
        np.abs(
            deta_exact
        )
    )

    if max_dx > 0.0:

        dx_nrmse = (
            dx_rmse
            /
            max_dx
        )

    else:

        dx_nrmse = np.nan

    if max_deta > 0.0:

        deta_nrmse = (
            deta_rmse
            /
            max_deta
        )

    else:

        deta_nrmse = np.nan

    # --------------------------------------------------------
    # Exact response sign
    # --------------------------------------------------------

    if DD_exact > 0.0:

        response_sign = (
            "ENHANCE"
        )

    elif DD_exact < 0.0:

        response_sign = (
            "SUPPRESS"
        )

    else:

        response_sign = (
            "ZERO"
        )

    if void_exact > 0.0:

        void_sign = (
            "VOID_ENRICH"
        )

    elif void_exact < 0.0:

        void_sign = (
            "VOID_DEPLETE"
        )

    else:

        void_sign = (
            "VOID_ZERO"
        )

    return {

        "kappa":
            par.kappa,

        "chi_ps":
            par.chi_ps,

        "DD_Gamma_nonlinear":
            DD_exact,

        "DD_Gamma_LR":
            DD_lr,

        "LR_error_percent":
            lr_error_percent,

        "LR_quality":
            classify_lr_error(
                lr_error_percent
            ),

        "void_response_nonlinear":
            void_exact,

        "void_LR":
            void_lr,

        "response_sign":
            response_sign,

        "void_sign":
            void_sign,

        "driver_integral_Reta":
            driver_integral,

        "coupling_integral_Hxeta":
            coupling_integral,

        "sign_product_integral":
            sign_product_integral,

        "response_kernel_integral":
            response_kernel_integral,

        "positive_driver_fraction":
            positive_driver_fraction,

        "negative_driver_fraction":
            negative_driver_fraction,

        "positive_Hxeta_fraction":
            positive_Hxeta_fraction,

        "negative_Hxeta_fraction":
            negative_Hxeta_fraction,

        "positive_product_fraction":
            positive_product_fraction,

        "negative_product_fraction":
            negative_product_fraction,

        "wall_Reta":
            wall_Reta,

        "wall_Hxeta":
            wall_Hxeta,

        "wall_product":
            wall_product,

        "wall_dx_nonlinear":
            wall_dx_exact,

        "wall_dx_LR":
            wall_dx_lr,

        "wall_deta_nonlinear":
            wall_deta_exact,

        "max_abs_dx_nonlinear":
            max_dx,

        "max_abs_delta_eta_nonlinear":
            max_deta,

        "dx_NRMSE":
            dx_nrmse,

        "deta_NRMSE":
            deta_nrmse,

        "min_lambda_CaseII":
            np.min(
                caseII[
                    "lambda_min"
                ]
            ),

        "max_CaseII_residual":
            np.max(
                caseII[
                    "Rmax"
                ]
            )
    }


# ============================================================
# 17. MATRIX PLOT HELPER
# ============================================================

def make_matrix(
    df,
    value_col,
    kappas,
    chis
):

    matrix = np.full(
        (
            len(chis),
            len(kappas)
        ),
        np.nan
    )

    for iy, chi in enumerate(
        chis
    ):

        for ix, kappa in enumerate(
            kappas
        ):

            sub = df[
                (
                    np.isclose(
                        df["chi_ps"],
                        chi
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

            if len(sub) == 0:

                continue

            matrix[
                iy,
                ix
            ] = float(
                sub.iloc[0][
                    value_col
                ]
            )

    return matrix


def save_heatmap(
    matrix,
    kappas,
    chis,
    title,
    cbar_label,
    filename
):

    fig = plt.figure(
        figsize=(
            9.0,
            6.5
        )
    )

    im = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto"
    )

    plt.xticks(
        np.arange(
            len(kappas)
        ),
        [
            str(k)
            for k in kappas
        ]
    )

    plt.yticks(
        np.arange(
            len(chis)
        ),
        [
            str(c)
            for c in chis
        ]
    )

    plt.xlabel(
        r"$\kappa$"
    )

    plt.ylabel(
        r"$\chi_{ps}$"
    )

    plt.title(
        title
    )

    cbar = plt.colorbar(
        im
    )

    cbar.set_label(
        cbar_label
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not os.path.exists(
        STEP14_FILE
    ):

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"  {STEP14_FILE}\n\n"
            "Run 01_equilibrium_regime_map.py first to create the required equilibrium table."
        )

    step14 = pd.read_csv(
        STEP14_FILE
    )

    if "final_class" not in step14.columns:

        raise ValueError(
            "Step-14 CSV does not contain "
            "'final_class'."
        )

    target = step14[
        step14[
            "final_class"
        ]
        == "BG-I+"
    ].copy()

    print()
    print("=" * 130)

    print(
        "GLOBAL LINEAR-RESPONSE ANALYSIS: GLOBAL LINEAR-RESPONSE MECHANISM MAP"
    )

    print("=" * 130)

    print(
        "Step-14 BG-I+ points =",
        len(target)
    )

    if len(target) == 0:

        raise RuntimeError(
            "No BG-I+ points found."
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

    results = []

    for count, (_, row) in enumerate(
        target.iterrows(),
        start=1
    ):

        kappa = float(
            row["kappa"]
        )

        chi_ps = float(
            row["chi_ps"]
        )

        print(
            f"[{count:02d}/{len(target):02d}] "
            f"kappa={kappa:7.3f} "
            f"chi_ps={chi_ps:7.3f}"
        )

        par = replace(

            base_par,

            kappa=kappa,

            chi_ps=chi_ps
        )

        try:

            out = analyze_point(
                par,
                wall,
                z_grid
            )

            out[
                "status"
            ] = "PASS"

        except Exception as exc:

            print(
                "  FAILED:",
                str(exc)
            )

            out = {

                "kappa":
                    kappa,

                "chi_ps":
                    chi_ps,

                "status":
                    "FAIL",

                "failure_reason":
                    str(exc)
            }

        results.append(
            out
        )

    result_df = pd.DataFrame(
        results
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        "global_linear_response_regular_states.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    passed = result_df[
        result_df[
            "status"
        ]
        == "PASS"
    ].copy()

    # ========================================================
    # 19. SUMMARY COUNTS
    # ========================================================

    print()
    print("=" * 130)

    print(
        "GLOBAL LINEAR-RESPONSE ANALYSIS SUMMARY"
    )

    print("=" * 130)

    print(
        "PASS =",
        len(passed)
    )

    print(
        "FAIL =",
        len(result_df)
        -
        len(passed)
    )

    if len(passed) > 0:

        print()

        print(
            "LR quality:"
        )

        print(
            passed[
                "LR_quality"
            ].value_counts()
        )

        print()

        print(
            "Response sign:"
        )

        print(
            passed[
                "response_sign"
            ].value_counts()
        )

        print()

        print(
            "Void sign:"
        )

        print(
            passed[
                "void_sign"
            ].value_counts()
        )

        print()

        print(
            "Worst |LR adsorption error| (%) =",
            np.max(
                np.abs(
                    passed[
                        "LR_error_percent"
                    ]
                )
            )
        )

        idx = np.argmax(
            np.abs(
                passed[
                    "LR_error_percent"
                ].to_numpy()
            )
        )

        worst = passed.iloc[
            idx
        ]

        print(
            "Worst LR point:"
        )

        print(
            "  kappa =",
            worst["kappa"]
        )

        print(
            "  chi_ps =",
            worst["chi_ps"]
        )

        print(
            "  error (%) =",
            worst[
                "LR_error_percent"
            ]
        )

        print(
            "  max |delta eta| =",
            worst[
                "max_abs_delta_eta_nonlinear"
            ]
        )

        print(
            "  min lambda =",
            worst[
                "min_lambda_CaseII"
            ]
        )

    # ========================================================
    # 20. MATRICES
    # ========================================================

    kappas = sorted(
        passed[
            "kappa"
        ].unique(),
        reverse=True
    )

    chis = sorted(
        passed[
            "chi_ps"
        ].unique()
    )

    # matrix CSV with LR quality
    matrix_rows = []

    for chi in chis:

        row = {
            "chi_ps":
                chi
        }

        for kappa in kappas:

            sub = passed[
                (
                    np.isclose(
                        passed[
                            "chi_ps"
                        ],
                        chi
                    )
                )
                &
                (
                    np.isclose(
                        passed[
                            "kappa"
                        ],
                        kappa
                    )
                )
            ]

            if len(sub) == 0:

                val = ""

            else:

                val = str(
                    sub.iloc[0][
                        "LR_quality"
                    ]
                )

            row[
                str(kappa)
            ] = val

        matrix_rows.append(
            row
        )

    matrix_df = pd.DataFrame(
        matrix_rows
    )

    matrix_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_linear_response_classification_matrix.csv"
        ),
        index=False
    )

    # ========================================================
    # 21. HEATMAPS
    # ========================================================

    heatmaps = [

        (
            "LR_error_percent",
            "Linear-response adsorption error",
            "LR error (%)",
            "linear_response_error_map.png"
        ),

        (
            "DD_Gamma_nonlinear",
            "Nonlinear density-relaxation adsorption response",
            r"$\Delta\Delta\tilde{\Gamma}_p$",
            "nonlinear_adsorption_response_map.png"
        ),

        (
            "void_response_nonlinear",
            "Integrated void response",
            "Integrated void excess",
            "free_volume_response_map.png"
        ),

        (
            "driver_integral_Reta",
            r"Integrated density-relaxation driver",
            r"$\int R_\eta^I dz$",
            "density_driver_integral_map.png"
        ),

        (
            "coupling_integral_Hxeta",
            r"Integrated density-composition coupling",
            r"$\int H_{x\eta} dz$",
            "density_composition_coupling_map.png"
        ),

        (
            "sign_product_integral",
            r"Integrated coupling-driver product",
            r"$\int H_{x\eta}R_\eta^I dz$",
            "coupling_driver_sign_map.png"
        )
    ]

    for (
        col,
        title,
        label,
        filename
    ) in heatmaps:

        matrix = make_matrix(
            passed,
            col,
            kappas,
            chis
        )

        save_heatmap(
            matrix,
            kappas,
            chis,
            title,
            label,
            filename
        )

    # ========================================================
    # 22. LR ERROR VS DENSITY EXCURSION
    # ========================================================

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.scatter(
        passed[
            "max_abs_delta_eta_nonlinear"
        ],
        np.abs(
            passed[
                "LR_error_percent"
            ]
        )
    )

    plt.xlabel(
        r"Maximum $|\Delta\eta|$"
    )

    plt.ylabel(
        "Absolute LR adsorption error (%)"
    )

    plt.title(
        "Linear-response error vs density excursion"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "linear_response_error_vs_density_excursion.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # ========================================================
    # 23. LR ERROR VS SOFTNESS
    # ========================================================

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.scatter(
        passed[
            "min_lambda_CaseII"
        ],
        np.abs(
            passed[
                "LR_error_percent"
            ]
        )
    )

    plt.xlabel(
        "Minimum Case-II Hessian eigenvalue"
    )

    plt.ylabel(
        "Absolute LR adsorption error (%)"
    )

    plt.title(
        "Linear-response error vs thermodynamic softness"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "linear_response_error_vs_minimum_curvature.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # ========================================================
    # 24. MECHANISM SCATTER
    # ========================================================

    fig = plt.figure(
        figsize=(
            7.0,
            5.0
        )
    )

    plt.scatter(
        passed[
            "sign_product_integral"
        ],
        passed[
            "DD_Gamma_nonlinear"
        ]
    )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.axvline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        r"$\int H_{x\eta}R_\eta^I dz$"
    )

    plt.ylabel(
        r"Nonlinear $\Delta\Delta\tilde{\Gamma}_p$"
    )

    plt.title(
        "Adsorption response vs coupling-driver product"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "adsorption_response_vs_coupling_driver.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # ========================================================
    # 25. SUMMARY TXT
    # ========================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "global_linear_response_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            "GLOBAL LINEAR-RESPONSE ANALYSIS GLOBAL LINEAR-RESPONSE MECHANISM MAP\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Total BG-I+ input points = {len(target)}\n"
        )

        f.write(
            f"PASS = {len(passed)}\n"
        )

        f.write(
            "FAIL = "
            f"{len(result_df)-len(passed)}\n\n"
        )

        if len(passed) > 0:

            f.write(
                "LR QUALITY COUNTS\n"
            )

            f.write(
                passed[
                    "LR_quality"
                ]
                .value_counts()
                .to_string()
            )

            f.write(
                "\n\n"
            )

            f.write(
                "RESPONSE SIGN COUNTS\n"
            )

            f.write(
                passed[
                    "response_sign"
                ]
                .value_counts()
                .to_string()
            )

            f.write(
                "\n\n"
            )

            f.write(
                "VOID SIGN COUNTS\n"
            )

            f.write(
                passed[
                    "void_sign"
                ]
                .value_counts()
                .to_string()
            )

            f.write(
                "\n\n"
            )

            f.write(
                "LR equations:\n\n"
            )

            f.write(
                "delta x_LR "
                "= (Hxeta/detH) Reta_I\n"
            )

            f.write(
                "delta eta_LR "
                "= -(Hxx/detH) Reta_I\n\n"
            )

            f.write(
                "Mechanistic interpretation:\n"
            )

            f.write(
                "The adsorption response is generated by "
                "the density-relaxation residual Reta_I, "
                "filtered through the coupled Hessian.\n\n"
            )

            f.write(
                "Void enrichment is not required for "
                "positive adsorption response because the "
                "sign of delta x depends on the product "
                "Hxeta * Reta_I rather than on delta eta "
                "alone.\n\n"
            )

            worst_idx = np.argmax(
                np.abs(
                    passed[
                        "LR_error_percent"
                    ].to_numpy()
                )
            )

            worst = passed.iloc[
                worst_idx
            ]

            f.write(
                "Worst LR point:\n"
            )

            f.write(
                f"  kappa = {worst['kappa']}\n"
            )

            f.write(
                f"  chi_ps = {worst['chi_ps']}\n"
            )

            f.write(
                "  LR error (%) = "
                f"{worst['LR_error_percent']}\n"
            )

            f.write(
                "  max |delta eta| = "
                f"{worst['max_abs_delta_eta_nonlinear']}\n"
            )

            f.write(
                "  min lambda = "
                f"{worst['min_lambda_CaseII']}\n"
            )

    # ========================================================
    # 26. FINAL PRINT
    # ========================================================

    print()
    print(
        "Files written to:",
        OUTPUT_DIR
    )

    print()

    print(
        "Primary data:"
    )

    print(
        "  global_linear_response_regular_states.csv"
    )

    print(
        "  global_linear_response_classification_matrix.csv"
    )

    print(
        "  global_linear_response_summary.txt"
    )

    print()

    print(
        "Primary figures:"
    )

    print(
        "  linear_response_error_map.png"
    )

    print(
        "  nonlinear_adsorption_response_map.png"
    )

    print(
        "  free_volume_response_map.png"
    )

    print(
        "  density_driver_integral_map.png"
    )

    print(
        "  density_composition_coupling_map.png"
    )

    print(
        "  coupling_driver_sign_map.png"
    )

    print(
        "  linear_response_error_vs_density_excursion.png"
    )

    print(
        "  linear_response_error_vs_minimum_curvature.png"
    )

    print(
        "  adsorption_response_vs_coupling_driver.png"
    )

    print()
    print("=" * 130)