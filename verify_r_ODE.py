"""Verify large-alpha asymptotics of the coupled Q and r order-parameter ODEs.

Compare fitted trajectory prefactors and the plateau Q*r against analytical
predictions for the mismatched sign-erf perceptron. Separately check the bare
subleading coefficient r5 at fixed s = Q*r and the effective coefficient K
along the evolving trajectory. These are ODE checks, not microscopic SGD runs.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import linregress

# Learning rates, integration interval, and initial order parameters (Q, r)
etas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
alpha_span = (1e-2, 1e7)
X0 = (1.0, 0.5)
n_eval = 500

# Fixed-Q evaluation and late-time fitting interval
Q_fixed = np.geomspace(20.0, 3000.0, 100)
alpha_min_fit = 0.9e6
alpha_max_fit = 1e6

# Sample the full evolution and resolve the original late-time fitting window
alpha_eval = np.unique(np.concatenate((
    np.geomspace(*alpha_span, n_eval),
    np.linspace(alpha_min_fit, alpha_max_fit, 120),
)))

# Analytical functions for the large-alpha expansion with s = Q*r
def G(s):
    return np.pi + 2 * np.arcsin(1 / 3) - 4 * np.arcsin(1 / np.sqrt(3 * (1 + 4 * s)))


def G_prime(s):
    return 8 / ((1 + 4 * s) * np.sqrt(2 + 12 * s))


def r3(s, eta):
    return -(4 * eta / np.pi) * s / np.sqrt(1 + 2 * s) + eta**2 / (np.pi**2 * np.sqrt(2)) * G(s)


def r3_prime(s, eta):
    return -(4 * eta / np.pi) * (1 + s) / (1 + 2 * s)**1.5 + eta**2 / (np.pi**2 * np.sqrt(2)) * G_prime(s)


def c_Q(s, eta):
    return ((4 * eta / np.pi) * (1 / np.sqrt(1 + 2 * s) - 1 / np.sqrt(2))
            + 2 * eta**2 / (np.pi**2 * np.sqrt(2)) * G(s))


def B(s):
    return (2 * (1 + 10 * s + 18 * s**2) / (3 * (1 + 4 * s) * np.sqrt(2 + 12 * s))
            - 1 / (3 * np.sqrt(2)))


def r5(s, eta):
    return ((2 * eta / np.pi) * s**2 * (1 + s) / (1 + 2 * s)**1.5
            + eta**2 / (np.pi**2 * np.sqrt(2)) * (B(s) - (s + 0.25) * G(s)))


# Predict the limiting plateau, growth prefactors, and subleading coefficients
def analytical_predictions(eta):
    upper = 1.0
    while r3(upper, eta) > 0:
        upper *= 2
        if upper > 1e10:
            raise RuntimeError("Unable to bracket physical root")

    s_star = brentq(lambda s: r3(s, eta), 0.0, upper, xtol=1e-15)
    c = c_Q(s_star, eta)
    r5_star = r5(s_star, eta)
    r3_prime_star = r3_prime(s_star, eta)
    beta = -(r5_star + s_star * c) / r3_prime_star
    C_Q = (1.5 * c)**(2 / 3)

    return {
        "s_star": s_star,
        "c": c,
        "C_Q": C_Q,
        "C_rho": np.sqrt(C_Q),
        "C_r": s_star / C_Q,
        "r5": r5_star,
        "r3_prime": r3_prime_star,
        "beta": beta,
        "K": -s_star * c,
    }


# Exact coupled equations for Q and r
def d_Q_r(alpha, X, eta):
    Q, r = X
    if Q <= 0:
        raise ValueError("Q must remain positive")

    rho = np.sqrt(Q) * (1 - r)
    sqrt_2Q1 = np.sqrt(2 * Q + 1)
    sqrt_term = np.sqrt(1 + Q - rho**2)
    arg1 = Q / (3 * Q + 1)
    arg2 = np.sqrt(Q) * (1 - r) / np.sqrt((3 * Q + 1) * (1 - 2 * Q * r * (r - 2)))
    bracket = np.pi + 2 * np.arcsin(np.clip(arg1, -1, 1)) - 4 * np.arcsin(np.clip(arg2, -1, 1))

    d_Q = (
        (4 * eta / (np.pi * (Q + 1))) * (rho / sqrt_term - Q / sqrt_2Q1)
        + (2 * eta**2 / (np.pi**2 * sqrt_2Q1)) * bracket
    )
    d_r = (
        (2 * eta / np.pi) * r * (r - 2) / np.sqrt(Q * (1 - Q * r * (r - 2)))
        + (eta**2 * (1 - r) / (np.pi**2 * Q * sqrt_2Q1)) * bracket
    )
    return [d_Q, d_r]


# Fit y = C * alpha^p over the original late-time interval
def power_law_fit(alpha, y, alpha_min_fit, alpha_max_fit):
    mask = ((alpha >= alpha_min_fit) & (alpha <= alpha_max_fit)
            & (alpha > 0) & (y > 0))
    if np.count_nonzero(mask) < 5:
        raise ValueError("Not enough positive samples in the fitting interval")

    p, log_C, _, _, _ = linregress(np.log(alpha[mask]), np.log(y[mask]))
    return np.exp(log_C), p


# Integrate once per eta and use the same trajectory in all numerical checks
def verify_r_ODE(eta):
    predicted = analytical_predictions(eta)
    sol = solve_ivp(
        lambda alpha, X: d_Q_r(alpha, X, eta),
        alpha_span,
        X0,
        t_eval=alpha_eval,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    alpha = sol.t
    Q, r = sol.y
    rho = np.sqrt(Q) * (1 - r)
    s = Q * r

    # First validation: plateau and power-law prefactors from the integrated ODE
    mask = (alpha >= alpha_min_fit) & (alpha <= alpha_max_fit)
    s_plateau = np.mean(s[mask])
    s_plateau_std = np.std(s[mask])
    C_Q_fit, p_Q = power_law_fit(alpha, Q, alpha_min_fit, alpha_max_fit)
    C_r_fit, p_r = power_law_fit(alpha, r, alpha_min_fit, alpha_max_fit)
    s_fit = C_Q_fit * C_r_fit
    K_fit = -(2 / 3) * C_Q_fit**1.5 * s_fit

    print(f"\neta = {eta:g}")
    print(f"  Q*r plateau:     numerical={s_plateau:.9e} (std={s_plateau_std:.3e}), analytical={predicted['s_star']:.9e}")
    print(f"  Q*r fit:         numerical={s_fit:.9e}, analytical={predicted['s_star']:.9e}")
    print(f"  C_Q fit:         numerical={C_Q_fit:.9e}, analytical={predicted['C_Q']:.9e} (p_Q={p_Q:.6f})")
    print(f"  C_r fit:         numerical={C_r_fit:.9e}, analytical={predicted['C_r']:.9e} (p_r={p_r:.6f})")
    print(f"  K from fits:     numerical={K_fit:.9e}, analytical={predicted['K']:.9e}")

    # Second validation: evaluate the bare r5 with s held fixed at s_star
    bare_fixed = np.array([
        Q_value**2.5 * d_Q_r(0.0, (Q_value, predicted["s_star"] / Q_value), eta)[1]
        for Q_value in Q_fixed
    ])

    # Third validation: compare full-trajectory K, beta, and prefactor limits
    dr_dalpha = np.array([
        d_Q_r(a, (q, rr), eta)[1]
        for a, q, rr in zip(alpha, Q, r)
    ])
    k_num = Q**2.5 * dr_dalpha
    beta_num = Q * (s - predicted["s_star"])
    r5_trajectory = k_num - Q * r3(s, eta)
    C_Q_num = Q / alpha**(2 / 3)
    C_rho_num = rho / alpha**(1 / 3)
    C_r_num = r * alpha**(2 / 3)

    print(f"  Fixed-s r5 at Q={Q_fixed[-1]:g}: numerical={bare_fixed[-1]:.9e}, analytical={predicted['r5']:.9e}")
    print(f"  At alpha={alpha[-1]:.2e}:")
    print(f"    s:            numerical={s[-1]:.9e}, analytical={predicted['s_star']:.9e}")
    print(f"    C_Q:          numerical={C_Q_num[-1]:.9e}, analytical={predicted['C_Q']:.9e}")
    print(f"    C_rho:        numerical={C_rho_num[-1]:.9e}, analytical={predicted['C_rho']:.9e}")
    print(f"    C_r:          numerical={C_r_num[-1]:.9e}, analytical={predicted['C_r']:.9e}")
    print(f"    beta:         numerical={beta_num[-1]:.9e}, analytical={predicted['beta']:.9e}")
    print(f"    r5 (traj):    numerical={r5_trajectory[-1]:.9e}, analytical={predicted['r5']:.9e}")
    print(f"    K (traj):     numerical={k_num[-1]:.9e}, analytical={predicted['K']:.9e}")
    print("  Note: r5 differs from K because s evolves along the trajectory.")


for eta in etas:
    verify_r_ODE(eta)
