"""Plot order-parameter dynamics for time-dependent learning-rate schedules.

Integrate the coupled Q and r ODEs for eta(alpha) = pi / (1 + alpha)^gamma,
then plot the generalization error and Q for each selected gamma. The compact
panels retain the separate formatting and reference scalings used in the thesis.

Run from the repository root so relative paths resolve correctly.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Plot formatting used for the thesis figures
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Model parameters and ODE evaluation points
N = 1000
alpha_min = 1e-2
alpha_max = 1e6
n_eval = 4000
alpha_eval = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_eval)
ETA0 = np.pi

# Initialize the teacher vector T and student vector J reproducibly
rng = np.random.default_rng(0)
T = rng.standard_normal(N)
T = (T / np.linalg.norm(T)) * np.sqrt(N)
J = rng.standard_normal(N)
rho0 = np.dot(J, T) / N
Q0 = np.dot(J, J) / N
r0 = 1.0 - rho0 / np.sqrt(Q0)
Y0 = np.array([Q0, r0])

# Learning-rate schedule eta(alpha) for each decay exponent gamma
def eta_schedule(alpha, gamma, eta0=ETA0):
    return eta0 / (1.0 + alpha) ** gamma

# Coupled ODE system for the student self-overlap Q and normalized-overlap deviation r
def d_Q_r(alpha, X, gamma):
    Q, r = X

    eta = eta_schedule(alpha, gamma)

    rho = np.sqrt(Q) * (1 - r)
    sqrt_2Q1 = np.sqrt(2 * Q + 1.0)
    sqrt_term = np.sqrt(max(Q - rho**2 + 1.0, 1e-12))

    denom_sqrt = np.sqrt(max(Q * (1 - Q * r * (r - 2)), 1e-12))
    A = (2 / np.pi) * (r * (r - 2) / denom_sqrt)

    arg1 = Q / (3 * Q + 1)
    denom2 = (3 * Q + 1) * (1 - 2 * Q * r * (r - 2))
    arg2 = rho / np.sqrt(max(denom2, 1e-12))

    arg1 = np.clip(arg1, -1.0, 1.0)
    arg2 = np.clip(arg2, -1.0, 1.0)

    arcsin_combo = 2 * np.arcsin(arg1) - 4 * np.arcsin(arg2) + np.pi

    B = (
        (1 - r)
        / (np.pi**2 * Q * sqrt_2Q1)
        * arcsin_combo
    )

    d_Q = (
        (4 * eta / (np.pi * (Q + 1)))
        * ((rho / sqrt_term) - (Q / sqrt_2Q1))
        + (2 * eta**2 / (np.pi**2 * sqrt_2Q1))
        * arcsin_combo
    )

    d_r = A * eta + B * eta**2

    return [d_Q, d_r]

# Integrate the Q and r trajectories and compute generalization error
def simulate_schedule(gamma):
    sol = solve_ivp(
        fun=lambda alpha, X: d_Q_r(alpha, X, gamma),
        t_span=(alpha_min, alpha_max),
        y0=Y0,
        t_eval=alpha_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed for gamma={gamma}: {sol.message}")

    alpha = sol.t
    Q = sol.y[0]
    r = sol.y[1]

    R = np.clip(1 - r, -1, 1)
    gen_error = np.arccos(R) / np.pi
    eta = eta_schedule(alpha, gamma)

    return alpha, Q, r, eta, gen_error

# Plot configurations for the separate thesis panels
# The first two use the larger layout; the remaining three use compact panels
gamma_cases = [
    {"gamma": 0.0, "filename": "learning_rate_gamma_0.pdf", "figsize": (3.2, 2.6), "linewidth": 1.5},
    {"gamma": 1.5, "filename": "learning_rate_gamma_greater_than_1.pdf", "figsize": (3.2, 2.6), "linewidth": 1.5},
    {"gamma": 0.5, "filename": "learning_rate_gamma_05.pdf", "figsize": (2.05, 2.25), "linewidth": 1.4},
    {"gamma": 0.98, "filename": "learning_rate_gamma_098.pdf", "figsize": (2.05, 2.25), "linewidth": 1.4},
    {"gamma": 1.0, "filename": "learning_rate_gamma_1.pdf", "figsize": (2.05, 2.25), "linewidth": 1.4},
]

# Save the learning-rate schedule figures to the repository
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

for case in gamma_cases:
    gamma = case["gamma"]
    filename = case["filename"]
    compact_panel = case["figsize"] == (2.05, 2.25)

    alpha, Q, r, eta, gen_error = simulate_schedule(gamma)
    alpha_ref = np.logspace(3, 6, 100)

    # Plot generalization error epsilon_g(alpha) and self-overlap Q(alpha)
    fig, ax = plt.subplots(figsize=case["figsize"])
    ax.loglog(alpha, gen_error, label=r"$\epsilon_g(\alpha)$", linewidth=case["linewidth"])
    ax.loglog(alpha, Q, label=r"$Q(\alpha)$", color="#FF9500", linewidth=case["linewidth"])

    # Retain the reference scalings and amplitudes from each original panel
    if compact_panel:
        if gamma == 1.0:
            theory_exp = (2.0 + gamma) / 6.0
            ax.loglog(
                alpha_ref,
                0.2 * alpha_ref**(-theory_exp),
                "--",
                color="grey",
                label=fr"$\alpha^{{-{theory_exp:.3f}}}$",
                linewidth=1.4,
                zorder=20,
            )
        elif 0.0 < gamma < 1.0:
            theory_exp = (2.0 + gamma) / 6.0
            ax.loglog(
                alpha_ref,
                0.2 * alpha_ref**(-theory_exp),
                "--",
                color="grey",
                label=fr"$\alpha^{{-{theory_exp:.3f}}}$",
                linewidth=1.4,
                zorder=20,
            )
    elif gamma == 0.0:
        ax.loglog(
            alpha_ref,
            0.35 * alpha_ref**(-1/3),
            "--",
            color="grey",
            label=r"$\alpha^{-1/3}$",
            linewidth=1.4,
            zorder=20,
        )

    # Axis labels and compact thesis legend
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("")
    ax.set_xlim(1e-1, alpha_max)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(
        frameon=False,
        fontsize=12.0,
        loc="best",
        handlelength=1.1,
        handletextpad=0.35,
        borderpad=0.15,
        labelspacing=0.2,
    )

    plot_path = os.path.join(save_dir, filename)
    fig.savefig(plot_path)
    plt.show()
    plt.close(fig)

    print(f"Saved plot at: {plot_path}")
