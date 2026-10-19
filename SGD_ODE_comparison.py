#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.integrate import solve_ivp

from matplotlib.colors import to_rgb

# Simulation parameters: system size, training duration, and learning rates
N = 1000
alpha_max = 10000
etas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
etas_to_plot = [0.01, 0.05, 0.1, 0.5, 1.0]
steps = int(alpha_max * N)
save_every = int(N / 200)

# Initialize the fixed teacher weight vector with norm sqrt(N)
T = np.random.randn(N)
T = (T / np.linalg.norm(T)) * np.sqrt(N)

# Student activation function and its derivative (error-function learning rule)
g = lambda t: erf(t / np.sqrt(2))
g_prime = lambda t: (2 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * t**2)

# Analytical evolution equations for the teacher-student overlap rho and student self-overlap Q
def d_rho_Q(alpha, X, eta):

    rho, Q = X

    rho2 = rho * rho

    # Auxiliary quantities appearing in the analytical Gaussian averages
    sqrt_term = np.sqrt(max(Q - rho2 + 1.0, 0.0))
    sqrt_2Q1 = np.sqrt(max(2 * Q + 1.0, 1e-12))
    denom2 = 3 * Q + 1
    denom3 = 2 * (Q - rho2) + 1

    arg1 = Q / denom2
    denom_arg2 = np.sqrt(max(denom2 * denom3, 1e-20))
    arg2 = rho / denom_arg2

    # Derivatives d(rho)/d(alpha) and dQ/d(alpha)
    d_rho = (2 * eta / (np.pi * (Q + 1))) * (sqrt_term - rho / sqrt_2Q1)

    d_Q = (
        (4 * eta / (np.pi * (Q + 1))) * ((rho / sqrt_term) - (Q / sqrt_2Q1))
        + (2 * eta**2 / (np.pi**2 * sqrt_2Q1))
        * (2 * np.arcsin(arg1) - 4 * np.arcsin(arg2) + np.pi)
    )

    return [d_rho, d_Q]

results = {}

for eta in etas:
    print(f"Running SGD simulation for η = {eta}")

    # Initialize a new student weight vector for this learning rate
    J = np.random.randn(N)
    rho_vals, Q_vals, R_vals, alpha_vals = [], [], [], []

    for mu in range(steps):
        xi = np.random.randn(N)
        u = np.dot(T, xi) / np.sqrt(N)
        t = np.dot(J, xi) / np.sqrt(N)

        # Teacher label, student output, and online SGD update
        tau = np.sign(u)
        sigma = g(t)
        F = eta * (tau - sigma) * g_prime(t)
        J += (F / np.sqrt(N)) * xi

        # Record the order parameters as a function of alpha = mu / N
        if mu % save_every == 0:
            rho = np.dot(J, T) / N
            Q = np.dot(J, J) / N
            R = rho / np.sqrt(Q)
            rho_vals.append(rho)
            Q_vals.append(Q)
            R_vals.append(R)
            alpha_vals.append(mu / N)

    # Integrate the analytical order-parameter ODEs on the SGD sampling grid
    y0 = [rho_vals[0], Q_vals[0]]
    sol = solve_ivp(
        lambda a, y: d_rho_Q(a, y, eta),
        (0, alpha_max),
        y0,
        t_eval=np.array(alpha_vals),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    # Compute generalization errors from the normalized teacher-student overlap
    gen_error_sgd = (1 / np.pi) * np.arccos(R_vals)
    R_theory = sol.y[0] / np.sqrt(sol.y[1])
    gen_error_theory = (1 / np.pi) * np.arccos(R_theory)

    # Store the SGD trajectories and corresponding analytical ODE solution
    results[eta] = {
        "alpha": np.array(alpha_vals),
        "rho_sgd": np.array(rho_vals),
        "Q_sgd": np.array(Q_vals),
        "err_sgd": np.array(gen_error_sgd),
        "rho_th": sol.y[0],
        "Q_th": sol.y[1],
        "err_th": np.array(gen_error_theory),
    }


# Save the SGD and ODE trajectories for use by the order-parameter plotting script
import os
import pickle

os.makedirs("results", exist_ok=True)

with open("results/SGD_ODE_comparison.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved results to results/SGD_ODE_comparison.pkl")


# Plot the SGD and ODE generalization-error trajectories
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Darken the SGD curves to distinguish them from the corresponding ODE curves
def darken_color(color, factor=0.75):
    r, g, b = to_rgb(color)
    return (factor * r, factor * g, factor * b)

fig, ax = plt.subplots(figsize=(5.2, 3.8))

for eta in etas_to_plot:
    alpha = results[eta]["alpha"]
    gen_sgd = results[eta]["err_sgd"]
    gen_ode = results[eta]["err_th"]

    mask = alpha > 0
    alpha = alpha[mask]
    gen_sgd = gen_sgd[mask]
    gen_ode = gen_ode[mask]

    ode_line, = ax.loglog(
        alpha,
        gen_ode,
        label=rf"$\eta={eta}$",
        linewidth=1,
    )

    ax.loglog(
        alpha,
        gen_sgd,
        linestyle="--",
        color=darken_color(ode_line.get_color(), factor=0.75),
        linewidth=1.5,
    )

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\epsilon_g(\alpha)$")

# Reference power-law slope alpha^(-1/3)
alpha_ref1 = np.logspace(1.5, 4, 100)

# Anchor the reference line to the ODE curve for the selected learning rate
alpha_anchor = 1e1
eta_anchor = 0.1 if 0.1 in etas_to_plot else etas_to_plot[0]

alpha_data = np.asarray(results[eta_anchor]["alpha"])
gen_data = np.asarray(results[eta_anchor]["err_th"])

mask = (alpha_data > 0) & (gen_data > 0)
alpha_data = alpha_data[mask]
gen_data = gen_data[mask]

idx_anchor = np.argmin(np.abs(alpha_data - alpha_anchor))
y_anchor = gen_data[idx_anchor]

C_ref1 = y_anchor * alpha_data[idx_anchor]**(1/3)

ax.loglog(
    alpha_ref1,
    0.3 * C_ref1 * alpha_ref1**(-1/3),
    linestyle="--",
    color="grey",
    linewidth=1.4,
    label=r"$\alpha^{-1/3}$",
    zorder=20,
)

ax.set_xlim(2e-1, 1e4)

ax.grid(False)

ax.legend(
    frameon=False,
    loc="best",
    ncol=1,
)

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/SGD_ODE_generalization_error.pdf")

plt.close(fig)

