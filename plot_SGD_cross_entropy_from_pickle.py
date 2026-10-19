
"""Plot online SGD generalization error for binary cross-entropy learning.

Load previously simulated student–teacher perceptron trajectories from
results/SGD_cross_entropy.pkl and plot the generalization error against
rescaled time alpha = mu / N for different learning rates.

Include the reference asymptotic scaling alpha^(-1/3).

Run from the repository root so relative paths resolve correctly.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths and plotting configuration
# ---------------------------------------------------------------------

data_file = "results/SGD_cross_entropy.pkl"
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

# Apply the thesis Matplotlib stylesheet
plt.style.use("styles/plt_style.pstyle")

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
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------
# Load saved SGD trajectories
# ---------------------------------------------------------------------

with open(data_file, "rb") as f:
    results = pickle.load(f)

print("Available learning rates:", sorted(results.keys()))

etas = [0.01, 0.05, 0.1, 0.5, 1.0]
etas_to_plot = [eta for eta in etas if eta in results]

if not etas_to_plot:
    raise ValueError("No requested learning rates found in the pickle file.")

print("Plotting learning rates:", etas_to_plot)


# ---------------------------------------------------------------------
# Plot SGD generalization error
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.2, 3.8))

for eta in etas_to_plot:
    alpha = np.asarray(results[eta]["alpha"])
    err_sgd = np.asarray(results[eta]["err_sgd"])

    # Keep only finite, positive values for logarithmic axes
    mask = (
        np.isfinite(alpha)
        & np.isfinite(err_sgd)
        & (alpha > 0)
        & (err_sgd > 0)
    )

    ax.loglog(
        alpha[mask],
        err_sgd[mask],
        label=rf"$\eta={eta}$",
        linewidth=1.5,
    )


# ---------------------------------------------------------------------
# Reference asymptotic scaling: alpha^(-1/3)
# ---------------------------------------------------------------------

alpha_ref = np.logspace(1.5, 4, 100)

# Anchor the reference amplitude to the eta = 0.1 trajectory
alpha_anchor = 1e1
eta_anchor = 0.1 if 0.1 in etas_to_plot else etas_to_plot[0]

alpha_data = np.asarray(results[eta_anchor]["alpha"])
err_data = np.asarray(results[eta_anchor]["err_sgd"])

mask = (
    np.isfinite(alpha_data)
    & np.isfinite(err_data)
    & (alpha_data > 0)
    & (err_data > 0)
)

alpha_data = alpha_data[mask]
err_data = err_data[mask]

if alpha_data.size > 0:
    idx_anchor = np.argmin(np.abs(alpha_data - alpha_anchor))

    # Reference amplitude chosen relative to the selected trajectory
    C_ref = err_data[idx_anchor] * alpha_data[idx_anchor]**(1 / 3)

    ax.loglog(
        alpha_ref,
        0.25 * C_ref * alpha_ref**(-1 / 3),
        linestyle="--",
        color="grey",
        linewidth=1.4,
        label=r"$\alpha^{-1/3}$",
        zorder=20,
    )


# ---------------------------------------------------------------------
# Axes, legend, and figure export
# ---------------------------------------------------------------------

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\epsilon_g(\alpha)$")

ax.set_xlim(2e-1, 1e4)

# Optional y-axis limits
# ax.set_ylim(1e-3, 1)

ax.grid(False)
ax.legend(
    frameon=False,
    loc="best",
    ncol=1,
)

fig.tight_layout()

plot_path = os.path.join(
    save_dir,
    "cross_entropy_SGD_generalization_error_loglog.pdf",
)

fig.savefig(plot_path)

plt.show()
plt.close(fig)

print(f"Saved figure to: {plot_path}")
