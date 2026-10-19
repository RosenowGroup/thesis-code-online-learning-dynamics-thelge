"""Simulate online SGD learning in a student–teacher perceptron.

For binary cross-entropy loss with a sigmoid student output, record the
student–teacher overlap rho, student self-overlap Q, normalized overlap R,
and generalization error as functions of rescaled time alpha = mu / N.
The corresponding Gaussian-field expectations <Fu>, <Ft>, and <F^2>
are examined in the separate numerical comparison code.

Run from the repository root so relative paths resolve correctly.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Output directory for SGD trajectories (relative to the repository root)
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving results to: {save_dir}")

# Model parameters, learning rates, and rescaled simulation time
N = 1000
alpha_max = 10000
etas = [0.01, 0.05, 0.1, 0.5, 1.0]
steps = int(alpha_max * N)
save_every = int(N / 200)

# Initialize the teacher vector T with norm sqrt(N)
T = np.random.randn(N)
T = (T / np.linalg.norm(T)) * np.sqrt(N)

# Sigmoid student activation and its derivative (binary cross-entropy setting)
sigmoid = lambda t: 1 / (1 + np.exp(-t))
sigmoid_prime = lambda t: sigmoid(t) * (1 - sigmoid(t))

results = {}
# Common initial student vector J for comparison across learning rates
J_start = np.random.randn(N)

for eta in etas:
    print(f"Running SGD simulation for η = {eta}")

    # Reset student vector and initialize order-parameter trajectories
    J = np.copy(J_start)
    rho_vals, Q_vals, R_vals, alpha_vals = [], [], [], []

    for mu in range(steps):
        # Draw a Gaussian input xi and compute teacher field u and student field t
        xi = np.random.randn(N)
        u = np.dot(T, xi) / np.sqrt(N)
        t = np.dot(J, xi) / np.sqrt(N)

        # Teacher label sign(u), mapped from {-1, +1} to {0, 1}
        tau = np.sign(u)
        y = (tau + 1) / 2

        # Student sigmoid output for the local field t
        sigma = sigmoid(t)

        # Learning amplitude F and online SGD update of the student weights
        F = eta * (y - sigma)
        J += (F / np.sqrt(N)) * xi

        if mu % save_every == 0:
            # Order parameters: student–teacher overlap rho, student self-overlap Q, normalized overlap R
            rho = np.dot(J, T) / N
            Q = np.dot(J, J) / N
            R = rho / np.sqrt(Q)
            rho_vals.append(rho)
            Q_vals.append(Q)
            R_vals.append(R)
            alpha_vals.append(mu / N)

    # Generalization error epsilon_g = arccos(R) / pi for a sign teacher
    gen_error_sgd = (1 / np.pi) * np.arccos(R_vals)

    # Store rescaled time, overlaps, and generalization error for this learning rate
    results[eta] = {
        "alpha": np.array(alpha_vals),
        "rho_sgd": np.array(rho_vals),
        "Q_sgd": np.array(Q_vals),
        "err_sgd": np.array(gen_error_sgd),
    }

# Save SGD trajectories for subsequent plotting and comparison
data_file = os.path.join(save_dir, "SGD_cross_entropy.pkl")
with open(data_file, "wb") as f:
    pickle.dump(results, f)
print(f"\nAll results saved to: {data_file}")

# Reload the saved SGD trajectories for plotting
with open(data_file, "rb") as f:
    results = pickle.load(f)

print("Available η in loaded results:", sorted(results.keys()))

# Select the learning rates available in the saved results
etas = [0.01, 0.05, 0.1, 0.5, 1.0]
etas_to_plot = [eta for eta in etas if eta in results]
print("Plotting only these η values:", etas_to_plot)

# Apply the thesis Matplotlib stylesheet from the repository styles directory
plt.style.use("styles/plt_style.pstyle")

# Plot formatting used for the thesis figures
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

# Plot the generalization error epsilon_g(alpha) for each learning rate
fig, ax = plt.subplots(figsize=(5.2, 3.8))

for eta in etas_to_plot:
    alpha = np.asarray(results[eta]["alpha"])
    err_sgd = np.asarray(results[eta]["err_sgd"])

    # Retain positive alpha and epsilon_g values for logarithmic axes
    mask = (alpha > 0) & (err_sgd > 0)
    alpha = alpha[mask]
    err_sgd = err_sgd[mask]

    ax.loglog(
        alpha,
        err_sgd,
        label=rf"$\eta={eta}$",
        linewidth=1.5,
    )

# Label the rescaled time alpha and generalization error epsilon_g
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\epsilon_g(\alpha)$")

# Reference asymptotic power law alpha^(-1/3)
alpha_ref1 = np.logspace(1.5, 4, 100)

# Set the reference-line amplitude using an SGD trajectory
alpha_anchor = 1e1
eta_anchor = 0.1 if 0.1 in etas_to_plot else etas_to_plot[0]

alpha_data = np.asarray(results[eta_anchor]["alpha"])
err_data = np.asarray(results[eta_anchor]["err_sgd"])

mask = (alpha_data > 0) & (err_data > 0)
alpha_data = alpha_data[mask]
err_data = err_data[mask]

idx_anchor = np.argmin(np.abs(alpha_data - alpha_anchor))
y_anchor = err_data[idx_anchor]

C_ref1 = y_anchor * alpha_data[idx_anchor]**(1/3)

ax.loglog(
    alpha_ref1,
    0.25 * C_ref1 * alpha_ref1**(-1/3),
    linestyle="--",
    color="grey",
    linewidth=1.4,
    label=r"$\alpha^{-1/3}$",
    zorder=20,
)

# Plot limits and legend
ax.set_xlim(2e-1, 1e4)

# Optional manual limits for the generalization-error axis
# ax.set_ylim(1e-3, 1)

ax.grid(False)
ax.legend(
    frameon=False,
    loc="best",
    ncol=1,
)

fig.tight_layout()

# Save the SGD generalization-error figure to the repository
os.makedirs("figures", exist_ok=True)
plot_path = "figures/cross_entropy_SGD_generalization_error_loglog.pdf"
fig.savefig(plot_path)

plt.show()
plt.close(fig)

print(f"Saved log-log SGD generalization error plot at: {plot_path}")
