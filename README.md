# On-line Learning Dynamics and Neural Scaling Laws for a perceptron Classification Problem

Numerical simulations and analytical validation accompanying the master's thesis on **On-line Learning Dynamics and Neural Scaling Laws for a perceptron Classification Problem**. The main model has a **sign-activation teacher** and a **smooth error-function (erf) student** trained using stochastic gradient descent (SGD) with mean-squared error (MSE). The repository also includes a separate sigmoid-student experiment trained with binary cross-entropy (BCE).

The objective is to connect microscopic stochastic weight updates with macroscopic order-parameter ordinary differential equations (ODEs), study the asymptotic generalisation error, and investigate whether time-dependent learning rates accelerate learning.

The analysis is based on the statistical-mechanics framework of neural learning, particularly the online perceptron dynamics developed in the foundational literature (Watkin et al., 1993; Biehl & Riegler, 1994; Biehl & Schwarze, 1995).

## Main findings

- **Fixed learning rate (sign–erf, MSE):** Analytical asymptotic analysis gives a generalisation-error decay of $\epsilon_g(\alpha)\sim\alpha^{-1/3}$. Numerical ODE solutions and finite-dimensional online SGD trajectories support this law.
- **Microscopic versus macroscopic dynamics:** Integrated order-parameter ODEs agree closely with SGD trajectories over the investigated training interval.
- **Binary cross-entropy comparison:** The sigmoid-student/BCE simulations show behaviour consistent with the same one-third decay. This is **numerical evidence**, not a closed-form ODE derivation for the BCE model.
- **Decaying learning rates:** For $\eta(\alpha)=\eta_0/(1+\alpha)^\gamma$ and $0<\gamma<1$, the quasi-stationary analysis predicts $\epsilon_g(\alpha)\sim\alpha^{-(2+\gamma)/6}$, approaching an exponent of $1/2$ as $\gamma\to1^-$. The case $\gamma=1$ is **marginal**, where the quasi-stationary assumption is not asymptotically guaranteed; for $\gamma>1$, the numerical ODE dynamics saturate.

These claims refer to the assumptions and asymptotic regimes analyzed in the thesis, not to general neural networks without qualification.

## Model and observables

Independent input vectors $\boldsymbol\xi\in\mathbb R^N$ are drawn from $\mathcal N(0,I)$. The fixed teacher has weights $\mathbf T$, normalized so that $\lVert\mathbf{T}\rVert^2=N$, and the evolving student has weights $\mathbf J$. Their local fields are

$$
u=\frac{\mathbf T\cdot\boldsymbol\xi}{\sqrt N},\qquad t=\frac{\mathbf J\cdot\boldsymbol\xi}{\sqrt N}.
$$

The teacher label is $\tau=\mathrm{sgn}(u)$ and the principal student output is $g(t)=\mathrm{erf}(t/\sqrt 2)$. With per-example squared loss $V=\tfrac12[\tau-g(t)]^2$, the online SGD update is

$$
\mathbf J^{\mu+1}=\mathbf J^\mu+\frac{F(u,t)}{\sqrt N}\boldsymbol\xi^\mu,\qquad
F(u,t)=\eta[\mathrm{sgn}(u)-g(t)]g'(t).
$$

The order parameters and rescaled learning time are

$$
\rho=\frac{\mathbf J\cdot\mathbf T}{N},\qquad
Q=\frac{\|\mathbf J\|^2}{N},\qquad
R=\frac{\rho}{\sqrt Q},\qquad
r=1-R,\qquad
\alpha=\frac{\mu}{N}.
$$

For isotropic Gaussian inputs, the classification generalisation error is

$$
\epsilon_g(\alpha)=\frac{1}{\pi}\arccos[R(\alpha)].
$$

In the thermodynamic limit, the macroscopic evolution is described by

$$
\frac{d\rho}{d\alpha}=\langle Fu\rangle,\qquad
\frac{dQ}{d\alpha}=\langle F^2+2Ft\rangle,
$$

with expectations over the jointly Gaussian teacher and student fields. The thesis derives closed expressions for the sign–erf/MSE model and analyzes the late-time scaling variable $s=Qr$.

## Files

| Script | Purpose |
| --- | --- |
| `SGD_ODE_comparison.py` | Simulate online sign–erf/MSE SGD, numerically integrate the analytical $\rho,Q$ ODEs, compare generalisation-error trajectories, and save `results/SGD_ODE_comparison.pkl`. |
| `numerical_comparison.py` | Check the analytical Gaussian-field averages $\langle Fu\rangle$, $\langle Ft\rangle$, and $\langle F^2\rangle$ against Monte Carlo estimates. |
| `order_parameter_plots.py` | Load the saved SGD/ODE comparison and plot theoretical $\rho$, $Q$, and $r$ trajectories with late-time power-law fits. |
| `verify_r_ODE.py` | Check late-time asymptotics of the coupled $Q,r$ ODEs, including scaling prefactors and the $Qr$ plateau. This runs ODE checks, not microscopic SGD. |
| `learning_rate_schedule.py` | Integrate the $Q,r$ ODEs for power-law-decaying learning rates and plot $Q$ and generalisation error for different $\gamma$. |
| `SGD_cross_entropy.py` | Run online SGD for a sign teacher and sigmoid student trained with BCE, saving `results/SGD_cross_entropy.pkl` and a figure. |
| `plot_SGD_ODE_from_pickle.py` | Recreate the SGD-versus-ODE generalisation-error comparison from saved pickle data, without rerunning SGD. |
| `plot_SGD_cross_entropy_from_pickle.py` | Recreate the BCE generalisation-error plot from saved pickle data, without rerunning SGD. |


## Installation

Python 3 with NumPy, SciPy, and Matplotlib is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

Run commands **from the repository root**, because the scripts use relative paths such as `results/`, `figures/`, and `styles/`.


## Reproducing the experiments

Create the output directories:

```bash
mkdir -p results figures
```

### 1. Sign–erf SGD versus ODE

```bash
python 'SGD_ODE_comparison.py'
```

This generates `results/SGD_ODE_comparison.pkl`. 

**Compute warning:** The microscopic SGD scripts use $N=1000$ and $\alpha_{\max}=10^4$, corresponding to **10 million sequential weight updates per learning rate**. Running every configuration can be time-consuming. Reduce $N$, `alpha_max`, or the learning-rate list for a quick demonstration; such shorter runs do not necessarily reach the asymptotic regime.

Recreate the saved-data figures with:

```bash
python 'plot_SGD_ODE_from_pickle.py'
python 'order_parameter_plots.py'
```

### 2. Check Gaussian averages and asymptotic ODE predictions

```bash
python 'numerical_comparison.py'
python 'verify_r_ODE.py'
```

The numerical Gaussian comparison uses a large Monte Carlo sample and can require substantial memory. Adjust `n_samples` in the script if needed.

### 3. Learning-rate schedules

```bash
python 'learning_rate_schedule.py'
```

The implemented regularized schedule is $\eta(\alpha)=\pi/(1+\alpha)^\gamma$. The script examines $\gamma=0,0.5,0.98,1,1.5$ and writes figures into `figures/`.

### 4. Binary cross-entropy comparison

```bash
python 'SGD_cross_entropy.py'
python 'plot_SGD_cross_entropy_from_pickle.py'
```

The first command generates `results/SGD_cross_entropy.pkl`; the second recreates its saved-data plot. Both require the custom stylesheet described above unless adapted.

**Compute warning:** The microscopic SGD scripts use $N=1000$ and $\alpha_{\max}=10^4$, corresponding to **10 million sequential weight updates per learning rate**. Running every configuration can be time consuming. Reduce $N$, `alpha_max`, or the learning-rate list for a quick demonstration; such shorter runs do not necessarily reach the asymptotic regime.

## Outputs and reproducibility notes

- `results/*.pkl` contains saved NumPy trajectories used by the plotting scripts. 
- `figures/` contains PDF figures written by the plotting scripts.
- The analytical ODEs describe the thermodynamic-limit macroscopic dynamics; the SGD simulations use finite $N$. Agreement should be interpreted within the simulated parameter ranges.
- The precomputed .pkl files are not included in this repository due to large file sizes. However, all scripts required to regenerate this data are provided. The simulation scripts save their outputs in the results/ directory. Once generated, these files can be used by the corresponding plotting scripts to reproduce the figures without rerunning the simulations.

## Academic context

This repository contains computational experiments associated with a master's thesis on **On-line Learning Dynamics and Neural Scaling Laws for a perceptron Classification Problem**, completed in Leipzig University in 2026. 

The thesis investigates the learning dynamics of a mismatched student–teacher perceptron utlising methods from statistical physics and deep learning. Particular emphasis is placed on the asymptotic behaviour of the generalisation error and the influence of time-dependent learning rates. 

The theoretical framework, derivations, asymptotic analysis, and limitations are developed in the thesis, particularly Chapters 3–5 and the appendices.

## Acknowledgements

I would like to thank Prof. Dr. Bernd Rosenow for supervising this thesis and Marcel Kühn for his guidance, valuable discussions, and support throughout the project.


## References

1. Engel, A., & Van den Broeck, C. (2001).
   *Statistical Mechanics of Learning*. Cambridge University Press.

2. Watkin, T. L. H., Rau, A., & Biehl, M. (1993).
   "The statistical mechanics of learning a rule."
   *Reviews of Modern Physics, 65*, 499–556.
   [DOI](https://doi.org/10.1103/RevModPhys.65.499)

3. Biehl, M., & Riegler, P. (1994).
   "On-Line Learning with a Perceptron."
   *Europhysics Letters, 28*(7), 525–530.
   [DOI](https://doi.org/10.1209/0295-5075/28/7/012)

4. Biehl, M., & Schwarze, H. (1995).
   "Learning by on-line gradient descent."
   *Journal of Physics A: Mathematical and General, 28*(3), 643–656.

5. Scarpetta, S., & Saad, D. (1999).
   "On-line learning of unrealizable tasks."
   *Physical Review E, 60*(5), 5902–5911.

6. Kühn, M., Thelge, Y., & Rosenow, B. (2026).
   "A Boundary-Layer Mechanism for One-Third Scaling
   in Online Softmax Classification."
   [arXiv:2605.22341](https://arxiv.org/abs/2605.22341)
