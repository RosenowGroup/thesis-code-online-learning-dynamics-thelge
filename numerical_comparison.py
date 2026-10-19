"""Compare Monte Carlo estimates with analytical Gaussian integrals.

The jointly Gaussian fields u (teacher) and t (student) are used to evaluate
< F u >, < F t >, and < F^2 >, including their separate integral contributions.
These averages enter the student-teacher overlap and student self-overlap
order-parameter dynamics.

The teacher field u and student field t have covariance
    Cov(u, t) = [[1, R sqrt(Q)], [R sqrt(Q), Q]].

With F = eta [sign(u) - erf(t / sqrt(2))] sqrt(2/pi) exp(-t**2 / 2),
we compare E[F u], E[F t], E[F**2] and their constituent integrals
against closed-form results.
"""

import numpy as np
from scipy.special import erfc

# Model parameters and Monte Carlo sample size
eta = 0.1
Q = 10
R = 0.9999
n_samples = 10**7
np.random.seed(42)

#Auxiliary quantity used in the analytical integrals
A = (Q * (1 - (R ** 2))) + 1

# Draw correlated Gaussian teacher and student fields (u, t)
mean = [0, 0]
cov = [[1, R*np.sqrt(Q)],
       [R*np.sqrt(Q), Q]]
samples = np.random.multivariate_normal(mean, cov, n_samples)
u, t = samples[:, 0], samples[:, 1]

# Gaussian tail function H, student activation g, and its derivative
def H(x):
    return 0.5 * erfc(x / np.sqrt(2))

def g(t_val):
    return 1 - 2 * H(t_val)

def g_prime(t_val):
    return 2 * np.exp(-0.5 * ( t_val) ** 2) / np.sqrt(2 * np.pi)

# Learning amplitude F for each sampled pair of fields
sgn_u = np.sign(u)
g_t = g(t)
g_t_prime = g_prime(t)
F = eta * (sgn_u - g_t) * g_t_prime

# Compute <Fu>
# Direct Monte Carlo estimate of <Fu> and its standard error
Fu_vals = F * u
Fu_numerical = np.mean(Fu_vals)
Fu_std_err = np.std(Fu_vals) / np.sqrt(n_samples)

# Individual contributions to <Fu> after expanding the learning amplitude
exp1_vals = u * sgn_u * np.exp(-0.5 * t ** 2)
exp2_vals = u * H(t) * np.exp(-0.5 * t ** 2)
exp3_vals = u * np.exp(-0.5 * t ** 2)

exp1 = np.mean(exp1_vals)
exp2 = 2 * np.mean(exp2_vals)
exp3 = np.mean(exp3_vals)

exp1_err = np.std(exp1_vals) / np.sqrt(n_samples)
exp2_err = np.std(exp2_vals) / np.sqrt(n_samples)
exp3_err = np.std(exp3_vals) / np.sqrt(n_samples)

# Reconstruct <Fu> numerically from its individual contributions
Fu_numerical_alt = (2 * eta / np.sqrt(2 * np.pi)) * (exp1 + exp2 - exp3)

# Closed-form analytical expressions for the same contributions
exp1_a = (2 / np.sqrt(2 * np.pi)) * np.sqrt(A) / (Q + 1)
exp2_a = -2* (R * np.sqrt(Q)) / (np.sqrt(2 * np.pi * (2 * Q + 1)) * (Q + 1))
exp3_a = 0

Fu_analytical = (2 * eta / np.sqrt(2 * np.pi)) * (exp1_a + exp2_a - exp3_a)

# Compare direct numerical, decomposed numerical, and analytical <Fu>
print("Numerical vs Analytical Comparison of <Fu> with Error Bars")
print("====================================================")
print(f"<Fu> Numerical      : {Fu_numerical:.6f} ± {Fu_std_err:.2e}")
print(f"<Fu> Alt Numerical  : {Fu_numerical_alt:.6f}")
print(f"<Fu> Analytical     : {Fu_analytical:.6f}")
print()
print(f"exp1 Numerical  : {exp1:.6f} ± {exp1_err:.6f}     exp1 Analytical : {exp1_a:.6f}")
print(f"exp2 Numerical  : {exp2:.6f} ± {exp2_err:.6f}     exp2 Analytical : {exp2_a:.6f}")
print(f"exp3 Numerical  : {exp3:.6f} ± {exp3_err:.6f}     exp3 Analytical : {exp3_a:.6f}")
print()
print(Fu_analytical)

# Compute <Ft>
# Direct Monte Carlo estimate of <Ft> and its standard error
Ft_vals = F * t
Ft_numerical = np.mean(Ft_vals)
Ft_std_err = np.std(Ft_vals) / np.sqrt(n_samples)

# Individual contributions to <Ft> after expanding the learning amplitude
exp4_vals = t * sgn_u * np.exp(-0.5 * t ** 2)
exp5_vals = t * H(t) * np.exp(-0.5 * t ** 2)
exp6_vals = t * np.exp(-0.5 * t ** 2)

exp4 = np.mean(exp4_vals)
exp5 = 2 * np.mean(exp5_vals)
exp6 = np.mean(exp6_vals)

exp4_err = np.std(exp4_vals) / np.sqrt(n_samples)
exp5_err = np.std(exp5_vals) / np.sqrt(n_samples)
exp6_err = np.std(exp6_vals) / np.sqrt(n_samples)

# Reconstruct <Ft> numerically from its individual contributions
Ft_numerical_alt = (2 * eta / np.sqrt(2 * np.pi)) * (exp4 + exp5 - exp6)

# Closed-form analytical expressions for the same contributions
exp4_a = (2 * R * np.sqrt(Q)) / ( (Q + 1) * (np.sqrt(2 * np.pi * A)))
exp5_a = -2 * Q / ((Q + 1) * np.sqrt(2 * np.pi * (2 * Q + 1)))
exp6_a = 0

Ft_analytical = (2 * eta / np.sqrt(2 * np.pi)) * (exp4_a + exp5_a - exp6_a)

# Compare direct numerical, decomposed numerical, and analytical <Ft>
print("Numerical vs Analytical Comparison of <Ft> with Error Bars")
print("====================================================")
print(f"<Ft> Numerical      : {Ft_numerical:.6f} ± {Ft_std_err:.2e}")
print(f"<Ft> Alt Numerical  : {Ft_numerical_alt:.6f}")
print(f"<Ft> Analytical     : {Ft_analytical:.6f}")
print()
print(f"exp4 Numerical  : {exp4:.6f} ± {exp4_err:.6f}     exp4 Analytical : {exp4_a:.6f}")
print(f"exp5 Numerical  : {exp5:.6f} ± {exp5_err:.6f}     exp5 Analytical : {exp5_a:.6f}")
print(f"exp6 Numerical  : {exp6:.6f} ± {exp6_err:.6f}     exp6 Analytical : {exp6_a:.6f}")

#Compute <F**2>
# Compare direct numerical, decomposed numerical, and analytical <Ft>
Ff_vals = F * F
Ff_numerical = np.mean(Ff_vals)
Ff_std_err = np.std(Ff_vals) / np.sqrt(n_samples)

# Individual contributions to <F^2> after squaring the learning amplitude
exp7_vals = sgn_u ** 2 * np.exp(-t ** 2)
exp8_vals = sgn_u * np.exp(-t ** 2)
exp9_vals = sgn_u * H(t) * np.exp(-t ** 2)
exp10_vals = np.exp(-t ** 2)
exp11_vals = H(t) * np.exp(-t ** 2)
exp12_vals = (H(t) ** 2) * np.exp(-t ** 2)

exp7 = np.mean(exp7_vals)
exp8 = 2 * np.mean(exp8_vals)
exp9 = 4 * np.mean(exp9_vals)
exp10 = np.mean(exp10_vals)
exp11 = 4 * np.mean(exp11_vals)
exp12 = 4 * np.mean(exp12_vals)

exp7_err = np.std(exp7_vals) / np.sqrt(n_samples)
exp8_err = np.std(exp8_vals) / np.sqrt(n_samples)
exp9_err = np.std(exp9_vals) / np.sqrt(n_samples)
exp10_err = np.std(exp10_vals) / np.sqrt(n_samples)
exp11_err = np.std(exp11_vals) / np.sqrt(n_samples)
exp12_err = np.std(exp12_vals) / np.sqrt(n_samples)

# Reconstruct <F^2> numerically from its six contributions
Ff_numerical_alt = ((4 * (eta ** 2) )/ (2 * np.pi)) * (exp7 - exp8 + exp9 + exp10 - exp11 + exp12)

# Auxiliary factor used in the analytical <F^2> integrals
B = 1 / np.sqrt((2 * Q) + 1)

# Closed-form analytical expressions for the six contributions
exp7_a = B
exp8_a = 2 * 0

# Check the denominator and arcsine argument in the exp9 expression
inner = ((3 * Q) + 1) * ((2 * Q * (1 - (R ** 2))) + 1)
if inner <= 0:
    print("Invalid or numerically unstable denominator:", inner)
else:
    den = np.sqrt(inner)
    ratio = R * np.sqrt(Q) / den
    if ratio < 0 or ratio > 1:
        print("Invalid ratio for arcsin:", ratio)
    else:
        exp9_a = 4 * (-1 / (np.pi)) * B * np.arcsin(ratio)

exp10_a = 1 / np.sqrt(2 * Q + 1)
exp11_a = 4 * 1 / (2 * np.sqrt(2 * Q + 1))

tau = Q / ((3 * Q) + 1)
exp12_a = 4 * B * ((0.25 + (1 / (2 * np.pi)) * np.arcsin(tau)))

Ff_analytical = (4 * (eta ** 2) / (2 * np.pi)) * (exp7_a - exp8_a + exp9_a + exp10_a - exp11_a + exp12_a)


# Compare direct numerical, decomposed numerical, and analytical <F^2>
print("Numerical vs Analytical Comparison of <Ff> with Error Bars")
print("====================================================")
print(f"<Ff> Numerical      : {Ff_numerical:.6f} ± {Ff_std_err:.2e}")
print(f"<Ff> Alt Numerical  : {Ff_numerical_alt:.6f}")
print(f"<Ff> Analytical     : {Ff_analytical:.6f}")
print()
print(f"exp7 Numerical  : {exp7:.6f} ± {exp7_err:.6f}     exp7 Analytical : {exp7_a:.6f}")
print(f"exp8 Numerical  : {exp8:.6f} ± {exp8_err:.6f}     exp8 Analytical : {exp8_a:.6f}")
print()
print(f"exp9 Numerical  : {exp9:.6f} ± {exp9_err:.6f}     exp9 Analytical : {exp9_a:.6f}")
print(f"exp10 Numerical  : {exp10:.6f} ± {exp10_err:.6f}     exp10 Analytical : {exp10_a:.6f}")
print()
print(f"exp11 Numerical  : {exp11:.6f} ± {exp11_err:.6f}     exp11 Analytical : {exp11_a:.6f}")
print(f"exp12 Numerical  : {exp12:.6f} ± {exp12_err:.6f}     exp12 Analytical : {exp12_a:.6f}")
print()
print(2*Ft_analytical + Ff_analytical)

# Assemble numerical and analytical order-parameter evolution terms:
# <Fu> for the student-teacher overlap and 2<Ft> + <F^2> for self-overlap
drho_term_n = Fu_numerical
dQ_term_n = 2*Ft_numerical + Ff_numerical

drho_term_a = Fu_analytical
dQ_term_a = 2*Ft_analytical + Ff_analytical

# Print numerical and analytical results for the two evolution terms
print("Numerical and analytical results")
print(drho_term_n)
print(dQ_term_n)
print()
print(drho_term_a)
print(dQ_term_a)
