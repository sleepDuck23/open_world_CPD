import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from functions_nougat import comp_H_h, comp_Gamma

# ---------------------------------------------------------
# 1. Parameter Setup
# ---------------------------------------------------------
nu = 1e-3
mu_initial = 5e-4 # Used in original script before loop
n_ref = 250
n_test = 250
L = 16
gamma = 0.25

mean_sig = np.zeros(2)
cov_sig = 0.5 * np.array([[1.0, 0.25], 
                          [0.25, 1.0]])

# Sample dictionary: (dimensions=2, samples=L)
dict_x = np.random.multivariate_normal(mean_sig, cov_sig, size=L).T

# ---------------------------------------------------------
# 2. Compute Base Matrices
# ---------------------------------------------------------
H, h = comp_H_h(dict_x, mean_sig, cov_sig, gamma)
Gamma = comp_Gamma(dict_x, mean_sig, cov_sig, gamma)

# mu_pl = np.linspace(start, stop, num_points)
mu_pl = np.linspace(1e-4, 1.0, 256)
var_inf = []
var_inf_approx = []

# Useful identity matrices
I_L = np.eye(L)
I_L2 = np.eye(L**2)

# ---------------------------------------------------------
# 3. Main Loop
# ---------------------------------------------------------
for mu in mu_pl:
    # Q = (H - h*h') * (n_ref + n_test) / (n_ref*n_test)
    Q = (H - np.outer(h, h)) * (n_ref + n_test) / (n_ref * n_test)

    # S = (μ^2/n_ref)*(Γ + (n_ref - 1) * kron(H, H)) + (1-μ*ν)^2*I 
    S = (mu**2 / n_ref) * (Gamma + (n_ref - 1) * np.kron(H, H)) + ((1 - mu * nu)**2) * I_L2
    
    # H_plus_H = kron(H, eye(L)) + kron(eye(L), H)
    H_plus_H = np.kron(H, I_L) + np.kron(I_L, H)
    
    # S -= μ*(1-μ*ν)*H_plus_H
    S -= mu * (1 - mu * nu) * H_plus_H

    # c_∞ = μ^2 * ((I - S) \ vec(Q))
    # Python Note: order='F' ensures column-major flattening to match Julia's vec()
    vec_Q = Q.flatten(order='F')
    c_inf = (mu**2) * solve(I_L2 - S, vec_Q)
    
    # tr(H * reshape(c_∞, L, L)) / n_test
    # Python Note: order='F' ensures column-major reshaping to match Julia's reshape()
    c_inf_matrix = c_inf.reshape((L, L), order='F')
    var_inf.append(np.trace(H @ c_inf_matrix) / n_test)

    # dot(vec(H), (2*ν*I + H_plus_H) \ vec(Q)) * μ / n_test
    vec_H = H.flatten(order='F')
    approx_term = solve(2 * nu * I_L2 + H_plus_H, vec_Q)
    var_inf_approx.append(np.dot(vec_H, approx_term) * mu / n_test)

# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------
# Set global font size mapping to Plots.scalefontsizes(1.5)
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(8, 6))

# Use raw strings (r"...") for native Matplotlib LaTeX rendering
plt.plot(mu_pl, var_inf, label=r"$\mathrm{var}\{g_\infty\}$ Eqs. (40,41)", linewidth=2)
plt.plot(mu_pl, var_inf_approx, label=r"$\mathrm{var}\{g_\infty\}$ first order approximation Eq. (43)", linewidth=2)

plt.yscale('log')
plt.xlabel(r"$\mu$")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3, which="both", ls="--") # Added grid for log-scale readability

plt.tight_layout()
plt.savefig("small_step.pdf")
plt.show()