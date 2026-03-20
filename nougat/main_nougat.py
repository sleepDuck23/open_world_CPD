import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

# Import your core algorithms from the my_funcs.py file
from functions_nougat import (nougat, rulsif, ma, knnt, comp_pfa, 
                      comp_mtd, comp_roc, pdf_gmm, median_trick)

print("Starting Quick Test Pipeline...")

# =========================================================
# 1. PARAMETERS (Small Scale for Quick Testing)
# =========================================================
d = 6
k_gmm = 3

# Signal parameters
nc = 100        # Change point
nt = 200        # Total time series length

# Algorithm parameters
n_ref = 30      # Reference window size
n_test = 30     # Test window size
mu = 0.047      # Step size (NOUGAT)
nu = 0.01       # Ridge regularization
k_knn = 10      # k for k-NN

# Execution parameters
realmax = 50    # Number of Monte Carlo iterations (reduced from 1,000,000)
t_burn = 20     # Burn-in period (reduced to fit the smaller windows)

# GMM Setup
means_h0, covs_h0, weights_h0 = pdf_gmm(d, k_gmm, sigma=1.0)
means_h1, covs_h1, weights_h1 = pdf_gmm(d, k_gmm, sigma=1.0)

def sample_gmm(means, covs, weights, n_samples):
    d = means.shape[1]
    k = len(weights)
    components = np.random.choice(k, size=n_samples, p=weights)
    samples = np.zeros((d, n_samples))
    for i in range(k):
        idx = (components == i)
        n_i = np.sum(idx)
        if n_i > 0:
            samples[:, idx] = np.random.multivariate_normal(means[i], covs[i], size=n_i).T
    return samples

def sample_h0(n): return sample_gmm(means_h0, covs_h0, weights_h0, n)
def sample_h1(n): return sample_gmm(means_h1, covs_h1, weights_h1, n)

# Compute kernel bandwidth
gamma = median_trick(sample_h0(100))

# =========================================================
# 2. MONTE CARLO SIMULATION
# =========================================================
def run_iteration(k):
    """Executes a single test iteration."""
    # Generate dictionary and time series
    dict_x = np.hstack([sample_h0(40), sample_h1(40)])
    x = np.hstack([sample_h0(nc - 1), sample_h1(nt - nc + 1)])
    
    # Run algorithms
    res_nougat = nougat(x, dict_x, n_ref, n_test, mu, nu, gamma)
    res_rulsif = rulsif(x, dict_x, n_ref, n_test, nu, gamma)
    res_ma = ma(x, dict_x, n_ref, n_test, gamma)
    res_knn = knnt(x, n_ref, n_test, k_knn)
    
    return res_nougat, res_rulsif, res_ma, res_knn

print(f"Running {realmax} iterations across CPU cores...")
start_time = time.time()

# Run in parallel and unzip the results
results = Parallel(n_jobs=-1)(delayed(run_iteration)(k) for k in range(realmax))
res_nougat, res_rulsif, res_ma, res_knn = zip(*results)

# Convert lists of arrays to 2D numpy arrays (shape: time x iterations)
t_nougat = np.column_stack(res_nougat)
t_rulsif = np.column_stack(res_rulsif)
t_ma = np.column_stack(res_ma)
t_knn = np.column_stack(res_knn)

print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

# =========================================================
# 3. EVALUATION & METRICS
# =========================================================
print("Computing metrics...")
nc_detect = nc - n_ref - n_test

# Slicing the pre-change (t0) and post-change (t1) periods
t0_nougat = t_nougat[t_burn-1 : nc_detect-1, :]
t0_rulsif = t_rulsif[t_burn-1 : nc_detect-1, :]
t0_ma     = t_ma[t_burn-1 : nc_detect-1, :]
t0_knn    = t_knn[t_burn-1 : nc_detect-1, :]

t1_nougat = t_nougat[nc_detect-1 :, :]
t1_rulsif = t_rulsif[nc_detect-1 :, :]
t1_ma     = t_ma[nc_detect-1 :, :]
t1_knn    = t_knn[nc_detect-1 :, :]

# Compute PFA
pfa_nougat, xi_nougat = comp_pfa(t0_nougat)
pfa_rulsif, xi_rulsif = comp_pfa(t0_rulsif)
pfa_ma, xi_ma         = comp_pfa(t0_ma)
pfa_knn, xi_knn       = comp_pfa(t0_knn)

# Compute ROC
pfa_roc_nougat, pd_roc_nougat, _ = comp_roc(t0_nougat, t1_nougat, xi_pl=xi_nougat)
pfa_roc_rulsif, pd_roc_rulsif, _ = comp_roc(t0_rulsif, t1_rulsif, xi_pl=xi_rulsif)
pfa_roc_ma, pd_roc_ma, _         = comp_roc(t0_ma, t1_ma, xi_pl=xi_ma)
pfa_roc_knn, pd_roc_knn, _       = comp_roc(t0_knn, t1_knn, xi_pl=xi_knn)

# =========================================================
# 4. PLOTTING
# =========================================================
print("Generating and saving plots...")
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

# --- Plot 1: ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pfa_roc_nougat, pd_roc_nougat, label="NOUGAT", color="C0")
ax.plot(pfa_roc_rulsif, pd_roc_rulsif, label="dRuLSIF", color="C1")
ax.plot(pfa_roc_ma, pd_roc_ma, label="MA", color="C2")
ax.plot(pfa_roc_knn, pd_roc_knn, label="k-NN", color="C3")

ax.set_xlim(0, 0.2)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel("PFA")
ax.set_ylabel("PD")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.title("ROC Curve (Quick Test - 50 Iterations)")
plt.savefig("quick_test_roc.pdf")
plt.close()

# --- Plot 2: Time Series ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

def plot_ribbon(ax, x, data, label, color):
    mean_val = np.mean(data, axis=1)
    std_val = np.std(data, axis=1)
    ax.plot(x, mean_val, label=label, color=color)
    ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.3, color=color)
    ax.axvline(x=nc, color='black', linestyle='--')
    ax.axvline(x=nc + n_test, color='black', linestyle='--')
    ax.legend(loc="upper left")

x_noug_rul = np.arange(n_ref + n_test, nt)
x_ma_knn = np.arange(n_ref + n_test - 1, nt)

plot_ribbon(axs[0], x_noug_rul, t_nougat, "NOUGAT", "C0")
plot_ribbon(axs[1], x_noug_rul, t_rulsif, "dRuLSIF", "C1")
plot_ribbon(axs[2], x_ma_knn, t_ma, "MA", "C2")
plot_ribbon(axs[3], x_ma_knn, t_knn, "k-NN", "C3")

plt.tight_layout()
plt.savefig("quick_test_AllStatistics.pdf")
plt.close()

print("All done! Check your folder for 'quick_test_roc.pdf' and 'quick_test_AllStatistics.pdf'.")