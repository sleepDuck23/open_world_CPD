import numpy as np
import matplotlib.pyplot as plt
import h5py
from functions_nougat import comp_pfa, comp_mtd, comp_roc
from params import nc, nt, n_ref, n_test

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
# Note: Julia's JLD2 format is HDF5. We use h5py to load it.
# Depending on how it was saved, you might need to transpose (.T) the arrays.
with h5py.File("MonteCarlo.jld2", "r") as f:
    t_nougat = np.array(f["t_nougat"])
    t_rulsif = np.array(f["t_rulsif"])
    t_ma = np.array(f["t_ma"])
    t_knn = np.array(f["t_knn"])

# ---------------------------------------------------------
# 2. Compute ROC & Metrics Setup
# ---------------------------------------------------------
nc_detect = nc - n_ref - n_test
t_burn = 20  # Burn-in period (mainly useful for nougat)

# Python is 0-indexed, so we subtract 1 from the Julia indices.
# Slicing in Python is exclusive at the top end: [start : end]
t0_nougat = t_nougat[t_burn - 1 : nc_detect - 1, :]
t0_rulsif = t_rulsif[t_burn - 1 : nc_detect - 1, :]
t0_ma     = t_ma[t_burn - 1 : nc_detect - 1, :]
t0_knn    = t_knn[t_burn - 1 : nc_detect - 1, :]

t1_nougat = t_nougat[nc_detect - 1 :, :]
t1_rulsif = t_rulsif[nc_detect - 1 :, :]
t1_ma     = t_ma[nc_detect - 1 :, :]
t1_knn    = t_knn[nc_detect - 1 :, :]

# ---------------------------------------------------------
# 3. Compute PFA
# ---------------------------------------------------------
pfa_nougat, xi_nougat = comp_pfa(t0_nougat)
pfa_ma, xi_ma         = comp_pfa(t0_ma)
pfa_rulsif, xi_rulsif = comp_pfa(t0_rulsif)
pfa_knn, xi_knn       = comp_pfa(t0_knn)

# ---------------------------------------------------------
# 4. Compute MTFA (Mean Time to False Alarm)
# ---------------------------------------------------------
# In Python, comp_mtd returns (mean_delay, xi_pl), so we take index [0]
mtfa_nougat = comp_mtd(t0_nougat, xi_pl=xi_nougat)[0]
mtfa_ma     = comp_mtd(t0_ma, xi_pl=xi_ma)[0]
mtfa_rulsif = comp_mtd(t0_rulsif, xi_pl=xi_rulsif)[0]
mtfa_knn    = comp_mtd(t0_knn, xi_pl=xi_knn)[0]

# ---------------------------------------------------------
# 5. Compute MTD (Mean Time to Detection)
# ---------------------------------------------------------
mtd_nougat = comp_mtd(t1_nougat, xi_pl=xi_nougat)[0] + (n_ref + n_test)
mtd_ma     = comp_mtd(t1_ma, xi_pl=xi_ma)[0] + (n_ref + n_test)
mtd_rulsif = comp_mtd(t1_rulsif, xi_pl=xi_rulsif)[0] + (n_ref + n_test)
mtd_knn    = comp_mtd(t1_knn, xi_pl=xi_knn)[0] + (n_ref + n_test)

# ---------------------------------------------------------
# 6. Compute ROC Curvess
# ---------------------------------------------------------
pfa_roc_nougat, pd_roc_nougat, _ = comp_roc(t0_nougat, t1_nougat, xi_pl=xi_nougat)
pfa_roc_rulsif, pd_roc_rulsif, _ = comp_roc(t0_rulsif, t1_rulsif, xi_pl=xi_rulsif)
pfa_roc_ma, pd_roc_ma, _         = comp_roc(t0_ma, t1_ma, xi_pl=xi_ma)
pfa_roc_knn, pd_roc_knn, _       = comp_roc(t0_knn, t1_knn, xi_pl=xi_knn)


# =========================================================
# PLOTTING
# =========================================================
# Set global plot parameters mimicking Julia's scalefontsizes
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

# --- Plot 1: All Statistics Time Series ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Helper function to plot ribbon (mean + std)
def plot_ribbon(ax, x, data, label, color):
    mean_val = np.mean(data, axis=1)
    std_val = np.std(data, axis=1)
    ax.plot(x, mean_val, label=label, color=color)
    ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.3, color=color)
    ax.axvline(x=nc, color='black', linestyle='--')
    ax.axvline(x=nc + n_test, color='black', linestyle='--')
    ax.legend(loc="upper left")

# X-axis arrays (adjusting lengths)
x_noug_rul = np.arange(n_ref + n_test, nt)
x_ma_knn = np.arange(n_ref + n_test - 1, nt)

plot_ribbon(axs[0], x_noug_rul, t_nougat, "NOUGAT", "C0")
plot_ribbon(axs[1], x_noug_rul, t_rulsif, "dRuLSIF", "C1")
plot_ribbon(axs[2], x_ma_knn, t_ma, "MA", "C2")
plot_ribbon(axs[3], x_ma_knn, t_knn, "k-NN", "C3")

plt.tight_layout()
plt.savefig("AllStatistics.pdf")
plt.show()

# --- Plot 2: MTFA vs PFA ---
plt.figure(figsize=(8, 6))
plt.plot(pfa_nougat, mtfa_nougat, label="NOUGAT")
plt.plot(pfa_ma, mtfa_ma, label="MA")
plt.plot(pfa_rulsif, mtfa_rulsif, label="dRuLSIF")
plt.plot(pfa_knn, mtfa_knn, label="k-NN")

plt.xlim(0.02, 0.2)
plt.ylim(30, 90)
plt.xlabel("PFA")
plt.ylabel("MTFA")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("mtfa.pdf")
plt.show()

# --- Plot 3: MTD vs PFA ---
plt.figure(figsize=(8, 6))
plt.plot(pfa_nougat, mtd_nougat, label="NOUGAT")
plt.plot(pfa_ma, mtd_ma, label="MA")
plt.plot(pfa_rulsif, mtd_rulsif, label="dRuLSIF")
plt.plot(pfa_knn, mtd_knn, label="k-NN")

plt.xlim(0.01, 0.2)
plt.xlabel("PFA")
plt.ylabel("MTD")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("mtfd.pdf")
plt.show()

# --- Plot 4: ROC Curves with Inset Zoom ---
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

# Create inset axes (lens! equivalent)
# Bounds: [x0, y0, width, height] relative to the main axes
axins = ax.inset_axes([0.3, 0.2, 0.45, 0.4])

axins.plot(pfa_roc_nougat, pd_roc_nougat, color="C0")
axins.plot(pfa_roc_rulsif, pd_roc_rulsif, color="C1")
axins.plot(pfa_roc_ma, pd_roc_ma, color="C2")
axins.plot(pfa_roc_knn, pd_roc_knn, color="C3")

# Set limits for the zoomed-in lens
axins.set_xlim(0, 0.01)
axins.set_ylim(0.9, 1)
axins.tick_params(labelsize=10)
axins.grid(True, alpha=0.3)

# Add bounding box lines showing where the zoom is connected
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.savefig("roc.pdf")
plt.show()