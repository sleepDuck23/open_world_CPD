import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
from function import  SPD_NOUGAT, warm_start_dict
from spd_generation import generate_wishart_series, generate_multiple_wishart_series

np.random.seed(42)
Total_Time = 1000
d = 3
#change_point = 150
true_changepoints = [300, 600, 900]  # Multiple change points for testing

N_window = 20  # Covariance matrices per reference/test window

eta_0_val = 0.35
sigma_val = 1.75 # Standard deviation for noise in the estimation of g_t
nu_val = 1e-4  
mu_val = 1e-1
xi_val = 0.2

cooldown_steps = 2 * N_window

#raw_data = generate_wishart_series(total_steps=Total_Time, change_point=change_point, dim=d)
raw_data = generate_multiple_wishart_series(total_steps=Total_Time, changepoints=true_changepoints, dim=d, df=10)

# Calculate when we can start (we need enough history to form the first windows)
start_t = 2 * N_window - 1

# Compute the full initial windows
Sref = raw_data[start_t - 2 * N_window + 1 : start_t - N_window + 1].copy()
Stest = raw_data[start_t - N_window + 1 : start_t + 1].copy()

initial_dict = warm_start_dict(Sref, eta_0=eta_0_val, sigma=sigma_val)


# Initialize NOUGAT with parameters tuned for covariance shifts
nougat = SPD_NOUGAT(mu=mu_val, initial_dictionary=initial_dict, nu=nu_val, 
                    eta_0=eta_0_val, xi=xi_val, sigma=sigma_val, 
                    cooldown_period=cooldown_steps)

g_statistics = []
dic_sizes = []
time_indices = []

e_circ_values = []
theta_values = []

print(f"Starting NOUGAT online change detection from t={start_t+1} to {Total_Time}...")

# 2. Main Loop
for t in range(start_t+1, Total_Time):
    S_new = raw_data[t]
    
    matrix_leaving_test = Stest[0].copy()
    Sref[:-1] = Sref[1:]
    Sref[-1] = matrix_leaving_test  
    
    Stest[:-1] = Stest[1:]
    Stest[-1] = S_new               
    
    g, theta, e_circ = nougat.step(t, Sref, Stest)
    
    # 4. Save results
    g_statistics.append(g)
    time_indices.append(t)  
    dic_sizes.append(nougat.L if nougat.cooldown_counter == 0 else np.nan)
    theta_values.append(theta)
    e_circ_values.append(e_circ)

# Catch the final active dictionary
nougat.finalize()

print(f"Algorithm finished.")
print(f"Total Dictionaries in Library: {len(nougat.dictionary_library)}")
print(f"Detected Change Points: {nougat.global_changepoints}")

# Print the sizes of all saved dictionaries
for i, saved_dict in enumerate(nougat.dictionary_library):
    print(f"Dictionary {i} has size: {saved_dict.shape[0]} matrices")

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ==========================================
# Plot 1: Raw Time Series & Detected Changes
# ==========================================
traces = [np.trace(mat) for mat in raw_data]
ax[0].plot(traces, alpha=0.7, color='teal', label='Trace of SPD Matrix')

# LOOP: Plot all actual ground-truth change points
for tcp in true_changepoints:
    ax[0].axvline(tcp, color='black', linestyle='--', linewidth=2, label='Actual Covariance Shift')

# Mark detected changes on the raw data
for cp in nougat.global_changepoints:
    ax[0].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

# Clean up duplicate legend labels (This perfectly handles the multiple actual/detected lines!)
handles, labels = ax[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[0].legend(by_label.values(), by_label.keys(), loc='upper right')

# Update the title and labels to reflect what we are actually plotting now
ax[0].set_title('Time Series of SPD Matrices (Represented by Matrix Trace)')
ax[0].set_ylabel('Trace Value')

# ==========================================
# Plot 2: NOUGAT Statistic (g_t) & Thresholds
# ==========================================
ax[1].plot(time_indices, g_statistics, label='estimation parameter $g_t$', color='blue')
ax[1].axhline(nougat.xi, color='red', linestyle='--', label='Detection Threshold ($+\\xi$)')
# LOOP: Plot all actual ground-truth change points
for tcp in true_changepoints:
    ax[1].axvline(tcp, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

# Mark detected changes on the statistic plot
for cp in nougat.global_changepoints:
    ax[1].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')

ax[1].set_title('NOUGAT statistic ($g_t$)')
ax[1].set_ylabel('Statistic Value')


# ==========================================
# Plot 3: Dictionary Size Evolution
# ==========================================
ax[2].plot(time_indices, dic_sizes, label='Dictionary Size ($L$)', color='purple', linestyle='-')

# LOOP: Plot all actual ground-truth change points
for tcp in true_changepoints:
    ax[2].axvline(tcp, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

# Optional: mark detected changes here too for alignment
for cp in nougat.global_changepoints:
    ax[2].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[2].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[2].legend(by_label.values(), by_label.keys(), loc='upper left')

ax[2].set_title('Dictionary Size Over Time')
ax[2].set_xlabel('Time Step ($t$)')
ax[2].set_ylabel('Size')

# Apply tight layout and show
plt.tight_layout()
plt.show()

# --- Plotting Phase ---
# Create a 3-panel subplot sharing the same X-axis (time)
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Safe extraction of the first element: 
# Grab index [0] if it's an array, otherwise keep it as np.nan
theta0_safe = [t[0] if isinstance(t, np.ndarray) else np.nan for t in theta_values]
ecirc0_safe = [e[0] if isinstance(e, np.ndarray) else np.nan for e in e_circ_values]

# Plot 1: Change Point Statistic (g)
axes[0].plot(time_indices, g_statistics, color='blue', label='Statistic (g)')
axes[0].axhline(y=nougat.xi, color='red', linestyle='--', label='Threshold (xi)')
for cp in nougat.global_changepoints:
    axes[0].axvline(x=cp, color='orange', linestyle=':', alpha=0.7)
axes[0].set_ylabel('g value')
axes[0].set_title('SPD NOUGAT: Detection Statistic')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Theta (First Element)
axes[1].plot(time_indices, theta0_safe, color='green', label=r'$\theta_0$')
for cp in nougat.global_changepoints:
    axes[1].axvline(x=cp, color='orange', linestyle=':', alpha=0.7)
axes[1].set_ylabel(r'$\theta[0]$')
axes[1].set_title('Evolution of Dictionary Weight 0')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: e_circ (First Element)
axes[2].plot(time_indices, ecirc0_safe, color='purple', label=r'$e_{circ, 0}$')
for cp in nougat.global_changepoints:
    axes[2].axvline(x=cp, color='orange', linestyle=':', alpha=0.7)
axes[2].set_ylabel(r'$e_{circ}[0]$')
axes[2].set_xlabel('Time Step (t)')
axes[2].set_title('Evolution of Virtual Error 0')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()