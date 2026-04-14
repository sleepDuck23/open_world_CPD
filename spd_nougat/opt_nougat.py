# Runing an optimized version of NOUGAT on the Wishart synthetic data

import numpy as np
import matplotlib.pyplot as plt
from function import SPD_NOUGAT_optimized, warm_start_dict
from spd_generation import generate_multiple_wishart_series

np.random.seed(42)
Total_Time = 1200
d = 3
true_changepoints = [300, 600, 900]  # Multiple change points for testing

N_window = 20  

eta_0_val = 0.2
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2

cooldown_steps = 2 * N_window


raw_data = generate_multiple_wishart_series(total_steps=Total_Time, changepoints=true_changepoints, dim=d, df=10)

Sref_initial = raw_data[0:N_window]
initial_dict = warm_start_dict(Sref_initial, eta_0=eta_0_val, sigma=sigma_val)

nougat = SPD_NOUGAT_optimized(
    mu=mu_val, 
    initial_dictionary=initial_dict, 
    nu=nu_val, 
    eta_0=eta_0_val, 
    xi=xi_val, 
    sigma=sigma_val, 
    cooldown_period=cooldown_steps,
    N=N_window  
)

g_statistics = []
dic_sizes = []
time_indices = []

print("Starting optimized NOUGAT online change detection...")


for t in range(Total_Time):
    S_new = raw_data[t]
    
    # All window management is now handled internally!
    g = nougat.step(t, S_new)
    
    g_statistics.append(g)
    time_indices.append(t)  
    
    # Track dictionary size (ignore during warmup and cooldown phases)
    if np.isnan(g) or nougat.cooldown_counter > 0:
        dic_sizes.append(np.nan)
    else:
        dic_sizes.append(nougat.L)

nougat.finalize()  # Save the last active dictionary 

print("Algorithm finished.")
print(f"Total Dictionaries in Library: {len(nougat.dictionary_library)}")
print(f"Detected Change Points: {nougat.global_changepoints}")

# Print the sizes of all saved dictionaries
for i, saved_dict in enumerate(nougat.dictionary_library):
    print(f"Dictionary {i} has size: {saved_dict.shape[0]} matrices")



fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Raw Time Series & Detected Changes
traces = [np.trace(mat) for mat in raw_data]
ax[0].plot(traces, alpha=0.7, color='teal', label='Trace of SPD Matrix')

for tcp in true_changepoints:
    ax[0].axvline(tcp, color='black', linestyle='--', linewidth=2, label='Actual Covariance Shift')

for cp in nougat.global_changepoints:
    ax[0].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[0].legend(by_label.values(), by_label.keys(), loc='upper right')
ax[0].set_title('Time Series of SPD Matrices (Represented by Matrix Trace)')
ax[0].set_ylabel('Trace Value')

# Plot 2: NOUGAT Statistic (g_t) & Thresholds
ax[1].plot(time_indices, g_statistics, label='estimation parameter $g_t$', color='blue')
ax[1].axhline(nougat.xi, color='red', linestyle='--', label='Detection Threshold ($+\\xi$)')

for tcp in true_changepoints:
    ax[1].axvline(tcp, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

for cp in nougat.global_changepoints:
    ax[1].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')
ax[1].set_title('NOUGAT statistic ($g_t$)')
ax[1].set_ylabel('Statistic Value')

# Plot 3: Dictionary Size Evolution
ax[2].plot(time_indices, dic_sizes, label='Dictionary Size ($L$)', color='purple', linestyle='-')

for tcp in true_changepoints:
    ax[2].axvline(tcp, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

for cp in nougat.global_changepoints:
    ax[2].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[2].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[2].legend(by_label.values(), by_label.keys(), loc='upper left')
ax[2].set_title('Dictionary Size Over Time')
ax[2].set_xlabel('Time Step ($t$)')
ax[2].set_ylabel('Size')

plt.tight_layout()
plt.show()