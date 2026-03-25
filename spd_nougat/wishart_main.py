import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
from function import  SPD_NOUGAT
from spd_generation import generate_wishart_series

np.random.seed(42)
Total_Time = 200
d = 3
change_point = 150

L_window = 10  # Data points per covariance matrix
N_window = 10  # Covariance matrices per reference/test window

raw_data = generate_wishart_series(total_steps=Total_Time, change_point=change_point, dim=d)

# Calculate when we can start (we need enough history to form the first windows)
start_t = 2 * N_window - 1

# Compute the full initial windows
Sref = raw_data[start_t - 2 * N_window + 1 : start_t - N_window + 1].copy()
Stest = raw_data[start_t - N_window + 1 : start_t + 1].copy()
initial_dict = Sref.copy()


# Initialize NOUGAT with parameters tuned for covariance shifts
nougat = SPD_NOUGAT(mu=0.1, initial_dictionary=initial_dict, nu=1e-4, 
                    eta_0=0.5, xi=1.2, sigma=2)

g_statistics = []
dic_sizes = []
time_indices = []

print(f"Starting NOUGAT online change detection from t={start_t} to {Total_Time}...")

for t in range(start_t, Total_Time - 1):
    S_new = raw_data[t + 1]
    
    # 2. Run the NOUGAT step with the CURRENT windows
    g = nougat.step(t, Sref, Stest, S_new)
    
    # 3. PROCEED WITH THE SLIDE 
    
    # Step A: Save the matrix that is bridging the gap
    matrix_leaving_test = Stest[0].copy()
    
    # Step B: Shift Sref left by 1 (drops the oldest Sref)
    Sref[:-1] = Sref[1:]
    
    # Step C: Append the bridge matrix to the end of Sref
    Sref[-1] = matrix_leaving_test
    
    # Step D: Shift Stest left by 1
    Stest[:-1] = Stest[1:]
    
    # Step E: Append the brand new matrix to the end of Stest
    Stest[-1] = S_new
    
    # 4. Save results
    g_statistics.append(g)
    time_indices.append(t + 1)
    dic_sizes.append(nougat.L)
    

    print(f"current time step: {t + 1}")

print(f"Algorithm finished. Final Dictionary Size: {nougat.L}")
print(f"Detected Change Points at time steps: {nougat.changepoints}")


fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ==========================================
# Plot 1: Raw Time Series & Detected Changes
# ==========================================
traces = [np.trace(mat) for mat in raw_data]
ax[0].plot(traces, alpha=0.7, color='teal', label='Trace of SPD Matrix')

ax[0].axvline(change_point, color='black', linestyle='--', linewidth=2, label='Actual Covariance Shift')

# Mark detected changes on the raw data
for cp in nougat.changepoints:
    ax[0].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

# Clean up duplicate legend labels
handles, labels = ax[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[0].legend(by_label.values(), by_label.keys(), loc='upper right')

# Update the title and labels to reflect what we are actually plotting now
ax[0].set_title('Time Series of SPD Matrices (Represented by Matrix Trace)')
ax[0].set_ylabel('Trace Value')

# ==========================================
# Plot 2: NOUGAT Statistic (g_t) & Thresholds
# ==========================================
ax[1].plot(time_indices, g_statistics, label='Test Statistic $g_t$', color='blue')
ax[1].axhline(nougat.xi - 1, color='red', linestyle='--', label='Detection Threshold ($+\\xi$)')
ax[1].axvline(change_point, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

# Mark detected changes on the statistic plot
for cp in nougat.changepoints:
    ax[1].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')

ax[1].set_title('NOUGAT Detection Statistic ($g_t$)')
ax[1].set_ylabel('Statistic Value')

# ==========================================
# Plot 3: Dictionary Size Evolution
# ==========================================
ax[2].plot(time_indices, dic_sizes, label='Dictionary Size ($L$)', color='purple', linestyle='-')
ax[2].axvline(change_point, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

# Optional: mark detected changes here too for alignment
for cp in nougat.changepoints:
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