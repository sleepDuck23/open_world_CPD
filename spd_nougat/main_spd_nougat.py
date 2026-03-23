import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
from function import compute_manifold_windows, SPD_NOUGAT

np.random.seed(42)
Total_Time = 500
d = 3
change_point = 250

# The mean stays exactly 0 for the entire timeline
shared_mean = np.zeros(d)

# Baseline State: Independent channels, variance of 1 (Identity matrix)
cov_1 = np.eye(d)

# Anomaly State: Channels become highly correlated, and variance increases
cov_2 = np.array([
    [3.0, 1.8, 1.2], 
    [1.8, 3.0, 1.8], 
    [1.2, 1.8, 3.0]
])

# Generate the two parts
data_part1 = np.random.multivariate_normal(shared_mean, cov_1, size=change_point)
data_part2 = np.random.multivariate_normal(shared_mean, cov_2, size=Total_Time - change_point)

# Combine them
raw_data = np.vstack((data_part1, data_part2))

L_window = 30  # Data points per covariance matrix
N_window = 10  # Covariance matrices per reference/test window

# Calculate when we can start (we need enough history to form the first windows)
start_t = 2 * N_window + L_window - 2

# Bootstrap the initial dictionary using the first reference window
Sref_initial, _ = compute_manifold_windows(raw_data, N_window, L_window, start_t)
initial_dict = Sref_initial  

# Initialize NOUGAT with parameters tuned for covariance shifts
nougat = SPD_NOUGAT(mu=0.05, initial_dictionary=initial_dict, nu=1e-4, 
                    eta_0=0.6, xi=0.85, sigma=2.0)

g_statistics = []
time_indices = []

print(f"Starting NOUGAT online change detection from t={start_t} to {Total_Time}...")

for t in range(start_t, Total_Time - 1):
    # 1. Get current reference and test windows
    Sref, Stest = compute_manifold_windows(raw_data, N_window, L_window, t)
    
    # 2. Look ahead one step to get the newest observation (S_new) for the dictionary check
    _, Stest_next = compute_manifold_windows(raw_data, N_window, L_window, t + 1)
    S_new = Stest_next[-1]
    
    # 3. Run the change detection step
    g = nougat.step(t, Sref, Stest, S_new)
    
    # 4. Save results
    g_statistics.append(g)
    time_indices.append(t + 1)

    print(f"current time step: {t + 1}")

print(f"Algorithm finished. Final Dictionary Size: {nougat.L}")
print(f"Detected Change Points at time steps: {nougat.changepoints}")


fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Raw Data
ax[0].plot(raw_data, alpha=0.6)
ax[0].axvline(change_point, color='black', linestyle='--', linewidth=2, label='Actual Covariance Shift')
ax[0].set_title('Raw Time Series (Mean = 0, Covariance changes at t=500)')
ax[0].set_ylabel('Amplitude')
ax[0].legend()

# Plot 2: NOUGAT Statistic
ax[1].plot(time_indices, g_statistics, label='Test Statistic $g_t$', color='blue')
ax[1].axhline(nougat.xi - 1, color='red', linestyle='--', label='Detection Threshold ($\\xi$)')
ax[1].axhline(-nougat.xi - 1, color='red', linestyle='--')
ax[1].axvline(change_point, color='green', linestyle='-', linewidth=2, label='Actual Change Point')

# Mark detected changes
for cp in nougat.changepoints:
    ax[1].axvline(cp, color='orange', linestyle=':', linewidth=2, label='Detected Change')

# Clean up duplicate legend labels
handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys())

ax[1].set_title('NOUGAT Detection Statistic')
ax[1].set_xlabel('Time Step ($t$)')
ax[1].set_ylabel('Statistic Value')

plt.tight_layout()
plt.show()