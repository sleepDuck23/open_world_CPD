import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from function import SPD_NOUGAT_OnlineOpenWorld, warm_start_dict 
from spd_generation import generate_alternating_wishart_series

# --- 1. Setup & Hyperparameters ---
np.random.seed(42)
Total_Time = 5000
num_runs = 10000  

d = 3
N_window = 20  

# Single change point setup
true_cp = 2500
true_changepoints = [true_cp]  
distribution_sequence = [0, 1] # State transitions from 0 to 1 at t=2500

eta_0_val = 0.35
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2
psi_val = 0.45 

# Tolerance for what counts as a "true detection" vs a random false alarm
tolerance_window = 400 

# --- 2. Data Storage Matrices ---
# Pre-allocate arrays for continuous tracking
all_g_stats = np.zeros((num_runs, Total_Time))
all_dic_sizes = np.zeros((num_runs, Total_Time))

# Storage for Delay metrics
all_delays = []
missed_detections = 0

print(f"Starting {num_runs} Monte Carlo simulations (Length: {Total_Time} steps)...")

# --- 3. Main Monte Carlo Loop ---
for run in tqdm(range(num_runs), desc="Simulation Progress"):
    
    # 3a. Generate data with a single change point for this run
    raw_data = generate_alternating_wishart_series(
        total_steps=Total_Time, 
        changepoints=true_changepoints, 
        distribution_sequence=distribution_sequence, 
        dim=d, 
        df=10
    )
    
    # 3b. Initialization
    Sref_initial = raw_data[0:N_window]
    initial_dict = warm_start_dict(Sref_initial, eta_0=eta_0_val, sigma=sigma_val)
    
    nougat = SPD_NOUGAT_OnlineOpenWorld(
        mu=mu_val, 
        initial_dictionary=initial_dict, 
        nu=nu_val, 
        eta_0=eta_0_val, 
        xi=xi_val, 
        sigma=sigma_val, 
        psi=psi_val,
        N=N_window  
    )
    
    # 3c. Online Loop
    for t in range(Total_Time):
        S_new = raw_data[t]
        g = nougat.step(t, S_new)
        
        all_g_stats[run, t] = g
        
        if nougat.state == "ACTIVE":
            all_dic_sizes[run, t] = nougat.L
        else:
            all_dic_sizes[run, t] = np.nan
            
    # 3d. Calculate Delay for this specific run
    cps = nougat.global_changepoints
    detected_true_cp = False
    
    for cp in cps:
        # Check if the detected CP falls within the acceptable window after the true CP
        if true_cp <= cp <= true_cp + tolerance_window:
            delay = cp - true_cp
            all_delays.append(delay)
            detected_true_cp = True
            break # We only care about the first detection in the window
            
    if not detected_true_cp:
        missed_detections += 1

# --- 4. Compute Statistical Aggregations ---
mean_g = np.mean(all_g_stats, axis=0)
std_g = np.std(all_g_stats, axis=0)

mean_dic = np.nanmean(all_dic_sizes, axis=0)
std_dic = np.nanstd(all_dic_sizes, axis=0)

print("\n--- Detection Delay Metrics ---")
print(f"Total Runs: {num_runs}")
print(f"Missed Detections: {missed_detections} ({(missed_detections/num_runs)*100:.2f}%)")

if all_delays:
    mean_delay = np.mean(all_delays)
    median_delay = np.median(all_delays)
    std_delay = np.std(all_delays)
    
    print(f"Mean Delay: {mean_delay:.2f} steps")
    print(f"Median Delay: {median_delay:.2f} steps")
    print(f"Std Dev of Delay: {std_delay:.2f} steps")
    print(f"Min Delay: {np.min(all_delays)} steps | Max Delay: {np.max(all_delays)} steps")

# --- 5. Export Plotting Data to CSV ---
print("\nExporting data to CSV...")

# 5A. Export Time-Series Statistics
df_time_series = pd.DataFrame({
    'time_step': np.arange(Total_Time),
    'mean_g': mean_g,
    'std_g': std_g,
    'mean_dic': mean_dic,
    'std_dic': std_dic
})
df_time_series.to_csv('delay_timeseries_stats.csv', index=False)
print("Saved: delay_timeseries_stats.csv")

# 5B. Export Delay Distribution Data
if all_delays:
    df_delays = pd.DataFrame({
        'detection_delay': all_delays
    })
    df_delays.to_csv('delay_detection_delays.csv', index=False)
    print("Saved: delay_detection_delays.csv")

# --- 6. Visualizations & PDF Export ---
print("\nGenerating and saving plots...")
time_axis = np.arange(Total_Time)

# Figure 1: Detection Statistic and Dictionary Size
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1A: Mean and Std of NOUGAT Statistic
ax1[0].plot(time_axis, mean_g, color='blue', label='Mean $g_t$')
ax1[0].fill_between(time_axis, mean_g - std_g, mean_g + std_g, color='blue', alpha=0.2, label='$\pm 1$ Std Dev')
ax1[0].axhline(xi_val, color='red', linestyle='--', label=f'Threshold ($\\xi={xi_val}$)')
ax1[0].axvline(true_cp, color='black', linestyle='-', linewidth=2, label=f'True Change (t={true_cp})')
ax1[0].set_title(f'Detection Statistic ($g_t$) over {num_runs} Runs')
ax1[0].set_ylabel('$g_t$')
ax1[0].legend(loc='upper right')
ax1[0].grid(True, linestyle='--', alpha=0.5)

# Plot 1B: Mean and Std of Dictionary Size
ax1[1].plot(time_axis, mean_dic, color='purple', label='Mean Dictionary Size')
ax1[1].fill_between(time_axis, mean_dic - std_dic, mean_dic + std_dic, color='purple', alpha=0.2, label='$\pm 1$ Std Dev')
ax1[1].axvline(true_cp, color='black', linestyle='-', linewidth=2)
ax1[1].set_title(f'Active Dictionary Size ($L$) over {num_runs} Runs')
ax1[1].set_xlabel('Time Step ($t$)')
ax1[1].set_ylabel('Size')
ax1[1].legend(loc='upper left')
ax1[1].grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()
fig1.savefig('delay_statistic_and_dictionary_size.pdf', format='pdf', bbox_inches='tight')
print("Saved: delay_statistic_and_dictionary_size.pdf")

# Figure 2: Histogram of True Detected Delays
if all_delays:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(all_delays, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    
    ax2.axvline(mean_delay, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean Delay: {mean_delay:.1f}')
    ax2.axvline(median_delay, color='blue', linestyle='dashed', linewidth=2, 
                label=f'Median Delay: {median_delay:.1f}')

    ax2.set_title(f'Distribution of Detection Delay\n(Excluding {missed_detections} Missed Detections)')
    ax2.set_xlabel('Delay (Time Steps after True Change Point)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5)
    
    fig2.tight_layout()
    fig2.savefig('detection_delay_distribution.pdf', format='pdf', bbox_inches='tight')
    print("Saved: detection_delay_distribution.pdf")

plt.show()