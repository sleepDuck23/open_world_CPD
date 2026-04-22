import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For the progress bar
from function import SPD_NOUGAT_OnlineOpenWorld, warm_start_dict 
from spd_generation import generate_alternating_wishart_series

# --- 1. Setup & Hyperparameters ---
np.random.seed(42)
Total_Time = 10000
num_runs = 10000  # Set to 10 for quick testing, 1000 for final paper results
d = 3
N_window = 20  

# NO change points. The system stays in state '0' forever.
true_changepoints = []  
distribution_sequence = [0] 

eta_0_val = 0.35
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2
psi_val = 0.45 

# --- 2. Data Storage Matrices ---
# Pre-allocate arrays for continuous tracking
all_g_stats = np.zeros((num_runs, Total_Time))
all_dic_sizes = np.zeros((num_runs, Total_Time))

# Storage for False Alarm metrics
false_alarms_per_run = []
all_fa_intervals = [] # Stores the time distance between every false alarm
time_to_first_fa = [] # Stores how long it took to get the very first false alarm

print(f"Starting {num_runs} Monte Carlo simulations (Length: {Total_Time} steps)...")

# --- 3. Main Monte Carlo Loop ---
for run in tqdm(range(num_runs), desc="Simulation Progress"):
    
    # 3a. Generate stationary data for this specific run
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
    
    # 3c. Online Loop for current run
    for t in range(Total_Time):
        S_new = raw_data[t]
        g = nougat.step(t, S_new)
        
        all_g_stats[run, t] = g
        
        # Track dictionary size only when active
        if nougat.state == "ACTIVE":
            all_dic_sizes[run, t] = nougat.L
        else:
            all_dic_sizes[run, t] = np.nan
            
    # 3d. Record False Alarm Counts & Intervals
    cps = nougat.global_changepoints
    false_alarms_per_run.append(len(cps))
    
    if len(cps) > 0:
        # Time from start (t=0) to the FIRST false alarm
        time_to_first_fa.append(cps[0])
        
        # Time BETWEEN all subsequent false alarms
        intervals = np.diff([0] + cps) 
        all_fa_intervals.extend(intervals)


# --- 4. Compute Statistical Aggregations ---
# Calculate the mean and std across the 'runs' axis (axis=0) at every time step t
mean_g = np.mean(all_g_stats, axis=0)
std_g = np.std(all_g_stats, axis=0)

mean_dic = np.nanmean(all_dic_sizes, axis=0)
std_dic = np.nanstd(all_dic_sizes, axis=0)

# Calculate General False Alarm metrics
mean_fa = np.mean(false_alarms_per_run)
std_fa = np.std(false_alarms_per_run)

print("\n--- Simulation Results (Stationary Data) ---")
print(f"Total Runs: {num_runs}")
print(f"False Alarms per sequence: Mean = {mean_fa:.3f}, Std = {std_fa:.3f}")

# Calculate Time-Based False Alarm metrics
if all_fa_intervals:
    mean_time_between_fa = np.mean(all_fa_intervals)
    median_time_between_fa = np.median(all_fa_intervals)
    std_time_between_fa = np.std(all_fa_intervals)
    
    print("\n--- False Alarm Time Metrics ---")
    print(f"Mean Time Between False Alarms (MTBFA): {mean_time_between_fa:.2f} steps")
    print(f"Median Time Between False Alarms: {median_time_between_fa:.2f} steps")
    print(f"Std Dev of Intervals: {std_time_between_fa:.2f} steps")
    
    if time_to_first_fa:
        print(f"Mean Time to FIRST False Alarm: {np.mean(time_to_first_fa):.2f} steps")
else:
    print("\n--- False Alarm Time Metrics ---")
    print("Zero false alarms detected across all runs. Algorithm is highly conservative.")


# --- 5. Visualizations ---
time_axis = np.arange(Total_Time)

# Figure 1: Detection Statistic and Dictionary Size
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1A: Mean and Std of NOUGAT Statistic (g_t)
ax1[0].plot(time_axis, mean_g, color='blue', label='Mean $g_t$')
ax1[0].fill_between(time_axis, mean_g - std_g, mean_g + std_g, color='blue', alpha=0.2, label='$\pm 1$ Std Dev')
ax1[0].axhline(xi_val, color='red', linestyle='--', label=f'Threshold ($\\xi={xi_val}$)')
ax1[0].set_title(f'Detection Statistic ($g_t$) over {num_runs} Runs (Stationary Data)')
ax1[0].set_ylabel('$g_t$')
ax1[0].legend(loc='upper right')
ax1[0].grid(True, linestyle='--', alpha=0.5)

# Plot 1B: Mean and Std of Dictionary Size
ax1[1].plot(time_axis, mean_dic, color='purple', label='Mean Dictionary Size')
ax1[1].fill_between(time_axis, mean_dic - std_dic, mean_dic + std_dic, color='purple', alpha=0.2, label='$\pm 1$ Std Dev')
ax1[1].set_title(f'Active Dictionary Size ($L$) over {num_runs} Runs')
ax1[1].set_xlabel('Time Step ($t$)')
ax1[1].set_ylabel('Size')
ax1[1].legend(loc='upper left')
ax1[1].grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()

# Figure 2: Histogram of False Alarm Intervals
if all_fa_intervals:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(all_fa_intervals, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    
    ax2.axvline(np.mean(all_fa_intervals), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {np.mean(all_fa_intervals):.0f}')
    ax2.axvline(np.median(all_fa_intervals), color='blue', linestyle='dashed', linewidth=2, 
                label=f'Median: {np.median(all_fa_intervals):.0f}')

    ax2.set_title('Distribution of Time Between False Alarms (MTBFA)')
    ax2.set_xlabel('Number of Time Steps Between Alarms')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5)
    fig2.tight_layout()

plt.show()