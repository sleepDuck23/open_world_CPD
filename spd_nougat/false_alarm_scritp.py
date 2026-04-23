import matplotlib
matplotlib.use('Agg') # CRITICAL: Must be called before importing pyplot on an SSH server

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from function import SPD_NOUGAT_OnlineOpenWorld, warm_start_dict 
from spd_generation import generate_alternating_wishart_series

# --- 1. Setup & Hyperparameters ---
np.random.seed(42)
Total_Time = 100000  
num_runs = 10     

d = 3
N_window = 20  

true_changepoints = []  
distribution_sequence = [0] 

eta_0_val = 0.35
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2
psi_val = 0.45 

# --- 2. Ultra-Lightweight Data Storage ---
false_alarms_per_run = []
all_fa_intervals = [] 
time_to_first_fa = [] 

print(f"Starting {num_runs} Monte Carlo simulations (Length: {Total_Time} steps)...")
print("Memory optimization active: Continuous tracking disabled.")

# --- 3. Main Monte Carlo Loop ---
for run in tqdm(range(num_runs), desc="Simulation Progress"):
    
    raw_data = generate_alternating_wishart_series(
        total_steps=Total_Time, 
        changepoints=true_changepoints, 
        distribution_sequence=distribution_sequence, 
        dim=d, 
        df=10
    )
    
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
    
    for t in range(Total_Time):
        S_new = raw_data[t]
        nougat.step(t, S_new) 
            
    cps = nougat.global_changepoints
    false_alarms_per_run.append(len(cps))
    
    if len(cps) > 0:
        time_to_first_fa.append(cps[0])
        intervals = np.diff([0] + cps) 
        all_fa_intervals.extend(intervals)


# --- 4. Compute Statistical Aggregations ---
print("\nAggregating False Alarm statistics...")

mean_fa = np.mean(false_alarms_per_run)
std_fa = np.std(false_alarms_per_run)

print("\n--- Simulation Results (Stationary Data) ---")
print(f"Total Runs: {num_runs}")
print(f"Sequence Length: {Total_Time}")
print(f"False Alarms per sequence: Mean = {mean_fa:.3f}, Std = {std_fa:.3f}")

if all_fa_intervals:
    mean_time_between_fa = np.mean(all_fa_intervals)
    median_time_between_fa = np.median(all_fa_intervals)
    std_time_between_fa = np.std(all_fa_intervals)
    
    print("\n--- False Alarm Time Metrics ---")
    print(f"Total False Alarms Recorded: {len(all_fa_intervals)}")
    print(f"Mean Time Between False Alarms (MTBFA): {mean_time_between_fa:.2f} steps")
    print(f"Median Time Between False Alarms: {median_time_between_fa:.2f} steps")
else:
    print("\n--- False Alarm Time Metrics ---")
    print("Zero false alarms detected across all runs.")


# --- 5. Export Plotting Data to CSV ---
print("\nExporting data to CSV...")

if all_fa_intervals:
    df_fa = pd.DataFrame({'false_alarm_intervals': all_fa_intervals})
    df_fa.to_csv('fa_intervals_distribution.csv', index=False)
    
    df_summary = pd.DataFrame([{
        'Total_Runs': num_runs,
        'Sequence_Length': Total_Time,
        'Total_FA_Recorded': len(all_fa_intervals),
        'MTBFA_Mean': mean_time_between_fa,
        'MTBFA_Median': median_time_between_fa,
        'MTBFA_Std': std_time_between_fa
    }])
    df_summary.to_csv('fa_summary_statistics.csv', index=False)
    print("Saved CSV files.")

# --- 6. Visualizations & PDF Export ---
if all_fa_intervals:
    print("\nGenerating and saving plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_fa_intervals, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_fa_intervals), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean (MTBFA): {np.mean(all_fa_intervals):.0f}')
    ax.axvline(np.median(all_fa_intervals), color='blue', linestyle='dashed', linewidth=2, 
                label=f'Median: {np.median(all_fa_intervals):.0f}')

    ax.set_title(f'Distribution of Time Between False Alarms (MTBFA)\nAggregated over {num_runs} runs of {Total_Time} steps')
    ax.set_xlabel('Number of Time Steps Between Alarms')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig('fa_distribution_100k.pdf', format='pdf', bbox_inches='tight')
    print("Saved: fa_distribution_100k.pdf")
    # Note: No plt.show() here. It just saves and closes.
    plt.close()