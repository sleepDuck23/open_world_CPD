import numpy as np
import matplotlib.pyplot as plt
from function import SPD_NOUGAT_OnlineOpenWorld, warm_start_dict 
from spd_generation import generate_alternating_wishart_series
import matplotlib.patches as mpatches

# --- 1. Setup & Hyperparameters ---
np.random.seed(42)
Total_Time = 10000
d = 3
N_window = 20  

# True Change points and the alternating states we want to test
true_changepoints = [2000, 4000, 6000, 8000]  
distribution_sequence = [0, 1, 0, 2, 0] 

eta_0_val = 0.35
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2
psi_val = 0.45 # New Open-World Coherence Threshold (tune this based on your data)

# --- 2. Data Generation ---
raw_data = generate_alternating_wishart_series(
    total_steps=Total_Time, 
    changepoints=true_changepoints, 
    distribution_sequence=distribution_sequence, 
    dim=d, 
    df=10
)

# --- 3. Initialization ---
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

# --- 4. Tracking Variables ---
g_statistics = []
dic_sizes = []
active_states = []
fsm_states = [] # To track "ACTIVE", "CLASSIFYING", "WARMING_UP"
time_indices = []

print("Starting Online Open-World SPD NOUGAT...")

# --- 5. Main Online Loop ---
for t in range(Total_Time):
    S_new = raw_data[t]
    
    # Process one matrix at a time
    g = nougat.step(t, S_new)
    
    g_statistics.append(g)
    time_indices.append(t)
    fsm_states.append(nougat.state)
    
    # Track metrics based on the FSM State
    if nougat.state == "ACTIVE":
        dic_sizes.append(nougat.L)
        active_states.append(nougat.current_state_id)
    else:
        dic_sizes.append(np.nan)
        active_states.append(np.nan) # State is undefined during classification/warmup

print("Algorithm finished.")
print(f"Total Dictionaries in Library: {len(nougat.dictionary_library)}")
print(f"Detected Change Points: {nougat.global_changepoints}")


# --- 6. Visualization ---
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Helper function to draw background shades for FSM States
def highlight_fsm_states(axis):
    for t_idx in range(Total_Time):
        if fsm_states[t_idx] == "CLASSIFYING":
            axis.axvspan(t_idx, t_idx+1, color='yellow', alpha=0.3, lw=0)
        elif fsm_states[t_idx] == "WARMING_UP":
            axis.axvspan(t_idx, t_idx+1, color='red', alpha=0.2, lw=0)

# Create custom legend patches for the FSM background colors
classifying_patch = mpatches.Patch(color='yellow', alpha=0.3, label='Classifying Phase')
warmup_patch = mpatches.Patch(color='red', alpha=0.2, label='Warm-up Phase')

# Plot 1: Raw Time Series (Trace)
traces = [np.trace(mat) for mat in raw_data]
ax[0].plot(traces, alpha=0.7, color='teal', label='Trace of SPD Matrix')
for tcp in true_changepoints:
    # Only label the first vertical line to avoid legend duplication
    label_str = 'True Changepoint' if tcp == true_changepoints[0] else ""
    ax[0].axvline(tcp, color='black', linestyle='--', linewidth=2, label=label_str)
ax[0].set_title('Time Series (Trace of SPD Matrices) & True Changepoints')
ax[0].set_ylabel('Trace Value')
ax[0].legend(loc='upper left')

# Plot 2: NOUGAT Statistic (g_t)
ax[1].plot(time_indices, g_statistics, color='blue', label='NOUGAT Statistic ($g_t$)')
ax[1].axhline(nougat.xi, color='red', linestyle='--', label='Detection Threshold ($+\\xi$)')
highlight_fsm_states(ax[1])

# Combine line legends with our custom background patch legends
handles, labels = ax[1].get_legend_handles_labels()
handles.extend([classifying_patch, warmup_patch])
ax[1].legend(handles=handles, loc='upper left')

ax[1].set_title('NOUGAT Statistic ($g_t$)')
ax[1].set_ylabel('$g_t$')

# Plot 3: Dictionary Size Evolution
ax[2].plot(time_indices, dic_sizes, color='purple', linestyle='-', label='Dictionary Size ($L$)')
highlight_fsm_states(ax[2])

handles, labels = ax[2].get_legend_handles_labels()
handles.extend([classifying_patch, warmup_patch])
ax[2].legend(handles=handles, loc='upper left')

ax[2].set_title('Active Dictionary Size ($L$)')
ax[2].set_ylabel('Size')

# Plot 4: Active Classification State (True vs Predicted)
# 1. Ground truth step plot using existing variables directly
ax[3].step(
    [0] + true_changepoints + [Total_Time], 
    distribution_sequence + [distribution_sequence[-1]], 
    color='gray', alpha=0.6, linewidth=3, where='post', label='True System State'
)

# 2. Predicted states scatter plot
ax[3].scatter(time_indices, active_states, color='darkgreen', s=15, 
              label='Recognized State (Model)')

for tcp in true_changepoints:
    ax[3].axvline(tcp, color='black', linestyle='--', linewidth=1, alpha=0.5)

highlight_fsm_states(ax[3])

handles, labels = ax[3].get_legend_handles_labels()
handles.extend([classifying_patch, warmup_patch])
ax[3].legend(handles=handles, loc='upper left')

ax[3].set_title('True State vs. Recognized System State')
ax[3].set_xlabel('Time Step ($t$)')
ax[3].set_ylabel('State ID')

# Ensure the y-axis uses integers and accommodates all states
max_state = max(distribution_sequence) + 2
ax[3].set_yticks(range(max_state)) 

# --- Add Grid to all subplots ---
for axes in ax:
    axes.grid(True, linestyle='--', alpha=0.5, color='gray')

plt.tight_layout()
plt.show()