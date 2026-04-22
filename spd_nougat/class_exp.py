import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from function import SPD_NOUGAT_OnlineOpenWorld, warm_start_dict 
from spd_generation import generate_alternating_wishart_series

# --- 1. Setup & Hyperparameters ---
np.random.seed(42)
Total_Time = 12000
num_runs = 10000  

d = 3
N_window = 20  

# Sequence setup
true_changepoints = [2000, 4000, 6000, 8000, 10000]  
distribution_sequence = [0, 1, 0, 2, 1, 2] 

# Define the start and end of each distinct segment for our analysis
segment_boundaries = [0] + true_changepoints + [Total_Time]
num_segments = len(distribution_sequence)

eta_0_val = 0.35
sigma_val = 1.44 
nu_val = 1e-2  
mu_val = 1e-1
xi_val = 0.2
psi_val = 0.45 

# --- 2. Data Storage Matrices ---
# We no longer need to track the g_stat or dict size continuously in memory since we aren't plotting them
all_predicted_states = np.full((num_runs, Total_Time), np.nan) 

print(f"Starting {num_runs} Monte Carlo classifications (Length: {Total_Time} steps)...")

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
        
        if nougat.state == "ACTIVE":
            all_predicted_states[run, t] = nougat.current_state_id

# --- 4. Compute Segment-Based Classification Aggregations ---
print("\nAggregating classification matrices per sequence segment...")

# Find the maximum predicted state ID to size our matrix properly
max_pred_label = int(np.nanmax(all_predicted_states)) if not np.isnan(np.nanmax(all_predicted_states)) else 0
unique_pred_labels = np.arange(max_pred_label + 1)

# Matrix to hold the distribution of predictions for each segment
# Shape: (Number of Predicted States, Number of Segments)
segment_distributions = np.zeros((len(unique_pred_labels), num_segments))

for i in range(num_segments):
    start_idx = segment_boundaries[i]
    end_idx = segment_boundaries[i+1]
    
    # Extract all predictions made during this time block across all 10,000 runs
    segment_preds = all_predicted_states[:, start_idx:end_idx]
    
    # Filter out the NaNs (transition periods)
    valid_preds = segment_preds[~np.isnan(segment_preds)].astype(int)
    
    if len(valid_preds) > 0:
        # Count occurrences of each state and normalize to a percentage (0 to 1)
        counts = np.bincount(valid_preds, minlength=len(unique_pred_labels))
        segment_distributions[:, i] = counts / len(valid_preds)

# --- 5. Export Plotting Data to CSV ---
print("Exporting data to CSV...")

# Create column names that indicate the sequence order and the ground truth state
segment_names = [f"True State {distribution_sequence[i]}" for i in range(num_segments)]
pred_names = [f"Predicted State {l}" for l in unique_pred_labels]

df_segments = pd.DataFrame(segment_distributions, index=pred_names, columns=segment_names)
df_segments.to_csv('class_segment_distributions.csv')
print("Saved: class_segment_distributions.csv")

# --- 6. Visualizations & PDF Export ---
print("\nGenerating and saving plots...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the matrix as a crisp, discrete heatmap
cax = ax.imshow(segment_distributions, aspect='auto', cmap='Blues', vmin=0, vmax=1)

# Configure axes
ax.set_xticks(np.arange(num_segments))
ax.set_yticks(np.arange(len(unique_pred_labels)))
ax.set_xticklabels(segment_names)
ax.set_yticklabels(pred_names)
ax.set_title(f"Classification Distribution per Sequence Phase\n(Aggregated over {num_runs} runs)")

# Add grid lines to separate the discrete blocks clearly
ax.set_xticks(np.arange(-.5, num_segments, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(unique_pred_labels), 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Loop over data dimensions and create text annotations in every box
for i in range(len(unique_pred_labels)):
    for j in range(num_segments):
        val = segment_distributions[i, j]
        # Only print text if the value is greater than 0.5% to keep the plot clean
        if val > 0.005:
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=text_color, fontweight='bold')

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Percentage of Valid Classifications")

fig.tight_layout()
fig.savefig('class_segment_distributions.pdf', format='pdf', bbox_inches='tight')

plt.show()