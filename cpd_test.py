import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from matplotlib.colors import ListedColormap
from generate_signal import generate_ssm_timeseries
from models import OnlineManifoldCPD

# --- 1. Generate the Multivariate Data ---
num_channels = 3
num_classes = 3
num_changes = 3
num_data_points = 5000
min_spacing = 100

data, labels, classes = generate_ssm_timeseries(num_channels, num_classes, num_changes, num_data_points, min_spacing)
true_changes = np.where(np.diff(labels) != 0)[0]
true_order = [int(labels[0])] + [int(labels[tc + 1]) for tc in true_changes]
print(f"True order of classes: {true_order}")

# --- 2. Initialize the Online Detector ---
window_size = 50
detector = OnlineManifoldCPD(
    window_size=window_size, 
    buffer_size=50, 
    threshold_multiplier=10.0, 
    cluster_radius=0.5,
    sigma=2.0
)

# --- 3. Storage for Online Visualization ---
online_distances = []
dynamic_thresholds = []
detected_changepoints = []
active_clusters = np.full(num_data_points, -1) # -1 means "Unknown/Warming up"
current_cluster = -1

print("Starting streaming simulation...")

# --- 4. The Real-Time Simulation Loop ---
for t in range(num_data_points):
    # Fetch exactly one point (simulating a sensor reading)
    x_t = data[t, :]
    
    # Process it through the state machine
    dist, event = detector.process_next_point(x_t)
    
    # Store distance (use 0 during the warmup phase)
    online_distances.append(dist if dist is not None else 0)
    
    # Manually compute the current dynamic threshold strictly for plotting
    if len(detector.dist_buffer) > 0:
        med = np.median(detector.dist_buffer)
        mad = median_abs_deviation(detector.dist_buffer)
        thresh = med + (detector.multiplier * mad)
    else:
        thresh = 0
    dynamic_thresholds.append(thresh)
    
    # Handle detected events
    if event and "Change at t=" in event:
        print(event)
        detected_changepoints.append(detector.max_peak_time)
        current_cluster = detector.state_labels[-1]
        
        # Backfill the cluster assignment to the actual change point
        active_clusters[detector.max_peak_time : t] = current_cluster
        
    # Keep tracking the current state
    if current_cluster != -1:
        active_clusters[t] = current_cluster

print("Simulation complete.")

# --- 5. Advanced Online Visualization ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
cmap = ListedColormap(['#e4f1fe', '#e8f7e2', '#fdf2e9', '#f4e9fd', '#fde9ec', '#eeeeee'])

# Plot A: The Multivariate Signal & Classified Regimes
axes[0].set_title(f"Online State Classification ({num_channels} Channels)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Amplitude")
for i in range(num_channels):
    axes[0].plot(data[:, i] + (i * 4), linewidth=1, alpha=0.8, label=f'Channel {i+1}')

# Shade background based on the ONLINE cluster assignments, not the ground truth
for t in range(num_data_points - 1):
    c_id = active_clusters[t]
    if c_id != -1: # Only shade if we have assigned a cluster
        axes[0].axvspan(t, t+1, color=cmap(c_id % cmap.N), alpha=0.5, lw=0)

# Draw ground truth lines for reference
for tc in true_changes:
    axes[0].axvline(tc, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='True Change' if tc == true_changes[0] else "")

axes[0].legend(loc='upper right', bbox_to_anchor=(1.12, 1))
axes[0].grid(True, linestyle='--', alpha=0.4)

# Plot B: The Online Distance & Dynamic Threshold
axes[1].set_title("Streaming Log-Euclidean Distance & Dynamic Baseline", fontsize=12, fontweight='bold')
axes[1].set_ylabel("1 - K_LE")
axes[1].plot(online_distances, color='orange', linewidth=1.5, label='Current Distance')
axes[1].plot(dynamic_thresholds, color='red', linestyle='-', linewidth=1.5, alpha=0.8, label='Dynamic Threshold')

# Mark the exact points where the system finalized a detection
for cp in detected_changepoints:
    axes[1].axvline(cp, color='purple', linestyle='-', linewidth=2)
    axes[1].plot(cp, online_distances[cp], "v", color='purple', markersize=8)

axes[1].legend(loc='upper right', bbox_to_anchor=(1.12, 1))
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.xlabel("Time Step", fontsize=12)
plt.tight_layout()
plt.show()