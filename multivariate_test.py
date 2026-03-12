import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from matplotlib.colors import ListedColormap

from models import detect_rulsif, detect_log_euclidean_kernel
from generate_signal import  generate_ssm_timeseries
from visual_manif import visualize_spd_manifold, visualize_manifold_with_ground_truth

# --- 1. Generate the Multivariate Data ---
num_channels = 3
num_classes = 3
num_changes = 3
num_data_points = 2000

# (Assuming generate_ssm_timeseries, detect_rulsif, and detect_log_euclidean_kernel are loaded)
data, labels, classes = generate_ssm_timeseries(num_channels, num_classes, num_changes, num_data_points)

# Find true change boundaries
true_changes = np.where(np.diff(labels) != 0)[0]
boundaries = [0] + list(true_changes) + [num_data_points]

# --- 2. Run Detectors ---
window_size = 40

# RULSIF
rulsif_scores = detect_rulsif(data, window_size=window_size, step=1, alpha=0.1, sigma=2.0, lambda_=0.05)
rulsif_threshold = np.median(rulsif_scores) + (5 * median_abs_deviation(rulsif_scores))
rulsif_detected, _ = find_peaks(rulsif_scores, height=rulsif_threshold, distance=window_size)

# Log-Euclidean Kernel
le_scores = detect_log_euclidean_kernel(data, window_size=window_size, step=1, sigma=2.0)
le_threshold = np.median(le_scores) + (5 * median_abs_deviation(le_scores))
le_detections, _ = find_peaks(le_scores, height=le_threshold, distance=window_size)

# --- 3. Advanced Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
cmap = ListedColormap(['#e4f1fe', '#e8f7e2', '#fdf2e9', '#f4e9fd', '#fde9ec']) # Soft background colors

# Helper function to draw background regimes and true change lines
def format_axis(ax, title, ylabel):
    # Shade the background based on the active class
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        class_id = labels[start]
        ax.axvspan(start, end, color=cmap(class_id % cmap.N), alpha=0.6, lw=0)
    
    # Draw solid vertical lines at the exact change points
    for tc in true_changes:
        ax.axvline(tc, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.4)

# Plot A: The Multivariate Signal
format_axis(axes[0], f"Multivariate SSM Data ({num_channels} Channels)", "Amplitude")
for i in range(num_channels):
    # Offset channels for clarity
    axes[0].plot(data[:, i] + (i * 4), linewidth=1, alpha=0.8, label=f'Channel {i+1}')
axes[0].legend(loc='upper right', bbox_to_anchor=(1.12, 1))

# Plot B: RULSIF
format_axis(axes[1], "RULSIF (Density Ratio Estimation)", "Divergence")
axes[1].plot(rulsif_scores, color='purple', linewidth=1.5, label='Pearson Divergence')
axes[1].axhline(rulsif_threshold, color='red', linestyle=':', linewidth=2, label='Dynamic Threshold')
axes[1].plot(rulsif_detected, rulsif_scores[rulsif_detected], "v", color='red', markersize=8, label='Detection')
axes[1].legend(loc='upper right', bbox_to_anchor=(1.12, 1))

# Plot C: Log-Euclidean Kernel
format_axis(axes[2], "Log-Euclidean Kernel (Manifold Geometry)", "1 - K_LE")
axes[2].plot(le_scores, color='orange', linewidth=1.5, label='Dissimilarity')
axes[2].axhline(le_threshold, color='red', linestyle=':', linewidth=2, label='Dynamic Threshold')
axes[2].plot(le_detections, le_scores[le_detections], "v", color='red', markersize=8, label='Detection')
axes[2].legend(loc='upper right', bbox_to_anchor=(1.12, 1))

plt.xlabel("Time Step", fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

visualize_spd_manifold(data, labels, window_size=window_size)
visualize_manifold_with_ground_truth(data, labels, classes, window_size=window_size)