import numpy as np
import matplotlib.pyplot as plt
from cumsum import detect_cusum, detect_pulsif
from generate_signal import generate_segmented_timeseries

# data parameters
K = 2000
num_classes = 3
num_changes = 5

data, labels, class_params = generate_segmented_timeseries(K, num_classes, num_changes)

# CUMSUM 
threshold = 100
s_pos, s_neg, detected_indices = detect_cusum(data, threshold, drift=0.5, calibration_points = 50)

# Calculate PE Scores
pe_scores = detect_pulsif(data, window_size=10, step=1)

# Visualization data input
plt.figure(figsize=(12, 5))
plt.plot(data, color='gray', alpha=0.5, label='Signal')
plt.scatter(range(K), data, c=labels, cmap='viridis', s=10, label='Class Label')
plt.title(f"Gaussian Time Series: {num_changes} Changes among {num_classes} Classes")
plt.colorbar(label='Class ID')
plt.grid()


# --- Improved Detection Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top Plot: Compare Truth vs Detection
ax1.plot(data, color='gray', alpha=0.4, label='Signal')

# 1. Draw TRUE changes (where the generator actually switched classes)
# We find where the labels array changes value
true_changes = np.where(np.diff(labels) != 0)[0]
for tc in true_changes:
    ax1.axvline(tc, color='blue', linestyle='-', linewidth=2, label='True Change' if tc == true_changes[0] else "")

# 2. Draw DETECTED changes (where CUSUM hit the threshold)
for dc in detected_indices:
    ax1.axvline(dc, color='red', linestyle='--', linewidth=2, label='CUSUM Detection' if dc == detected_indices[0] else "")

ax1.set_title("Detection Performance: Blue (Actual) vs Red (CUSUM)")
ax1.legend()

# Bottom Plot: The "Accumulator"
ax2.plot(s_pos, color='green', label='Positive Accumulator')
ax2.plot(s_neg, color='blue', label='Negative Accumulator')
ax2.axhline(threshold, color='black', linestyle=':', label='Threshold')
ax2.axhline(-threshold, color='black', linestyle=':')
ax2.set_ylabel("Cumulative Deviation")
ax2.set_title("CUSUM Accumulation (Triggers when line hits dots)")
ax2.legend()

plt.tight_layout()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(data, color='gray', alpha=0.5)
true_changes = np.where(np.diff(labels) != 0)[0]
for tc in true_changes:
    ax1.axvline(tc, color='blue', label='True Change' if tc == true_changes[0] else "")
ax1.set_title("Signal and True Change Points")

ax2.plot(pe_scores, color='purple', label='Pearson Divergence (uLSIF)')
ax2.set_title("Pearson Divergence Score (Spikes = Changes)")
ax2.set_ylim(0, np.percentile(pe_scores, 99)*1.5) # Zoom in on relevant spikes
ax2.legend()

plt.tight_layout()
plt.show()