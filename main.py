import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from models import detect_cusum, detect_rulsif, detect_log_euclidean_kernel
from generate_signal import generate_segmented_timeseries

# data parameters
K = 1000
num_classes = 2
num_changes = 3

data, labels, class_params = generate_segmented_timeseries(K, num_classes, num_changes)

# CUMSUM 
threshold = 100
s_pos, s_neg, cumsum_detected = detect_cusum(data, threshold, drift=0.5, calibration_points = 50)

# Rulsif
rulsif_scores = detect_rulsif(data, window_size=35, step=1, alpha=0.1, sigma=2, lambda_=0.05)

median_score = np.median(rulsif_scores)

mad = median_abs_deviation(rulsif_scores)

rulsif_threshold = median_score + (7 * mad)

rulsif_detected, _ = find_peaks(rulsif_scores, height=rulsif_threshold, distance=35)

# Log-Euclidean Kernel
le_scores = detect_log_euclidean_kernel(data, window_size=20, step=1, sigma=1.5)

# Dynamic thresholding 
le_median = np.median(le_scores)
le_mad = median_abs_deviation(le_scores)
le_threshold = le_median + (7 * le_mad)

# Extract peaks
le_detections, _ = find_peaks(le_scores, height=le_threshold, distance=35)


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
for dc in cumsum_detected:
    ax1.axvline(dc, color='red', linestyle='--', linewidth=2, label='CUSUM Detection' if dc == cumsum_detected[0] else "")

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

# Top Plot: Signal, True Changes, AND Detected Changes
ax1.plot(data, color='gray', alpha=0.6, label='Signal')

# Draw True Changes (Blue)
true_changes = np.where(np.diff(labels) != 0)[0]
for tc in true_changes:
    ax1.axvline(tc, color='blue', linestyle='-', linewidth=2, label='True Change' if tc == true_changes[0] else "")

# Draw RULSIF Detections (Red)
for dc in rulsif_detected:
    ax1.axvline(dc, color='red', linestyle='--', linewidth=2, label='RULSIF Detection' if dc == rulsif_detected[0] else "")

ax1.set_title(f"Detection Performance: Blue (Actual) vs Red (RULSIF)")
ax1.legend(loc="upper right")

# Bottom Plot: RULSIF Divergence Score
ax2.plot(rulsif_scores, color='purple', linewidth=1.5, label='Relative Pearson Divergence')
ax2.axhline(rulsif_threshold, color='red', linestyle=':', label=f'Threshold ({rulsif_threshold:.2f})')

# Mark the specific peaks we found on the score line
ax2.plot(rulsif_detected, rulsif_scores[rulsif_detected], "x", color='red', markersize=8, label='Detected Peaks')

ax2.set_title("RULSIF Score (Spikes above threshold trigger a detection)")
ax2.set_ylim(0, np.percentile(rulsif_scores, 99.5) * 1.2) 
ax2.set_ylabel("Divergence Score")
ax2.legend(loc="upper right")

plt.tight_layout()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top Plot: Signal and Detections
ax1.plot(data, color='gray', alpha=0.6, label='Signal')

# Draw True Changes (Blue)
for tc in true_changes:
    ax1.axvline(tc, color='blue', linestyle='-', linewidth=2, label='True Change' if tc == true_changes[0] else "")

# Draw LE Detections (Red)
for dc in le_detections:
    ax1.axvline(dc, color='red', linestyle='--', linewidth=2, label='Log-Euclidean Detection' if dc == le_detections[0] else "")

ax1.set_title("Detection Performance: Blue (Actual) vs Red (Log-Euclidean)")
ax1.legend(loc="upper right")

# Bottom Plot: Kernel Dissimilarity Score
ax2.plot(le_scores, color='orange', linewidth=1.5, label='Kernel Dissimilarity (1 - K_LE)')
ax2.axhline(le_threshold, color='red', linestyle=':', label=f'Threshold ({le_threshold:.2f})')

# Mark the specific peaks we found
ax2.plot(le_detections, le_scores[le_detections], "x", color='red', markersize=8, label='Detected Peaks')

ax2.set_title("Log-Euclidean Kernel Score (Spikes above threshold trigger detection)")
ax2.set_ylim(0, 1.1) # Bounded between 0 and 1
ax2.set_ylabel("Dissimilarity")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()