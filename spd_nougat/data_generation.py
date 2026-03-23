import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
Total_Time = 1000
d = 3
change_point = 500

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

# --- 4. Plot the Time Series ---
plt.figure(figsize=(12, 4))
plt.plot(raw_data, alpha=0.7)
plt.axvline(x=change_point, color='black', linestyle='--', linewidth=2, label='Mean Shift (t=500)')
plt.title('Synthetic Time Series with a Mean Shift')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()