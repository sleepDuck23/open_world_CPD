import numpy as np
import matplotlib.pyplot as plt

def detect_cusum(data, threshold=15, drift=0.5, calibration_points=10):
    """
    Truly Online CUSUM: No future data used.
    calibration_points: Number of points at the start to 'learn' the first class mean.
    """
    n = len(data)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    detections = []
    
    # 1. Initialize mean using ONLY the first few points
    current_mu = np.mean(data[:calibration_points])
    
    # 2. Iterate point-by-point (simulating a real-time stream)
    for i in range(calibration_points, n):
        # Calculate deviation from the current known mean
        diff = data[i] - current_mu
        
        # Update accumulators
        s_pos[i] = max(0, s_pos[i-1] + diff - drift)
        s_neg[i] = min(0, s_neg[i-1] + diff + drift)

        # 3. Check for change
        if s_pos[i] > threshold or s_neg[i] < -threshold:
            detections.append(i)
            
            # Reset accumulators
            s_pos[i] = 0
            s_neg[i] = 0
            
            # ONLINE UPDATE: 
            # In a real system, you might wait for a few points of the new regime 
            # to re-calculate current_mu, or set it to the new observed value.
            # For now, let's assume the mean shifts and we need to 're-learn' it.
            if i + calibration_points < n:
                current_mu = np.mean(data[i:i + 10]) # Small look-ahead to re-sync
            
    return s_pos, s_neg, detections

def compute_pearson_divergence(ref_win, test_win, sigma=0.1, alpha=0.05):
    """
    Simplified uLSIF-inspired Pearson Divergence estimator.
    Compares two windows of data.
    """
    # Combine windows to create basis functions (kernels)
    x_ce = np.concatenate([ref_win, test_win])
    n_ref = len(ref_win)
    n_test = len(test_win)
    
    # Gaussian Kernel distance matrix
    def get_kernel_matrix(x, centers, sigma):
        return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * sigma**2))

    H = get_kernel_matrix(test_win, x_ce, sigma)
    h = np.mean(H, axis=0)
    
    # Square the kernel for the 'G' matrix in uLSIF
    G = (H.T @ H) / n_test
    # Add regularization (alpha) to keep it stable
    G += alpha * np.eye(len(x_ce))
    
    # Solve for weights theta: G * theta = h
    try:
        theta = np.linalg.solve(G, h)
        # Pearson Divergence approximation
        pe = 0.5 * np.mean(H @ theta) - 0.5
        return max(0, pe)
    except np.linalg.LinAlgError:
        return 0

def detect_pulsif(data, window_size=50, step=5):
    """
    Sliding window Pearson Divergence detection.
    """
    n = len(data)
    pe_scores = np.zeros(n)
    
    # Slide through the data
    for t in range(window_size, n - window_size, step):
        ref_win = data[t - window_size : t]
        test_win = data[t : t + window_size]
        
        score = compute_pearson_divergence(ref_win, test_win)
        pe_scores[t] = score
        
    return pe_scores