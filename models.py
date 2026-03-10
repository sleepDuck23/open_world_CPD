import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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


def compute_rulsif_score(X_ref, X_test, alpha=0.1, sigma=1.0, lambda_=0.01):
    """
    Computes the alpha-Relative Pearson Divergence between two sets of samples.
    """
    n_ref = X_ref.shape[0]
    n_test = X_test.shape[0]
    
    # Step 1: Define basis function centers
    # We use the test samples as the centers for our Gaussian kernels
    centers = X_test 
    b = centers.shape[0]
    
    # Step 2: Compute Gaussian Kernel Matrices (Phi)
    # cdist computes the squared euclidean distance between every pair of points
    dist_ref = cdist(X_ref, centers, metric='sqeuclidean')
    Phi_ref = np.exp(-dist_ref / (2.0 * sigma**2))
    
    dist_test = cdist(X_test, centers, metric='sqeuclidean')
    Phi_test = np.exp(-dist_test / (2.0 * sigma**2))
    
    # Step 3: Construct the Empirical Matrices (H_hat and h_hat)
    H_hat_ref = np.dot(Phi_ref.T, Phi_ref) / n_ref
    H_hat_test = np.dot(Phi_test.T, Phi_test) / n_test
    
    # H_hat represents the mixture distribution q_alpha
    H_hat = (1.0 - alpha) * H_hat_ref + alpha * H_hat_test
    
    # h_hat represents the test distribution p_test
    h_hat = np.mean(Phi_test, axis=0)
    
    # Step 4: Solve for theta analytically
    # (H_hat + lambda * I) * theta = h_hat
    I = np.eye(b)
    theta = np.linalg.solve(H_hat + lambda_ * I, h_hat)
    
    # Optional but recommended: Density ratios shouldn't be negative
    theta = np.maximum(theta, 0)
    
    # Step 5: Compute the Relative Pearson Divergence score
    score = 0.5 * np.dot(theta, h_hat) - 0.5
    
    return max(0, score) # Divergence is strictly non-negative

def detect_rulsif(data, window_size=20, step=1, alpha=0.1, sigma=1.0, lambda_=0.01):
    """
    Slides two adjacent windows across the time series to detect distribution changes.
    """
    # Ensure data is 2D for scipy's cdist (samples x features)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    N = len(data)
    scores = np.zeros(N)
    
    # Slide a pair of adjacent windows across the data
    # [ --- X_ref --- ][ --- X_test --- ]
    for t in range(window_size, N - window_size, step):
        X_ref = data[t - window_size : t]
        X_test = data[t : t + window_size]
        
        # Calculate divergence between the past window and current window
        score = compute_rulsif_score(X_ref, X_test, alpha, sigma, lambda_)
        
        # The score is recorded at the boundary point 't'
        scores[t] = score
        
    return scores