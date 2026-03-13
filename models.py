import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh, expm
from scipy.stats import median_abs_deviation

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

def compute_spd_covariance(X, epsilon=1e-5):
    """
    Computes the covariance matrix and regularizes it to be strictly 
    Symmetric Positive Definite (SPD). Handles both 1D and 2D data.
    """
    # Ensure X is 2D for consistent np.cov behavior: (Samples, Features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Calculate covariance (rowvar=False means columns are variables/channels)
    S = np.cov(X, rowvar=False)
    
    # Handle the 1D case where np.cov returns a 0D scalar
    if S.ndim == 0:
        S = np.array([[S]])
        
    # Diagonal loading (regularization) to guarantee SPD
    S_spd = S + epsilon * np.eye(S.shape[0])
    return S_spd

def logm_spd(S):
    """
    Computes the principal matrix logarithm of an SPD matrix.
    """
    # Eigendecomposition (highly stable for symmetric matrices)
    evals, evecs = eigh(S)
    
    # Take the natural log of the eigenvalues 
    # (np.maximum prevents math errors if an evalue is microscopically close to 0)
    log_evals = np.diag(np.log(np.maximum(evals, 1e-12)))
    
    # Reconstruct the matrix in the Tangent Space
    return evecs @ log_evals @ evecs.T

def detect_log_euclidean_kernel(data, window_size=35, step=1, sigma=1.0):
    """
    Slides two adjacent windows across the time series, computes their SPD 
    covariance matrices, maps them to the Tangent Space, and calculates 
    the Log-Euclidean Gaussian Kernel difference.
    """
    N = len(data)
    kernel_dissimilarities = np.zeros(N)
    
    # Slide a pair of adjacent windows: [ --- Past --- ][ --- Future --- ]
    for t in range(window_size, N - window_size, step):
        X_past = data[t - window_size : t]
        X_future = data[t : t + window_size]
        
        # 1. Get SPD Covariances
        S_past = compute_spd_covariance(X_past)
        S_future = compute_spd_covariance(X_future)
        
        # 2. Map to flat Tangent Space via Matrix Logarithm
        log_S_past = logm_spd(S_past)
        log_S_future = logm_spd(S_future)
        
        # 3. Compute Log-Euclidean Distance Squared (Frobenius norm)
        diff = log_S_past - log_S_future
        d_le_squared = np.trace(diff @ diff.T) 
        
        # 4. Compute Gaussian Kernel Similarity (1 = identical, 0 = completely different)
        K_LE = np.exp(-d_le_squared / (2.0 * sigma**2))
        
        # 5. Store Dissimilarity (1 - K_LE) so peaks represent changes
        kernel_dissimilarities[t] = 1.0 - K_LE
        
    return kernel_dissimilarities

class OnlineManifoldCPD:
    def __init__(self, window_size=40, buffer_size=100, threshold_multiplier=5.0, cluster_radius=1.5, sigma=2.0):
        # Configuration
        self.W = window_size
        self.buffer_size = buffer_size
        self.multiplier = threshold_multiplier
        self.cluster_radius = cluster_radius
        self.sigma = sigma
        
        # Data Streams
        self.data_stream = []        # Holds recent raw data (size 2*W)
        self.cov_history = []        # Holds covariances of the CURRENT state
        self.dist_buffer = []        # History buffer of normal distances (for baseline)
        
        # State Machine Variables
        self.state = 'A_MONITORING'  # States: A_MONITORING, B_PEAK_HUNTING, C_COOLDOWN
        self.cooldown_counter = 0
        self.max_peak_val = -1
        self.max_peak_time = -1
        self.current_time = 0
        
        # Clustering Memory
        self.clusters = []           # List of dicts: {'N': int, 'tangent_mean': array, 'centroid': array}
        self.detected_changes = []
        self.state_labels = []       # Tracks which cluster ID is active at each change

    def expm_spd(self, M):
        """Helper: Maps tangent space matrix back to SPD Manifold."""
        evals, evecs = eigh(M)
        exp_evals = np.diag(np.exp(evals))
        return evecs @ exp_evals @ evecs.T

    def process_next_point(self, x):
        """Feeds one data point into the system and returns status."""
        self.current_time += 1
        self.data_stream.append(x)
        
        # Maintain only enough data for Past and Future windows
        if len(self.data_stream) > 2 * self.W:
            self.data_stream.pop(0)
            
        # 1. Warm-up Phase: Not enough data yet
        if len(self.data_stream) < 2 * self.W:
            return None, "Warming up"

        # 2. Compute Windows & Distance
        X_past = np.array(self.data_stream[:self.W])
        X_future = np.array(self.data_stream[self.W:])
        
        S_past = compute_spd_covariance(X_past)
        S_future = compute_spd_covariance(X_future)
        
        # Save past covariance for potential clustering later
        self.cov_history.append(S_past)
        
        log_S_past = logm_spd(S_past)
        log_S_future = logm_spd(S_future)
        
        diff = log_S_past - log_S_future
        d_le_squared = np.trace(diff @ diff.T)
        K_LE = np.exp(-d_le_squared / (2.0 * self.sigma**2))
        dist = 1.0 - K_LE # Current dissimilarity
        
        # 3. Dynamic Baseline Calculation
        if len(self.dist_buffer) < self.buffer_size:
            self.dist_buffer.append(dist)
            return dist, "Building Baseline"
            
        median_dist = np.median(self.dist_buffer)
        mad_dist = median_abs_deviation(self.dist_buffer)
        threshold = median_dist + (self.multiplier * mad_dist)

        # 4. The Peak Hunting State Machine
        event = None

        if self.state == 'A_MONITORING':
            if dist > threshold:
                # Spike detected! Enter State B. Do NOT add to dist_buffer.
                self.state = 'B_PEAK_HUNTING'
                self.max_peak_val = dist
                self.max_peak_time = self.current_time - self.W
            else:
                # Normal operation. Update baseline.
                self.dist_buffer.append(dist)
                self.dist_buffer.pop(0)

        elif self.state == 'B_PEAK_HUNTING':
            if dist > self.max_peak_val:
                # Still climbing the peak
                self.max_peak_val = dist
                self.max_peak_time = self.current_time - self.W
                
            elif dist < threshold: # The spike has ended
                # PEAK CONFIRMED!
                self.detected_changes.append(self.max_peak_time)
                
                # Classify the completed past state
                cluster_id = self._classify_and_update_state()
                event = f"Change at t={self.max_peak_time}, Assigned to Cluster {cluster_id}"
                
                # Reset for cooldown
                self.state = 'C_COOLDOWN'
                self.cooldown_counter = self.W
                self.cov_history = [] # Clear history for the new state

        elif self.state == 'C_COOLDOWN':
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = 'A_MONITORING'
                # Optional: clear baseline to learn the new regime quickly
                self.dist_buffer = [] 

        return dist, event

    def _classify_and_update_state(self):
        """Computes the Fréchet mean of the completed state and clusters it."""
        if not self.cov_history:
            return -1
            
        # 1. Compute Tangent-Space Mean for the recent state
        log_covs = [logm_spd(S) for S in self.cov_history]
        M_new = np.mean(log_covs, axis=0) # Arithmetic mean in tangent space
        m = len(log_covs)
        
        # 2. Check against known clusters
        best_dist = float('inf')
        best_cluster = -1
        
        for idx, cluster in enumerate(self.clusters):
            # Compute Log-Euclidean distance between new mean and cluster centroid
            diff = M_new - cluster['tangent_mean']
            dist = np.sqrt(np.trace(diff @ diff.T))
            
            if dist < best_dist:
                best_dist = dist
                best_cluster = idx
                
        # 3. Known vs Unknown
        if best_dist < self.cluster_radius:
            # KNOWN STATE: Incremental Update
            c = self.clusters[best_cluster]
            N_tot = c['N']
            # Weighted average in tangent space
            M_updated = ((N_tot * c['tangent_mean']) + (m * M_new)) / (N_tot + m)
            
            self.clusters[best_cluster]['tangent_mean'] = M_updated
            self.clusters[best_cluster]['centroid'] = self.expm_spd(M_updated)
            self.clusters[best_cluster]['N'] += m
            
            self.state_labels.append(best_cluster)
            return best_cluster
            
        else:
            # UNKNOWN STATE: Create new cluster
            new_cluster = {
                'N': m,
                'tangent_mean': M_new,
                'centroid': self.expm_spd(M_new)
            }
            self.clusters.append(new_cluster)
            new_id = len(self.clusters) - 1
            self.state_labels.append(new_id)
            return new_id