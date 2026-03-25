import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt

def compute_single_spd(data, t, L, reg_epsilon=1e-5):
    """
    Computes a single SPD covariance matrix ending at time t.
    """
    d = data.shape[1]
    
    # Slice exactly L data points up to t
    window_data = data[t - L + 1 : t + 1]
    
    # Compute standard covariance
    cov_matrix = np.cov(window_data, rowvar=False)
    
    # Regularize to ensure strictly SPD
    spd_matrix = cov_matrix + np.eye(d) * reg_epsilon
    
    return spd_matrix

def compute_manifold_windows(data, N, L, t, reg_epsilon=1e-5):
    """
    Computes reference and test windows of SPD covariance matrices from a time series.
    
    Parameters:
    -----------
    data : np.ndarray
        The input time series data of shape (Total_Time, d), where d is the number of channels.
    N : int
        The number of covariance matrices in both the reference and test windows.
    L : int
        The number of data points used to compute each individual covariance matrix.
    t : int
        The current time step index (0-indexed).
    reg_epsilon : float
        A small regularization term added to the diagonal to ensure the matrices 
        are strictly Symmetric Positive-Definite (SPD).
        
    Returns:
    --------
    Sref : np.ndarray
        Reference manifold window of shape (N, d, d).
    Stest : np.ndarray
        Test manifold window of shape (N, d, d).
    """
    data = np.asarray(data)
    d = data.shape[1]
    
    # Calculate the starting index of the earliest required data point
    start_idx = t - 2 * N - L + 2
    
    if start_idx < 0:
        raise ValueError(f"Not enough data history. For t={t}, N={N}, L={L}, "
                         f"we need data starting from index {start_idx}, which is out of bounds.")

    # Initialize empty arrays to hold the sequence of covariance matrices
    Sref = np.zeros((N, d, d))
    Stest = np.zeros((N, d, d))
    
    # Helper function to compute a single SPD matrix at time tau
    def get_spd_cov(tau):
        # Slice the L data points: from tau - L + 1 up to tau (inclusive)
        window_data = data[tau - L + 1 : tau + 1]
        
        # np.cov expects features as rows if rowvar=True, so we set rowvar=False
        cov_matrix = np.cov(window_data, rowvar=False)
        
        # Regularize to ensure strict positive-definiteness
        spd_matrix = cov_matrix + np.eye(d) * reg_epsilon
        return spd_matrix

    # 1. Compute Sref: tau goes from (t - 2N + 1) to (t - N)
    for i, tau in enumerate(range(t - 2 * N + 1, t - N + 1)):
        Sref[i] = get_spd_cov(tau)
        
    # 2. Compute Stest: tau goes from (t - N + 1) to t
    for i, tau in enumerate(range(t - N + 1, t + 1)):
        Stest[i] = get_spd_cov(tau)
        
    return Sref, Stest

class SPD_NOUGAT:
    def __init__(self, mu, initial_dictionary, nu, eta_0, xi, sigma):
        """
        Initialize the NOUGAT algorithm for SPD matrices.
        initial_dictionary expects a 3D numpy array of shape (L, d, d)
        """
        self.mu = mu
        self.nu = nu
        self.eta_0 = eta_0
        self.xi = xi
        self.sigma = sigma
        
        # Dictionary D is a 3D array: L matrices, each of size d x d
        self.D = np.array(initial_dictionary) 
        self.L = self.D.shape[0]
        self.theta = np.zeros(self.L)
        
        self.changepoints = []

    def _kernel_LE_dictionary(self, S_i):
        """
        Compute the Log-Euclidean kernel between the dictionary and a sequence of SPD matrices.

        """
        kernel_values = np.zeros(self.D.shape[0])
        for i in range(self.D.shape[0]):
            log_D = logm(self.D[i])
            log_S = logm(S_i)
            distance = np.linalg.norm(log_S - log_D, 'fro')
            kernel_values[i]  = np.exp(-distance**2 / (2 * self.sigma**2))
        return kernel_values

    def _h_window(self, S):
        """
        Compute the h-window function (average kernel vector) for a sequence of SPD matrices.
        """
        h_sum = np.zeros(self.D.shape[0]) 

        for i in range(S.shape[0]):
            h_sum += self._kernel_LE_dictionary(S[i])

        h_result = h_sum / S.shape[0] 

        return h_result


    def _H_window(self, Sref):
        """
        Compute the H_ref matrix (average outer product of kernel vectors) 
        for the reference window of SPD matrices.
        """
        H_sum = np.zeros((self.D.shape[0], self.D.shape[0]))

        for i in range(Sref.shape[0]):
            k_vec = self._kernel_LE_dictionary(Sref[i])
            H_sum += np.outer(k_vec, k_vec)

        H_result = H_sum / Sref.shape[0]

        return H_result
    
    def step(self, t, S_ref, S_test, S_new):
        """
        Executes one time step (t) of the NOUGAT algorithm.
        S_new is the new SPD matrix observation (shape d x d).
        S_ref and S_test are the current sliding windows (shape N x d x d).
        """
        # 1. Compute kernel of the new observation against current dictionary
        k_S_new = self._kernel_LE_dictionary(S_new)
        max_k = np.max(np.abs(k_S_new)) # Coherence measure for the new observation against the dictionary
        
        # 2. Dictionary Update Logic
        if max_k > self.eta_0:
            # Dictionary remains unchanged. theta remains the same size.
            pass 
        else:
            # Add S_new to dictionary (Expand D from L to L+1)
            # We use np.vstack to add the d x d matrix to the L x d x d dictionary
            self.D = np.vstack((self.D, [S_new]))
            self.L += 1
            
            # Pad theta with a zero at the end
            self.theta = np.append(self.theta, 0.0)

        # 3. Compute H, h, and e using the *current* (possibly expanded) dictionary
        # This resolves the dimension mismatch in Step 10 mathematically cleanly.
        H_ref = self._H_window(S_ref)
        h_test = self._h_window(S_test)
        h_ref = self._h_window(S_ref)
        
        e_circ = h_ref - h_test # Step 3 (returns a vector of size L)
        
        # 4. Gradient Update for theta (Steps 6 and 10 unified)
        identity = np.eye(self.L)
        gradient = np.dot((H_ref + self.nu * identity), self.theta) + e_circ
        self.theta = self.theta - self.mu * gradient
        
        # 5. Compute test statistic 
        g = np.dot(self.theta.T, h_test)

        print(f"Time {t}: max_k={max_k:.4f}, g={g:.4f}, dictionary size={self.L}")
        
        # 6. Check for change point 
        if abs(g + 1) > self.xi:
            self.changepoints.append(t + 1)
            
        return g

