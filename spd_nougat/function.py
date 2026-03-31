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
    
def warm_start_dict(warmup_series, eta_0, sigma):
    """
    Builds a sparse initial dictionary from a pre-training window.
    Highly optimized: caches matrix logarithms and vectorizes distance calculations
    to avoid redundant scipy.linalg.logm calls.
    """
   
    dictionary = [warmup_series[0].copy()]
    
    log_dict = [np.real(logm(warmup_series[0]))]
    
   
    for S in warmup_series[1:]:
        
        log_S = np.real(logm(S))
        
        log_D_array = np.array(log_dict)
        
        diffs = log_D_array - log_S
        
        sq_distances = np.sum(diffs**2, axis=(1, 2))
       
        kernel_values = np.exp(-sq_distances / (2 * sigma**2))
        
        max_k = np.max(kernel_values)
        
        if max_k <= eta_0:
            
            dictionary.append(S)
            
            log_dict.append(log_S)
            
    return np.array(dictionary)

class SPD_NOUGAT:
    def __init__(self, mu, initial_dictionary, nu, eta_0, xi, sigma, cooldown_period):
        """
        Initialize the NOUGAT algorithm with internal dictionary library management.
        """
        self.mu = mu
        self.nu = nu
        self.eta_0 = eta_0
        self.xi = xi
        self.sigma = sigma
        
        # Dictionary state
        self.D = np.array(initial_dictionary) 
        self.L = self.D.shape[0]
        self.theta = np.zeros(self.L)
        
        # --- NEW: Internal state management ---
        self.dictionary_library = []
        self.global_changepoints = []
        self.cooldown_period = cooldown_period
        self.cooldown_counter = 0

    def _warm_start_dict(self, warmup_series):
        """
        Internal method to build a sparse initial dictionary from a flushed window.
        """
        dictionary = [warmup_series[0].copy()]
        log_dict = [np.real(logm(warmup_series[0]))]
        
        for S in warmup_series[1:]:
            log_S = np.real(logm(S))
            log_D_array = np.array(log_dict)
            diffs = log_D_array - log_S
            sq_distances = np.sum(diffs**2, axis=(1, 2))
            kernel_values = np.exp(-sq_distances / (2 * self.sigma**2))
            
            if np.max(kernel_values) <= self.eta_0:
                dictionary.append(S)
                log_dict.append(log_S)
                
        return np.array(dictionary)

    def _kernel_LE_dictionary(self, S_i):
        kernel_values = np.zeros(self.D.shape[0])
        for i in range(self.D.shape[0]):
            log_D = logm(self.D[i])
            log_S = logm(S_i)
            distance = np.linalg.norm(log_S - log_D, 'fro')
            kernel_values[i]  = np.exp(-distance**2 / (2 * self.sigma**2))
        return kernel_values

    def _h_window(self, S):
        h_sum = np.zeros(self.D.shape[0]) 
        for i in range(S.shape[0]):
            h_sum += self._kernel_LE_dictionary(S[i])
        return h_sum / S.shape[0] 

    def _H_window(self, Sref):
        H_sum = np.zeros((self.D.shape[0], self.D.shape[0]))
        for i in range(Sref.shape[0]):
            k_vec = self._kernel_LE_dictionary(Sref[i])
            H_sum += np.outer(k_vec, k_vec)
        return H_sum / Sref.shape[0]
    
    def step(self, t, S_ref, S_test, S_new):
        """
        Executes one time step. Handles cooldown and dictionary saving internally.
        """
        # --- PHASE 1: Stabilization / Cooldown ---
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
            # If cooldown just finished, execute the warm start
            if self.cooldown_counter == 0:
                # S_ref is now fully flushed and represents the new distribution
                self.D = self._warm_start_dict(S_ref)
                self.L = self.D.shape[0]
                self.theta = np.zeros(self.L)
                print(f"Time {t}: Warm restart complete. New dict size = {self.L}")
                
            return np.nan # Return NaN for the statistic plot during stabilization
            
        # --- PHASE 2: Active Detection ---
        k_S_new = self._kernel_LE_dictionary(S_test[0])
        max_k = np.max(np.abs(k_S_new)) 
        
        if max_k <= self.eta_0:
            self.D = np.vstack((self.D, [S_new]))
            self.L += 1
            self.theta = np.append(self.theta, 0.0)
            print(f"Time {t}: Added new matrix to dictionary. New size = {self.L}")

        H_ref = self._H_window(S_ref)
        h_test = self._h_window(S_test)
        h_ref = self._h_window(S_ref)
        
        e_circ = h_ref - h_test 
        
        identity = np.eye(self.L)
        gradient = np.dot((H_ref + self.nu * identity), self.theta) + e_circ
        self.theta = self.theta - self.mu * gradient
        
        g = np.dot(self.theta.T, h_test)
        
        # --- PHASE 3: Check for Change Point ---
        if abs(g) > self.xi:
            print(f"Time {t}: *** CHANGE DETECTED *** (g={g:.4f})")
            
            self.global_changepoints.append(t + 1)
            
            # Save current dictionary to the library
            self.dictionary_library.append(self.D.copy())
            
            # Trigger cooldown for the next steps
            self.cooldown_counter = self.cooldown_period
            
        return g

    def finalize(self):
        """Call this at the very end of your time series to save the last active dictionary."""
        if self.cooldown_counter == 0:
            self.dictionary_library.append(self.D.copy())