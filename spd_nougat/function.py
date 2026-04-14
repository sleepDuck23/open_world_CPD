import numpy as np
from scipy.linalg import logm, eigh
from collections import deque

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
        
        # Internal state management 
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
            kernel_values[i]  = np.exp(-(distance**2 / (2 * self.sigma**2)))
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
    
    def step(self, t, S_ref, S_test):
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
                
            return np.nan 
            
        # --- PHASE 2: Active Detection ---
        k_S_new = self._kernel_LE_dictionary(S_test[0])
        max_k = np.max(np.abs(k_S_new)) 
        
        if max_k <= self.eta_0:
            self.D = np.vstack((self.D, [S_test[0].copy()]))
            self.L += 1
            self.theta = np.append(self.theta, 0.0)
            print(f"Time {t}: Added new matrix to dictionary. New size = {self.L}")
            print(f"Time {t}: Max kernel value for new matrix = {max_k:.4f} (added to dictionary)")

        H_ref = self._H_window(S_ref)
        h_test = self._h_window(S_test)
        h_ref = self._h_window(S_ref)        
        e_circ = h_ref - h_test 

        
        identity = np.eye(self.L)
        gradient = (H_ref + self.nu * identity) @ self.theta + e_circ
        self.theta = self.theta - self.mu * gradient
        
        g = self.theta.T @ h_test
        
        # --- PHASE 3: Check for Change Point ---
        if g > self.xi:
            print(f"Time {t}: *** CHANGE DETECTED *** (g={g:.4f})")
            
            self.global_changepoints.append(t)

            # Calculate the individual terms of the dot product
            contributions = self.theta * h_test
            
            print(f"Individual Atom Contributions (Total L={self.L}):")
            for i, val in enumerate(contributions):
                # We print the index, the weight (theta), the kernel (h), and the product
                print(f"  Atom {i:2d}: term = {val:8.4f}  [theta={self.theta[i]:7.4f}, h_test={h_test[i]:7.4f}]")
            
            print("-" * 40)
            
            # Save current dictionary to the library
            self.dictionary_library.append(self.D.copy())
            
            # Trigger cooldown for the next steps
            self.cooldown_counter = self.cooldown_period
            
        return g

    def finalize(self):
        """Call this at the very end of your time series to save the last active dictionary."""
        if self.cooldown_counter == 0:
            self.dictionary_library.append(self.D.copy())



def fast_spd_logm(S, min_eigval=1e-10):
    """
    Computes the matrix logarithm up to 5x faster for SPD matrices 
    using Eigenvalue decomposition instead of Schur decomposition.
    """
    # eigh is highly optimized for symmetric matrices
    evals, evecs = eigh(S)
    
    # Clip eigenvalues to ensure strict positivity (numerical stability)
    evals = np.maximum(evals, min_eigval)
    
    # log(S) = V * log(Lambda) * V^T
    return (evecs * np.log(evals)) @ evecs.T

class SPD_NOUGAT_optimized:
    def __init__(self, mu, initial_dictionary, nu, eta_0, xi, sigma, cooldown_period, N):
        """
        N: The size of the reference and test windows.
        """
        self.mu = mu
        self.nu = nu
        self.eta_0 = eta_0
        self.xi = xi
        self.sigma = sigma
        self.N = N
        
        # 1. Map initial dictionary to tangent space
        self.log_D = np.array([fast_spd_logm(S) for S in initial_dictionary])
        self.L = self.log_D.shape[0]
        self.theta = np.zeros(self.L)
        
        # 2. Stateful Sliding Windows (storing log-domain matrices)
        self.window_test = deque(maxlen=N)
        self.window_ref = deque(maxlen=N)
        
        # Stateful Kernel Buffers: shape (N, L)
        # These cache the kernel evaluations so we don't recalculate them
        self.K_test = np.zeros((N, self.L)) 
        self.K_ref = np.zeros((N, self.L))
        self.buffer_filled = False # Flag to wait until 2N matrices are ingested
        
        # Tracking
        self.dictionary_library = []
        self.global_changepoints = []
        self.cooldown_period = cooldown_period
        self.cooldown_counter = 0

    def _eval_kernel_vector(self, log_S):
        """Vectorized kernel evaluation against the current dictionary."""
        diffs = self.log_D - log_S
        sq_distances = np.sum(diffs**2, axis=(1, 2))
        return np.exp(-sq_distances / (2 * self.sigma**2))

    def _expand_dictionary(self, log_S_new):
        """Handles the logic of expanding the dictionary and updating cached buffers."""
        # 1. Add new atom to dictionary
        self.log_D = np.vstack((self.log_D, [log_S_new.copy()]))
        self.L += 1
        self.theta = np.append(self.theta, 0.0)
        
        # 2. Update the Kernel Caches (K_ref, K_test) with a new column
        # We must evaluate the new dictionary atom against the existing historical windows
        new_col_test = np.array([np.exp(-np.sum((log_S - log_S_new)**2) / (2 * self.sigma**2)) 
                                 for log_S in self.window_test])
        new_col_ref = np.array([np.exp(-np.sum((log_S - log_S_new)**2) / (2 * self.sigma**2)) 
                                for log_S in self.window_ref])
        
        self.K_test = np.column_stack((self.K_test, new_col_test))
        self.K_ref = np.column_stack((self.K_ref, new_col_ref))

    def step(self, t, S_new):
        """
        Streaming step: Feed ONE new covariance matrix per time step.
        """
        # Convert immediately to Log-domain using the fast method
        log_S_new = fast_spd_logm(S_new)
        
        # --- Queue Management ---
        if len(self.window_test) == self.N:
            # Shift data: Oldest test matrix becomes newest ref matrix
            matrix_leaving_test = self.window_test.popleft()
            self.window_ref.append(matrix_leaving_test)
            
            # Shift kernel caches: roll rows upward
            k_leaving_test = self.K_test[0].copy()
            self.K_test = np.roll(self.K_test, -1, axis=0)
            
            if len(self.window_ref) == self.N:
                self.K_ref = np.roll(self.K_ref, -1, axis=0)
                self.K_ref[-1] = k_leaving_test
            else:
                self.K_ref[len(self.window_ref)-1] = k_leaving_test
        
        # Add newest matrix to test window
        self.window_test.append(log_S_new)
        
        # Evaluate kernel for new matrix and place at end of test cache
        k_new = self._eval_kernel_vector(log_S_new)
        self.K_test[len(self.window_test)-1] = k_new

        # Wait until we have 2N matrices to start detecting
        if len(self.window_ref) < self.N:
            return np.nan 

        # --- PHASE 1: Stabilization / Cooldown ---
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                # Rebuild dictionary from current reference window (flushed state)
                log_D_list = [self.window_ref[0]]
                for log_S in list(self.window_ref)[1:]:
                    diffs = np.array(log_D_list) - log_S
                    sq_distances = np.sum(diffs**2, axis=(1, 2))
                    kernel_values = np.exp(-sq_distances / (2 * self.sigma**2))
                    if np.max(kernel_values) <= self.eta_0:
                        log_D_list.append(log_S)
                
                self.log_D = np.array(log_D_list)
                self.L = self.log_D.shape[0]
                self.theta = np.zeros(self.L)
                
                # Re-initialize Kernel caches from scratch
                self.K_test = np.array([self._eval_kernel_vector(s) for s in self.window_test])
                self.K_ref = np.array([self._eval_kernel_vector(s) for s in self.window_ref])
                print(f"Time {t}: Warm restart complete. New dict size = {self.L}")
            return np.nan 
            
        # --- PHASE 2: Active Detection ---
        max_k = np.max(np.abs(k_new)) 
        if max_k <= self.eta_0:
            self._expand_dictionary(log_S_new)
            print(f"Time {t}: Added new matrix to dict. Size = {self.L}, max_k = {max_k:.4f}")

        # Ultra-fast O(1) expectations using cached matrices
        h_test = np.mean(self.K_test, axis=0)
        h_ref = np.mean(self.K_ref, axis=0)
        
        # Highly optimized Gram matrix computation
        H_ref = (self.K_ref.T @ self.K_ref) / self.N
        
        e_circ = h_ref - h_test 
        
        # Gradient Descent Step
        identity = np.eye(self.L)
        gradient = (H_ref + self.nu * identity) @ self.theta + e_circ
        self.theta -= self.mu * gradient
        
        g = np.dot(self.theta, h_test)
        
        # --- PHASE 3: Check for Change Point ---
        if g > self.xi:
            print(f"Time {t}: *** CHANGE DETECTED *** (g={g:.4f})")
            self.global_changepoints.append(t)
            self.dictionary_library.append(self.log_D.copy())
            self.cooldown_counter = self.cooldown_period
            
        return g
    def finalize(self):
        """Call this at the very end of your time series to save the last active dictionary."""
        if self.cooldown_counter == 0:
            self.dictionary_library.append(self.log_D.copy())

class SPD_NOUGAT_OnlineOpenWorld:
    def __init__(self, mu, initial_dictionary, nu, eta_0, xi, sigma, psi, N):
        self.mu = mu
        self.nu = nu
        self.eta_0 = eta_0
        self.xi = xi
        self.sigma = sigma
        self.psi = psi
        self.N = N  # This is 'm' in your equations
        
        # --- State Machine & Streaming Tracking ---
        self.state = "ACTIVE"  
        self.phase_step_count = 0  # Tracks progress through the N-step phases
        
        # --- Memory & Dictionaries ---
        self.dictionary_library = [] 
        self.global_changepoints = []
        self.last_active_id = -1   # To exclude the immediately previous state
        
        # Online Accumulator for Classification
        self.coherence_sums = {}
        
        # Online Builder for Warm-up
        self.warmup_dict_list = []
        
        self._init_active_state(initial_dictionary, state_id=0)
        
    def _init_active_state(self, dictionary, state_id):
        """Flushes buffers and loads a dictionary to resume active detection."""
        self.log_D = np.array([fast_spd_logm(S) if not isinstance(S, np.ndarray) else S for S in dictionary])
        self.L = self.log_D.shape[0]
        self.theta = np.zeros(self.L)
        
        self.current_state_id = state_id
        
        self.window_test = deque(maxlen=self.N)
        self.window_ref = deque(maxlen=self.N)
        self.K_test = np.zeros((self.N, self.L)) 
        self.K_ref = np.zeros((self.N, self.L))

    def _eval_kernel_vector(self, log_S):
        diffs = self.log_D - log_S
        sq_distances = np.sum(diffs**2, axis=(1, 2))
        return np.exp(-sq_distances / (2 * self.sigma**2))

    def _expand_dictionary(self, log_S_new):
        self.log_D = np.vstack((self.log_D, [log_S_new.copy()]))
        self.L += 1
        self.theta = np.append(self.theta, 0.0)
        
        new_col_test = np.array([np.exp(-np.sum((log_S - log_S_new)**2) / (2 * self.sigma**2)) 
                                 for log_S in self.window_test])
        new_col_ref = np.array([np.exp(-np.sum((log_S - log_S_new)**2) / (2 * self.sigma**2)) 
                                for log_S in self.window_ref])
        
        self.K_test = np.column_stack((self.K_test, new_col_test))
        self.K_ref = np.column_stack((self.K_ref, new_col_ref))

    def step(self, t, S_new):
        """Streaming entry point."""
        log_S_new = fast_spd_logm(S_new)
        
        if self.state == "ACTIVE":
            return self._step_active(t, log_S_new)
        elif self.state == "CLASSIFYING":
            return self._step_classifying(t, log_S_new)
        elif self.state == "WARMING_UP":
            return self._step_warming_up(t, log_S_new)

    def _step_active(self, t, log_S_new):
        """Phase 1: Online SPD-NOUGAT Detection."""
        # --- Queue Management (Same as before) ---
        if len(self.window_test) == self.N:
            matrix_leaving_test = self.window_test.popleft()
            self.window_ref.append(matrix_leaving_test)
            
            k_leaving_test = self.K_test[0].copy()
            self.K_test = np.roll(self.K_test, -1, axis=0)
            
            if len(self.window_ref) == self.N:
                self.K_ref = np.roll(self.K_ref, -1, axis=0)
                self.K_ref[-1] = k_leaving_test
            else:
                self.K_ref[len(self.window_ref)-1] = k_leaving_test
        
        self.window_test.append(log_S_new)
        k_new = self._eval_kernel_vector(log_S_new)
        self.K_test[len(self.window_test)-1] = k_new

        if len(self.window_ref) < self.N:
            return np.nan 

        # --- Detection Logic ---
        max_k = np.max(np.abs(k_new)) 
        if max_k <= self.eta_0:
            self._expand_dictionary(log_S_new)

        h_test = np.mean(self.K_test, axis=0)
        h_ref = np.mean(self.K_ref, axis=0)
        H_ref = (self.K_ref.T @ self.K_ref) / self.N
        
        e_circ = h_ref - h_test 
        gradient = (H_ref + self.nu * np.eye(self.L)) @ self.theta + e_circ
        self.theta -= self.mu * gradient
        
        g = np.dot(self.theta, h_test)
        
        # --- Changepoint Trigger ---
        if g > self.xi:
            print(f"Time {t}: *** CHANGE DETECTED *** (g={g:.4f})")
            self.global_changepoints.append(t)
            
            # Save current state to library if it's the first time, or update it
            if self.current_state_id >= len(self.dictionary_library):
                self.dictionary_library.append(self.log_D.copy())
            else:
                self.dictionary_library[self.current_state_id] = self.log_D.copy()
            
            self.last_active_id = self.current_state_id
            
            # Transition to Online Classification
            self.state = "CLASSIFYING"
            self.phase_step_count = 0
            # Initialize accumulators for all dicts EXCEPT the one we just left
            self.coherence_sums = {j: 0.0 for j in range(len(self.dictionary_library)) 
                                   if j != self.last_active_id}
            
        return g

    def _step_classifying(self, t, log_S_new):
        """Phase 2: Online Coherence Accumulation (1 matrix at a time)."""
        self.phase_step_count += 1
        
        # Evaluate this single matrix against allowed dictionaries
        for j in self.coherence_sums.keys():
            D_j = self.dictionary_library[j]
            diffs = D_j - log_S_new  # Broadcasting spatial difference
            sq_distances = np.sum(diffs**2, axis=(1, 2))
            c_f = np.max(np.exp(-sq_distances / (2 * self.sigma**2)))
            
            self.coherence_sums[j] += c_f # Accumulate online
            
        # Decision point: S_class window is over
        if self.phase_step_count == self.N:
            best_j = -1
            max_c_mean = -1.0
            
            # Calculate final means
            for j, total_coherence in self.coherence_sums.items():
                c_mean = total_coherence / self.N
                if c_mean > max_c_mean:
                    max_c_mean = c_mean
                    best_j = j
                    
            if max_c_mean > self.psi:
                print(f"Time {t}: Re-using State {best_j} (Coherence {max_c_mean:.4f} > {self.psi})")
                self._init_active_state(self.dictionary_library[best_j], state_id=best_j)
                self.state = "ACTIVE"
            else:
                print(f"Time {t}: Unknown State (Max Coherence {max_c_mean:.4f} <= {self.psi}). Initiating Warm-up.")
                self.state = "WARMING_UP"
                self.phase_step_count = 0
                self.warmup_dict_list = [] # Reset for inline building
                
        return np.nan 

    def _step_warming_up(self, t, log_S_new):
        """Phase 3: Online Dictionary Building (1 matrix at a time)."""
        # First matrix of the warm-up automatically becomes the first dictionary atom
        if self.phase_step_count == 0:
            self.warmup_dict_list.append(log_S_new)
        else:
            # Check coherence against the CURRENT state of the new dictionary
            diffs = np.array(self.warmup_dict_list) - log_S_new
            sq_distances = np.sum(diffs**2, axis=(1, 2))
            kernel_values = np.exp(-sq_distances / (2 * self.sigma**2))
            
            if np.max(kernel_values) <= self.eta_0:
                self.warmup_dict_list.append(log_S_new)
                
        self.phase_step_count += 1
        
        # Decision point: S_warm window is over
        if self.phase_step_count == self.N:
            new_state_id = len(self.dictionary_library) # Create new ID
            new_dict = np.array(self.warmup_dict_list)
            
            # Note: We don't append to dictionary_library yet. 
            # We append it when it finishes its active run (at the next changepoint).
            self._init_active_state(new_dict, state_id=new_state_id)
            
            print(f"Time {t}: New Dictionary created with size {self.L}. Resuming tracking.")
            self.state = "ACTIVE"
            
        return np.nan