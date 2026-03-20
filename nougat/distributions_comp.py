import numpy as np
import os
from joblib import Parallel, delayed
from functions_nougat import nougat, rulsif, ma, knnt # Assuming previous code is saved here

# ---------------------------------------------------------
# 1. Load Parameters (Mimicking `include("params.jl")`)
# ---------------------------------------------------------
# You will need to define these in a separate params.py file 
# and import them, or define them directly here.
# Example placeholders:
try:
    from params import nt, nc, n_ref, n_test, mu, nu, gamma, k_knn
    from params import sample_h0, sample_h1 # Custom functions to sample from your distributions
except ImportError:
    print("Warning: params.py not found. Using dummy variables for demonstration.")
    nt, nc, n_ref, n_test = 200, 100, 20, 20
    mu, nu, gamma, k_knn = 0.1, 0.01, 1.0, 5
    
    # Dummy sampling functions returning shape (dimensions, num_samples)
    def sample_h0(n): return np.random.randn(5, n) 
    def sample_h1(n): return np.random.randn(5, n) + 2.0 

# ---------------------------------------------------------
# 2. Setup Shared Arrays via Memory Mapping
# ---------------------------------------------------------
realmax = 10

# Calculate sizes based on previous Julia logic
size_nougat_rulsif = nt - n_ref - n_test
size_ma_knn = nt - n_ref - n_test + 1

# Create memory-mapped files on disk (acts like SharedArray)
# mode='w+' creates or overwrites the file
t_nougat = np.memmap('t_nougat.dat', dtype='float64', mode='w+', shape=(size_nougat_rulsif, realmax))
t_rulsif = np.memmap('t_rulsif.dat', dtype='float64', mode='w+', shape=(size_nougat_rulsif, realmax))
t_ma     = np.memmap('t_ma.dat',     dtype='float64', mode='w+', shape=(size_ma_knn, realmax))
t_knn    = np.memmap('t_knn.dat',    dtype='float64', mode='w+', shape=(size_ma_knn, realmax))


# ---------------------------------------------------------
# 3. Define the Worker Function
# ---------------------------------------------------------
def run_iteration(k):
    """Executes a single Monte Carlo iteration."""
    if k % 100 == 0:
        print(f"> {k}", flush=True)

    # Generate Dictionary (40 samples from H0, 40 from H1)
    dict_x = np.hstack([sample_h0(40), sample_h1(40)])

    # Generate Time Series with a change point at 'nc'
    x = np.hstack([sample_h0(nc - 1), sample_h1(nt - nc + 1)])

    # Compute statistics and write directly to the memory-mapped shared arrays
    # Note: Python uses 0-based indexing, so k is used directly
    t_nougat[:, k] = nougat(x, dict_x, n_ref, n_test, mu, nu, gamma)
    t_rulsif[:, k] = rulsif(x, dict_x, n_ref, n_test, nu, gamma)
    t_ma[:, k]     = ma(x, dict_x, n_ref, n_test, gamma)
    t_knn[:, k]    = knnt(x, n_ref, n_test, k_knn)


# ---------------------------------------------------------
# 4. Execute Parallel Loop
# ---------------------------------------------------------
if __name__ == '__main__':
    print(f"Starting {realmax} iterations across available CPU cores...")
    
    # n_jobs=-1 uses all available CPU cores. 
    # require='sharedmem' ensures workers can write to the memmap safely
    Parallel(n_jobs=-1, require='sharedmem')(
        delayed(run_iteration)(k) for k in range(realmax)
    )
    
    print("Simulation complete. Results saved to disk via memmap.")
    
    # Flush memory map changes to disk
    t_nougat.flush()
    t_rulsif.flush()
    t_ma.flush()
    t_knn.flush()