import numpy as np
import matplotlib.pyplot as plt
import time

# Import your translated functions and parameters
from functions_nougat import nougat, rulsif, ma, knnt
from params import sample_h0, sample_h1, nc, nt, n_ref, n_test, mu, nu, gamma, k_knn

# ---------------------------------------------------------
# 1. Setup Signal
# ---------------------------------------------------------
# Generate Time Series with a change point at 'nc'
x = np.hstack([sample_h0(nc - 1), sample_h1(nt - nc + 1)])

t_nougat = []
t_rulsif = []
t_ma = []
t_knn = []

# Julia's 10:50:600 gives [10, 60, 110, ..., 560] (12 elements)
# np.arange(start, stop_exclusive, step)
L_vals = np.arange(10, 601, 50) 

# ---------------------------------------------------------
# 2. Benchmarking Helper Function
# ---------------------------------------------------------
def benchmark_median(func, *args, runs=5):
    """Runs a function multiple times and returns the median execution time in seconds."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times)

# ---------------------------------------------------------
# 3. Execution Loop
# ---------------------------------------------------------
for l in L_vals:
    print(f"L = {l}", flush=True)
    dict_x = sample_h0(l)

    # Benchmark NOUGAT
    t_nougat.append(benchmark_median(nougat, x, dict_x, n_ref, n_test, mu, nu, gamma))
    
    # Benchmark dRuLSIF
    t_rulsif.append(benchmark_median(rulsif, x, dict_x, n_ref, n_test, nu, gamma))
    
    # Benchmark MA
    t_ma.append(benchmark_median(ma, x, dict_x, n_ref, n_test, gamma))
    
    # Benchmark k-NN (Note: k-NN doesn't use the dictionary, but we benchmark it anyway)
    t_knn.append(benchmark_median(knnt, x, n_ref, n_test, k_knn))

# Convert lists to numpy arrays for easy slicing
t_nougat = np.array(t_nougat)
t_rulsif = np.array(t_rulsif)
t_ma = np.array(t_ma)
t_knn = np.array(t_knn)

# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------
# Set global font size (similar to Plots.scalefontsizes(1.5))
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(8, 6))

# Slicing: Julia's 2:12 takes elements from index 2 to 12 (1-based index).
# This translates to dropping the first element in Python (0-based index), hence [1:]
# Note: time.perf_counter returns seconds, so we DO NOT divide by 1e9 like in Julia.

plt.plot(L_vals[1:], t_rulsif[1:], label="dRuLSIF", marker='o', linewidth=2)
plt.plot(L_vals[1:], t_nougat[1:], label="NOUGAT",  marker='o', linewidth=2)
plt.plot(L_vals[1:], t_ma[1:],     label="MA",      marker='o', linewidth=2)
plt.plot(L_vals[1:], t_knn[1:],    label="k-NN",    marker='o', linewidth=2)

# Using raw string (r"...") allows Matplotlib to render LaTeX properly
plt.xlabel(r"$L$")
plt.ylabel("sec.")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("timing.pdf")
plt.show()