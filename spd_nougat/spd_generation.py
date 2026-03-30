import numpy as np
from scipy.stats import wishart
import matplotlib.pyplot as plt

def generate_wishart_series(total_steps, change_point, dim):
    """
    Generates a series of SPD matrices from Wishart distributions
    with a change point.
    """
    # --- Parameters for Regime 1 (Before change point) ---
    df1 = dim   # Degrees of freedom must be >= dimension
    scale1 = np.eye(dim) * 2.0  # Simple diagonal scale matrix
    
    # --- Parameters for Regime 2 (After change point) ---
    df2 = dim + 10
    # Create a dense, random SPD scale matrix for the second regime
    np.random.seed(42) 
    A = np.random.randn(dim, dim)
    scale2 = A @ A.T  # A*A^T ensures positive definiteness

    # --- Generate the Series ---
    matrix_series = []
    
    for i in range(total_steps):
        if i < change_point:
            # Sample from the first Wishart distribution
            sample = wishart.rvs(df=df1, scale=scale1)
        else:
            # Sample from the second Wishart distribution
            sample = wishart.rvs(df=df2, scale=scale2)
            
        matrix_series.append(sample)
        
    return np.array(matrix_series)

## --- Execution and Visualization ---
#N = 100       # Total number of matrices in the sequence
#t = 50        # The time index where the change occurs
#d = 3         # Dimension of the matrices (3x3)
#
## Generate the data
#spd_matrices = generate_wishart_series(total_steps=N, change_point=t, dim=d)
#
## Calculate a scalar summary (the trace) to visualize the change
#traces = [np.trace(mat) for mat in spd_matrices]
#
## Plot the results
#plt.figure(figsize=(10, 5))
#plt.plot(traces, marker='o', linestyle='-', alpha=0.7, color='b')
#plt.axvline(x=t, color='r', linestyle='--', label=f'Change Point (t={t})')
#plt.title('Trace of Wishart-distributed SPD Matrices Over Time')
#plt.xlabel('Time Step')
#plt.ylabel('Matrix Trace')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.show()
#
## Verify the properties of a generated matrix
#print(f"Shape of the generated sequence: {spd_matrices.shape}")
#print(f"Matrix at t=10 is symmetric: {np.allclose(spd_matrices[10], spd_matrices[10].T)}")
## Check positive definiteness (all eigenvalues must be > 0)
#eigenvalues = np.linalg.eigvals(spd_matrices[10])
#print(f"Matrix at t=10 is positive definite: {np.all(eigenvalues > 0)}")