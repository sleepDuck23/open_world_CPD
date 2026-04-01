import numpy as np
from scipy.stats import wishart
import matplotlib.pyplot as plt

def generate_wishart_series(total_steps, change_point, dim):
    """
    Generates a series of SPD matrices from Wishart distributions
    with a change point.
    """
    # --- Parameters for Regime 1 (Before change point) ---
    df1 = dim + 50  # Degrees of freedom must be >= dimension
    scale1 = np.eye(dim)   # Simple diagonal scale matrix
    print(f"Scale matrix for Regime 1:\n{scale1}")
    
    # --- Parameters for Regime 2 (After change point) ---
    df2 = dim + 10
    # Create a dense, random SPD scale matrix for the second regime
    np.random.seed(42) 
    A = np.random.randn(dim, dim)
    scale2 = np.eye(dim) + 0.1 * (A @ A.T) # A*A^T ensures positive definiteness
    print(f"Scale matrix for Regime 2:\n{scale2}")

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

def generate_multiple_wishart_series(total_steps, changepoints, dim=3, df=10):
    """
    Generates a time series of SPD matrices with multiple change points.
    """
    data = []
    
    # Start with a base identity covariance matrix
    current_cov = np.eye(dim)
    
    cp_idx = 0
    for t in range(total_steps):
        # If we hit a designated change point, shift the underlying distribution
        if cp_idx < len(changepoints) and t == changepoints[cp_idx]:
            # Create a brand new random symmetric positive-definite base matrix
            A = np.random.randn(dim, dim)
            current_cov = A @ A.T + np.eye(dim) * 0.5 
            cp_idx += 1
            
        # Generate a Wishart SPD matrix from the current base covariance
        # We simulate this by drawing 'df' samples from a multivariate normal
        X = np.random.multivariate_normal(np.zeros(dim), current_cov, size=df)
        S = np.dot(X.T, X) / df
        data.append(S)
        
    return np.array(data)

import numpy as np

def generate_alternating_wishart_series(total_steps, changepoints, distribution_sequence, base_covariances=None, dim=3, df=10):
    """
    Generates a time series of SPD matrices alternating between predefined distributions.
    
    Parameters:
    - total_steps: Total length of the time series.
    - changepoints: List of time indices where the distribution shifts.
    - distribution_sequence: List of integers mapping to the distribution state (e.g., [0, 1, 0, 2]).
                             Must have a length of len(changepoints) + 1.
    - base_covariances: Optional list of pre-defined SPD matrices for each state. 
                        If None, they will be generated automatically.
    - dim: Dimension of the matrices.
    - df: Degrees of freedom for the Wishart distribution.
    """
    # Validate that the sequence length matches the number of segments
    if len(distribution_sequence) != len(changepoints) + 1:
        raise ValueError("Length of distribution_sequence must be exactly len(changepoints) + 1")
        
    num_unique_distributions = max(distribution_sequence) + 1
    
    # Step 1: Initialize the base covariances if not provided
    if base_covariances is None:
        base_covariances = []
        for _ in range(num_unique_distributions):
            # Create a random symmetric positive-definite base matrix
            A = np.random.randn(dim, dim)
            cov = A @ A.T + np.eye(dim) * 0.5
            base_covariances.append(cov)
            
    data = []
    cp_idx = 0
    
    # Step 2: Set the initial distribution based on the first item in the sequence
    current_state = distribution_sequence[0]
    current_cov = base_covariances[current_state]
    
    # Step 3: Generate the time series
    for t in range(total_steps):
        # If we hit a designated change point, shift to the next state in the sequence
        if cp_idx < len(changepoints) and t == changepoints[cp_idx]:
            cp_idx += 1
            current_state = distribution_sequence[cp_idx]
            current_cov = base_covariances[current_state]
            
        # Generate a Wishart SPD matrix from the current base covariance
        X = np.random.multivariate_normal(np.zeros(dim), current_cov, size=df)
        S = np.dot(X.T, X) / df
        data.append(S)
        
    return np.array(data)