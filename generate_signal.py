import numpy as np
import matplotlib.pyplot as plt

def generate_segmented_timeseries(K, num_classes, num_changes):
    """
    Generates a Gaussian time series with regime shifts.
    
    K: Total number of data points
    num_classes: Number of unique parameter sets (mu, sigma)
    num_changes: Number of times the regime switches
    """
    
    # 1. Define the unique classes (parameters)
    # Using random means between -10, 10 and std devs between 0.5, 3.0
    classes = [
        {'mu': np.random.uniform(-10, 10), 'sigma': np.random.uniform(0.5, 3.0)}
        for _ in range(num_classes)
    ]
    
    # 2. Determine change point indices
    # We pick (num_changes) unique indices from the total length K
    change_indices = np.sort(np.random.choice(range(1, K), num_changes, replace=False))
    # Add start and end boundaries
    boundaries = [0] + list(change_indices) + [K]
    
    timeseries = np.zeros(K)
    labels = np.zeros(K, dtype=int)
    
    # 3. Fill the segments
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        
        # Pick a random class index for this segment
        class_idx = np.random.randint(0, num_classes)
        params = classes[class_idx]
        
        # Generate the Gaussian data for this segment
        timeseries[start:end] = np.random.normal(params['mu'], params['sigma'], end - start)
        labels[start:end] = class_idx
        
    return timeseries, labels, classes

def generate_ssm_timeseries(num_channels, num_classes, num_changes, num_data_points, min_spacing=50):
    """
    Generates observable time series data from a hidden Linear Gaussian State-Space Model.
    
    System formulation:
    X_t = A * X_{t-1} + B * u_t  (Hidden Physics)
    Y_t = C * X_t + D * q_t      (Observed Sensors)
    """
    
    # 1. Define the unique classes (Regimes)
    classes = []
    for _ in range(num_classes):
        # A: State Transition Matrix
        A = np.random.randn(num_channels, num_channels)
        max_eig = np.max(np.abs(np.linalg.eigvals(A)))
        # Lowered eigenvalues to prevent long wandering arcs that confuse small sliding windows
        A = (A / max_eig) * np.random.uniform(0.2, 0.6) 
        
        # B: Process Noise Mapping 
        # Using uniform instead of randn to guarantee no channel accidentally flatlines at 0
        B_diag_values = np.random.uniform(0.3, 0.8, num_channels)
        B = np.diag(B_diag_values)
        
        # C: Observation Matrix
        C = np.eye(num_channels)
        
        # D: Measurement Noise Mapping
        D_diag_values = np.random.randn(num_channels) * 0.1
        D = np.diag(D_diag_values)
        
        classes.append({'A': A, 'B': B, 'C': C, 'D': D})
        
    # 2. Determine change point boundaries with guaranteed minimum spacing
    boundaries = [0]
    for i in range(num_changes):
        # Calculate the safe window to place the next change
        min_possible = boundaries[-1] + min_spacing
        max_possible = num_data_points - ((num_changes - i) * min_spacing)
        
        # Safety check: Ensure the user didn't ask for too many changes in too little time
        if min_possible >= max_possible:
            raise ValueError(f"Cannot fit {num_changes} changes with min_spacing={min_spacing} in {num_data_points} points.")
            
        next_change = np.random.randint(min_possible, max_possible)
        boundaries.append(next_change)
        
    boundaries.append(num_data_points)
    
    # 3. Initialize arrays
    Y = np.zeros((num_data_points, num_channels))
    labels = np.zeros(num_data_points, dtype=int)
    
    # 4. Generate the time series
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        
        # Strictly enforce a class change (No A -> A allowed)
        if i == 0:
            class_idx = np.random.randint(0, num_classes)
        else:
            possible_classes = list(range(num_classes))
            possible_classes.remove(labels[start-1])
            class_idx = np.random.choice(possible_classes)
            
        params = classes[class_idx]
        A, B, C, D = params['A'], params['B'], params['C'], params['D']
        
        # Reset the hidden physical state to zero at the start of a new regime
        X = np.zeros(num_channels)
        
        for t in range(start, end):
            u = np.random.randn(num_channels) # Process noise
            q = np.random.randn(num_channels) # Measurement noise
            
            # Step the hidden physical state forward
            X = A @ X + B @ u
            
            # Read the state through the sensors
            Y[t] = C @ X + D @ q
            labels[t] = class_idx
            
    return Y, labels, classes