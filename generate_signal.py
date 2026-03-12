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

def generate_ssm_timeseries(num_channels, num_classes, num_changes, num_data_points):
    """
    Generates observable time series data from a hidden Linear Gaussian State-Space Model.
    
    System formulation:
    X_t = A * X_{t-1} + B * u_t  (Hidden Physics)
    Y_t = C * X_t + D * q_t      (Observed Sensors)
    """
    
    # 1. Define the unique classes (Regimes)
    classes = []
    for _ in range(num_classes):
        # A: State Transition Matrix (Causality of the physics)
        A = np.random.randn(num_channels, num_channels)
        # MUST scale A to ensure system stability (eigenvalues < 1)
        max_eig = np.max(np.abs(np.linalg.eigvals(A)))
        A = (A / max_eig) * np.random.uniform(0.5, 0.95) 
        
        # B: Process Noise Mapping (Only diagonal to ensure independent noise inputs)
        B_diag_values = np.random.randn(num_channels) * 0.5
        B = np.diag(B_diag_values)
        
        # C: Observation Matrix (How hidden states map to your sensors)
        # We start with an identity matrix and add some cross-talk/mixing
        #C = np.eye(num_channels) + np.random.randn(num_channels, num_channels) * 0.2
        C = np.eye(num_channels)
        
        # D: Measurement Noise Mapping (Sensor static/inaccuracy)
        D_diag_values = np.random.randn(num_channels) * 0.1
        D = np.diag(D_diag_values)
        
        classes.append({'A': A, 'B': B, 'C': C, 'D': D})
        
    # 2. Determine change point boundaries
    change_indices = np.sort(np.random.choice(range(1, num_data_points), num_changes, replace=False))
    boundaries = [0] + list(change_indices) + [num_data_points]
    
    # 3. Initialize arrays
    # Y is the ONLY thing your CPD algorithm is allowed to see
    Y = np.zeros((num_data_points, num_channels))
    labels = np.zeros(num_data_points, dtype=int)
    
    # Initial hidden state
    X = np.zeros(num_channels) 
    
    # 4. Generate the time series
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        
        # Pick a random class for this segment (ensure it's different from the previous one)
        if i == 0:
            class_idx = np.random.randint(0, num_classes)
        else:
            possible_classes = list(range(num_classes))
            possible_classes.remove(labels[start-1])
            class_idx = np.random.choice(possible_classes)
            
        params = classes[class_idx]
        A, B, C, D = params['A'], params['B'], params['C'], params['D']
        
        for t in range(start, end):
            # Generate standard Gaussian noise for this time step
            u = np.random.randn(num_channels) # Process noise
            q = np.random.randn(num_channels) # Measurement noise
            
            # Step the hidden physical state forward
            X = A @ X + B @ u
            
            # Read the state through the sensors
            Y[t] = C @ X + D @ q
            labels[t] = class_idx
            
    # I am returning the labels as well so you can plot the blue "True Change" lines
    return Y, labels, classes

