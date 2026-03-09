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

