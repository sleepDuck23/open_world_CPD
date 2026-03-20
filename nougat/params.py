import numpy as np
from functions_nougat import pdf_gmm, median_trick

def sample_gmm(means, covs, weights, n_samples):
    """
    Helper function to sample from a Gaussian Mixture Model.
    Returns an array of shape (d, n_samples).
    """
    d = means.shape[1]
    k = len(weights)
    
    # Randomly choose which component each sample comes from based on the weights
    components = np.random.choice(k, size=n_samples, p=weights)
    
    samples = np.zeros((d, n_samples))
    for i in range(k):
        # Find which samples belong to component i
        idx = (components == i)
        n_i = np.sum(idx)
        if n_i > 0:
            # Sample from the i-th multivariate normal and assign to columns
            samples[:, idx] = np.random.multivariate_normal(means[i], covs[i], size=n_i).T
            
    return samples

# --- data ---
d = 6
k_gmm = 3

# Generate GMM parameters for H0 and H1
means_h0, covs_h0, weights_h0 = pdf_gmm(d, k_gmm, sigma=1.0)
means_h1, covs_h1, weights_h1 = pdf_gmm(d, k_gmm, sigma=1.0)

# Create convenient wrapper functions to be imported by your main script
def sample_h0(n):
    return sample_gmm(means_h0, covs_h0, weights_h0, n)

def sample_h1(n):
    return sample_gmm(means_h1, covs_h1, weights_h1, n)

# --- cpd ---
nc = 100 
nt = 200

# --- nougat ---
n_ref = 30
n_test = 30
mu = 0.047   # step size
nu = 0.01    # ridge regularization

# Create the dictionary by horizontally stacking 40 samples from H0 and H1
dict_x = np.hstack([sample_h0(40), sample_h1(40)])

# --- kernel bandwidth ---
data = sample_h0(100)
gamma = median_trick(data)

# --- knn ---
k_knn = 10