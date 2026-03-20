import numpy as np
from scipy.linalg import inv, det, solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import wishart, multivariate_normal, dirichlet

def norm2(x):
    """Computes the squared L2 norm."""
    return np.sum(np.square(x))

def esp_exp_quad(k, a, b, w, s):
    """
    returns E[exp(s*(y'wy + b'y))]
    where y : N(a,k)
    """
    n = len(a)
    I = np.eye(n)
    
    # Using @ for matrix multiplication (Python 3.5+)
    res = np.exp(s * (np.dot(a, w @ a) + np.dot(b, a)))
    res /= np.sqrt(det(I - 2 * s * (w @ k)))
    
    term1 = 2 * (w @ a) + b
    term2 = k @ inv(I - 2 * s * (w @ k)) @ term1
    res *= np.exp((s**2 / 2) * np.dot(term1, term2))
    
    return res

def comp_H_h(dict_x, m, R, gamma):
    """
    computes matrices H and h
    - dict_x : dictionary (numpy array of shape (k, l))
    - m : mean vector of the signal
    - R : covariance matrix of the signal
    - gamma : kernel bandwidth
    """
    k_dim, l = dict_x.shape
    
    h = np.zeros(l)
    for q in range(l):
        b = -2 * dict_x[:, q]
        h[q] = np.exp(-norm2(dict_x[:, q]) / (2 * gamma))
        h[q] *= esp_exp_quad(R, m, b, np.eye(k_dim), -1 / (2 * gamma)) 
        
    H = np.zeros((l, l))
    for q in range(l):
        for n in range(l):
            b = -(dict_x[:, q] + dict_x[:, n])
            H[q, n] = np.exp(-(norm2(dict_x[:, q]) + norm2(dict_x[:, n])) / (2 * gamma))
            H[q, n] *= esp_exp_quad(R, m, b, np.eye(k_dim), -1 / gamma)  
            
    return H, h

def comp_Gamma(dict_x, m, R, gamma):
    """computes matrix Gamma"""
    k_dim, l = dict_x.shape
    Gamma = np.zeros((l**2, l**2))

    for q in range(l):
        for n in range(l):
            for i in range(l):
                for j in range(l):
                    # Python is 0-indexed, adapting the linear indexing
                    row = q * l + i
                    col = n * l + j
                    
                    b = -2 * (dict_x[:, q] + dict_x[:, n] + dict_x[:, i] + dict_x[:, j])
                    
                    val = np.exp(-(norm2(dict_x[:, q]) + norm2(dict_x[:, n]) + 
                                   norm2(dict_x[:, i]) + norm2(dict_x[:, j])) / (2 * gamma))
                    val *= esp_exp_quad(R, m, b, 4 * np.eye(k_dim), -1 / (2 * gamma))
                    Gamma[row, col] = val
                    
    return Gamma

def comp_Delta(dict_x, m, R, gamma):
    """computes matrix Delta"""
    k_dim, l = dict_x.shape
    Delta = np.zeros((l**2, l))

    for q in range(l):
        for i in range(l):
            for j in range(l):
                row = q * l + i
                
                b = -2 * (dict_x[:, q] + dict_x[:, i] + dict_x[:, j])
                
                val = np.exp(-(norm2(dict_x[:, q]) + norm2(dict_x[:, i]) + norm2(dict_x[:, j])) / (2 * gamma))
                val *= esp_exp_quad(R, m, b, 3 * np.eye(k_dim), -1 / (2 * gamma))
                Delta[row, j] = val
                
    return Delta

def comp_kappa(x, dict_x, gamma):
    """computes the vector of k(x, x_i^{dict})"""
    k_dim, l = dict_x.shape
    kappa = np.array([np.exp(-norm2(x - dict_x[:, m]) / (2 * gamma)) for m in range(l)])
    return kappa

def nougat(x, dict_x, n_ref, n_test, mu, nu, gamma):
    """computes NOUGAT online density ratio estimation"""
    k_dim, l = dict_x.shape
    n_iter = x.shape[1] - n_ref - n_test
    nougat_scores = np.zeros(n_iter)
    theta = np.zeros(l)

    # compute initial H_n and h_n
    H_nr = np.zeros((l, l))
    h_nr = np.zeros(l)
    for m in range(n_ref):
        kappa = comp_kappa(x[:, m], dict_x, gamma)
        H_nr += np.outer(kappa, kappa)
        h_nr += kappa
    H_nr /= n_ref
    h_nr /= n_ref

    h_nt = np.zeros(l)
    for m in range(n_ref, n_ref + n_test):
        kappa = comp_kappa(x[:, m], dict_x, gamma)
        h_nt += kappa
    h_nt /= n_test

    I = np.eye(l)
    for n in range(n_iter - 1):
        theta = theta - mu * ((H_nr + nu * I) @ theta + (h_nr - h_nt))
        
        kappa_n = comp_kappa(x[:, n_ref + n_test + n], dict_x, gamma)
        nougat_scores[n] = np.dot(h_nt, theta)

        # update H_nr, h_nr and h_nt the fast way
        um_r = comp_kappa(x[:, n], dict_x, gamma)
        up_r = comp_kappa(x[:, n + n_ref], dict_x, gamma)

        H_nr += (np.outer(up_r, up_r) - np.outer(um_r, um_r)) / n_ref
        h_nr += (up_r - um_r) / n_ref

        um_t = comp_kappa(x[:, n + n_ref], dict_x, gamma)
        up_t = comp_kappa(x[:, n + n_ref + n_test], dict_x, gamma)
        h_nt += (up_t - um_t) / n_test

    theta = theta - mu * ((H_nr + nu * I) @ theta + (h_nr - h_nt))
    nougat_scores[n_iter - 1] = np.dot(theta, h_nt)

    return nougat_scores

def rulsif(x, dict_x, n_ref, n_test, nu, gamma):
    """computes RuLSIF density ratio estimation"""
    k_dim, l = dict_x.shape
    n_iter = x.shape[1] - n_ref - n_test
    rulsif_scores = np.zeros(n_iter)

    H_nr = np.zeros((l, l))
    h_nr = np.zeros(l)
    for m in range(n_ref):
        kappa = comp_kappa(x[:, m], dict_x, gamma)
        H_nr += np.outer(kappa, kappa)
        h_nr += kappa
    H_nr /= n_ref
    h_nr /= n_ref

    h_nt = np.zeros(l)
    for m in range(n_ref, n_ref + n_test):
        kappa = comp_kappa(x[:, m], dict_x, gamma)
        h_nt += kappa
    h_nt /= n_test

    I = np.eye(l)
    for n in range(n_iter):
        # solve (H_nr + \nu*I) \ (h_nt - h_nr)
        theta = solve(H_nr + nu * I, h_nt - h_nr)
        rulsif_scores[n] = np.dot(theta, h_nt)

        # update fast way
        um_r = comp_kappa(x[:, n], dict_x, gamma)
        up_r = comp_kappa(x[:, n + n_ref], dict_x, gamma)
        H_nr += (np.outer(up_r, up_r) - np.outer(um_r, um_r)) / n_ref
        h_nr += (up_r - um_r) / n_ref

        um_t = comp_kappa(x[:, n + n_ref], dict_x, gamma)
        up_t = comp_kappa(x[:, n + n_ref + n_test], dict_x, gamma)
        h_nt += (up_t - um_t) / n_test

    return rulsif_scores

def newma(y, dict_x, n_ref, n_test, gamma, lam=0.1, Lam=0.01):
    """
    computes NEWMA (added lam and Lam missing from original code)
    """
    z = z_p = comp_kappa(y[:, 0], dict_x, gamma)
    test_scores = []

    for n in range(1, y.shape[1]):
        Psi = comp_kappa(y[:, n], dict_x, gamma)
        z = (1 - lam) * z + lam * Psi
        z_p = (1 - Lam) * z_p + Lam * Psi
        test_scores.append(np.linalg.norm(z - z_p))

    return np.array(test_scores)

def knnt(x, n_ref, n_test, k):
    """computes knn based two-sample test"""
    n_s = x.shape[1]
    n_iter = n_s - n_ref - n_test + 1
    knn_test = np.zeros(n_iter)

    for n in range(n_iter):
        data = x[:, n:n + n_ref + n_test]
        
        # cKDTree expects shape (n_samples, n_features)
        tree = cKDTree(data.T)
        
        # Query k+1 neighbors because the point itself is included
        dists, idxs = tree.query(data.T, k=k+1)
        idxs = idxs[:, 1:] # Remove self
        
        # Count neighbors in ref and test windows
        ref_counts = np.sum(idxs[:n_ref, :] >= n_ref)
        test_counts = np.sum(idxs[n_ref:, :] < n_ref)
        
        knn_test[n] = ref_counts + test_counts
        
    mean_test = k * n_ref * n_test / (n_ref + n_test - 1)
    return mean_test - knn_test

def ma(x, dict_x, n_ref, n_test, gamma):
    """computes moving average baseline"""
    k_dim, l = dict_x.shape
    n_iter = x.shape[1] - n_ref - n_test
    ma_scores = np.zeros(n_iter + 1)

    h_nr = np.zeros(l)
    for m in range(n_ref):
        h_nr += comp_kappa(x[:, m], dict_x, gamma)
    h_nr /= n_ref

    h_nt = np.zeros(l)
    for m in range(n_ref, n_ref + n_test):
        h_nt += comp_kappa(x[:, m], dict_x, gamma)
    h_nt /= n_test

    ma_scores[0] = np.linalg.norm(h_nr - h_nt)

    for n in range(n_iter):
        um_r = comp_kappa(x[:, n], dict_x, gamma)
        up_r = comp_kappa(x[:, n + n_ref], dict_x, gamma)
        h_nr += (up_r - um_r) / n_ref

        um_t = comp_kappa(x[:, n + n_ref], dict_x, gamma)
        up_t = comp_kappa(x[:, n + n_ref + n_test], dict_x, gamma)
        h_nt += (up_t - um_t) / n_test

        ma_scores[n + 1] = np.linalg.norm(h_nr - h_nt)

    return ma_scores

def comp_roc(t_0, t_1, n_xi=128, xi_pl=None):
    if xi_pl is None:
        xi_pl_inf = np.max(np.min(np.abs(t_0), axis=0))
        xi_pl_sup = np.min(np.max(np.abs(t_1), axis=0))
        xi_pl = np.linspace(xi_pl_inf, xi_pl_sup, n_xi)

    t_0_abs = np.abs(t_0)
    t_1_abs = np.abs(t_1)
    
    pfa = [np.sum(t_0_abs > xi) / t_0_abs.size for xi in xi_pl]
    pd = [np.mean(np.any(t_1_abs > xi, axis=0)) for xi in xi_pl]

    return pfa, pd, xi_pl

def comp_pfa(t_0, n_xi=128, xi_pl=None):
    if xi_pl is None:
        xi_pl_inf = np.min(np.abs(t_0))
        xi_pl_sup = np.max(np.abs(t_0))
        xi_pl = np.linspace(xi_pl_inf, xi_pl_sup, n_xi)

    t_0_abs = np.abs(t_0)
    pfa = [np.sum(t_0_abs > xi) / t_0_abs.size for xi in xi_pl]

    return pfa, xi_pl

def comp_mtd(t, n_xi=128, xi_pl=None):
    if xi_pl is None:
        xi_pl_inf = np.min(np.abs(t))
        xi_pl_sup = np.max(np.abs(t))
        xi_pl = np.linspace(xi_pl_inf, xi_pl_sup, n_xi)

    mean_delay = np.full(len(xi_pl), np.nan)
    t_abs = np.abs(t)
    
    for q, xi in enumerate(xi_pl):
        first_detec = []
        for k in range(t.shape[1]):
            # Find the first index where condition is met
            idx = np.where(t_abs[:, k] > xi)[0]
            if len(idx) > 0:
                first_detec.append(idx[0])
                
        if len(first_detec) == 0:
            mean_delay[q] = np.nan
        else:
            mean_delay[q] = np.mean(first_detec)

    return mean_delay, xi_pl

def pdf_gmm(d, k, n=None, sigma=1.0, alpha=5.0):
    """
    Returns parameters for a GMM.
    In Python, it's often easier to return the parameters (means, covs, weights)
    rather than a complex MixtureModel object.
    """
    if n is None:
        n = d + 2
        
    covs = wishart.rvs(df=n, scale=np.eye(d), size=k) / n
    if k == 1:
        covs = covs[np.newaxis, ...] # Ensure it's 3D even for k=1
        
    means = multivariate_normal.rvs(mean=np.zeros(d), cov=(sigma**2)*np.eye(d), size=k)
    if k == 1:
        means = means[np.newaxis, ...]
        
    weights = dirichlet.rvs(alpha * np.ones(k))[0]
    
    return means, covs, weights

def median_trick(data):
    """Computes median heuristic for RBF kernel bandwidth"""
    # pdist computes pairwise Euclidean distances
    dists = pdist(data.T, metric='euclidean')
    return np.median(dists)**2