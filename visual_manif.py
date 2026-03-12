import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def visualize_spd_manifold(data, labels, window_size=40, epsilon=1e-5):
    """
    Computes sliding covariance matrices, maps them to the tangent space,
    and visualizes them in 3D using PCA.
    """
    N, D = data.shape
    tangent_vectors = []
    valid_labels = []
    
    # 1. Extract Covariance Matrices and map to Tangent Space
    for t in range(window_size, N):
        window_data = data[t - window_size : t]
        
        # Compute SPD Covariance
        S = np.cov(window_data, rowvar=False) + epsilon * np.eye(D)
        
        # Map to Tangent Space via Matrix Logarithm
        log_S = scipy.linalg.logm(S).real
        
        # Flatten the symmetric matrix into a 1D Euclidean vector
        # (We take the upper triangle to avoid redundant data)
        upper_triangle_indices = np.triu_indices(D)
        v = log_S[upper_triangle_indices]
        
        tangent_vectors.append(v)
        # Store the label of the most recent point in the window
        valid_labels.append(labels[t - 1]) 
        
    tangent_vectors = np.array(tangent_vectors)
    valid_labels = np.array(valid_labels)
    
    # 2. Compress to 3D using PCA
    pca = PCA(n_components=3)
    manifold_3d = pca.fit_transform(tangent_vectors)
    
    # 3. Visualize in 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a distinct colormap for the classes
    cmap = plt.get_cmap('Set1')
    unique_classes = np.unique(valid_labels)
    
    # Plot each class as a distinct cluster
    for cls in unique_classes:
        idx = valid_labels == cls
        ax.scatter(manifold_3d[idx, 0], 
                   manifold_3d[idx, 1], 
                   manifold_3d[idx, 2], 
                   c=[cmap(cls % cmap.N)], 
                   label=f'Regime {cls}', 
                   alpha=0.6, 
                   edgecolor='k', 
                   s=40)
        
    # Optional: Draw a faint line connecting the points chronologically to show the "jump"
    ax.plot(manifold_3d[:, 0], manifold_3d[:, 1], manifold_3d[:, 2], color='gray', alpha=0.2, linewidth=0.5)

    ax.set_title("3D PCA Projection of the Log-Euclidean Tangent Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% Variance)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_manifold_with_ground_truth(data, labels, classes, window_size=40, epsilon=1e-5):
    """
    Visualizes the empirical covariance clusters and overlays the exact 
    theoretical centroids calculated from the state-space physics.
    """
    N, D = data.shape
    empirical_vectors = []
    valid_labels = []
    
    # ==========================================
    # 1. EMPIRICAL DATA (The sliding windows)
    # ==========================================
    upper_triangle_indices = np.triu_indices(D)
    
    for t in range(window_size, N):
        window_data = data[t - window_size : t]
        
        # Sample Covariance
        S = np.cov(window_data, rowvar=False) + epsilon * np.eye(D)
        
        # Map to Tangent Space
        log_S = scipy.linalg.logm(S).real
        
        # Flatten into vector
        empirical_vectors.append(log_S[upper_triangle_indices])
        valid_labels.append(labels[t - 1]) 
        
    empirical_vectors = np.array(empirical_vectors)
    valid_labels = np.array(valid_labels)
    
    # ==========================================
    # 2. THEORETICAL TRUTH (The Lyapunov Anchors)
    # ==========================================
    true_vectors = []
    
    for cls_dict in classes:
        A, B, C, D_mat = cls_dict['A'], cls_dict['B'], cls_dict['C'], cls_dict['D']
        
        # Solve Discrete Lyapunov: Sigma_X = A * Sigma_X * A.T + B * B.T
        Q = B @ B.T
        Sigma_X = scipy.linalg.solve_discrete_lyapunov(A, Q)
        
        # Calculate theoretical observation covariance: Sigma_Y = C * Sigma_X * C.T + D * D.T
        Sigma_Y = C @ Sigma_X @ C.T + D_mat @ D_mat.T + epsilon * np.eye(D)
        
        # Map theoretical true covariance to Tangent Space
        log_Sigma_Y = scipy.linalg.logm(Sigma_Y).real
        
        # Flatten into vector
        true_vectors.append(log_Sigma_Y[upper_triangle_indices])
        
    true_vectors = np.array(true_vectors)
    
    # ==========================================
    # 3. PCA DIMENSIONALITY REDUCTION
    # ==========================================
    pca = PCA(n_components=3)
    
    # We fit the PCA purely on the empirical data
    empirical_3d = pca.fit_transform(empirical_vectors)
    
    # We project the theoretical truths into that exact same 3D space
    true_3d = pca.transform(true_vectors)
    
    # ==========================================
    # 4. 3D VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('Set1')
    unique_classes = np.unique(valid_labels)
    
    # Plot the Empirical Clusters (Small transparent dots)
    for cls in unique_classes:
        idx = valid_labels == cls
        ax.scatter(empirical_3d[idx, 0], 
                   empirical_3d[idx, 1], 
                   empirical_3d[idx, 2], 
                   c=[cmap(cls % cmap.N)], 
                   label=f'Empirical Class {cls}', 
                   alpha=0.3, 
                   edgecolor='none', 
                   s=30)
        
    # Plot the Theoretical True Centers (Massive Stars)
    for i in range(len(classes)):
        ax.scatter(true_3d[i, 0], 
                   true_3d[i, 1], 
                   true_3d[i, 2], 
                   c=[cmap(i % cmap.N)], 
                   marker='D', 
                   s=200,           # Make them huge
                   edgecolor='black', 
                   linewidth=2,
                   label=f'TRUE Anchor {i}',
                   zorder=10)       # Force them to render on top
        
    ax.set_title("Manifold of Covariance: Empirical Data vs. Theoretical Truth")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    
    # Put legend outside the plot so it doesn't block the 3D view
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

# --- How to call it in your main script ---
# data, labels, classes = generate_ssm_timeseries(...)
# visualize_manifold_with_ground_truth(data, labels, classes, window_size=40)