"""
Clustering utilities for latent merging.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cosine_dist_from_decoder(W: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance matrix from decoder weight matrix.
    
    Args:
        W: Decoder weight matrix [d_model, K]
        
    Returns:
        Cosine distance matrix [K, K]
    """
    # Normalize columns (each latent's decoder vector)
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    
    # Cosine similarity matrix
    cos_sim = (W_norm.T @ W_norm).clip(-1.0, 1.0)
    
    # Convert to distance (1 - similarity)
    cos_dist = 1.0 - cos_sim
    
    return cos_dist


def agglomerative_labels(dist: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform agglomerative clustering on distance matrix.
    
    Args:
        dist: Precomputed distance matrix [K, K]
        n_clusters: Number of clusters to form
        
    Returns:
        Cluster labels [K]
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    
    labels = clustering.fit_predict(dist)
    return labels


def auto_n_clusters(K: int, target_compression: float = 0.125) -> int:
    """
    Automatically determine number of clusters.
    
    Args:
        K: Number of original latents
        target_compression: Target compression ratio (e.g., 0.125 for 8:1)
        
    Returns:
        Suggested number of clusters
    """
    n_clusters = max(16, int(K * target_compression))
    return min(n_clusters, K)
