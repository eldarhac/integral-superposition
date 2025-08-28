"""
Medoid selection for cluster representatives.
"""

import numpy as np
from typing import Dict, List


def medoids_from_labels(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Find medoid (most central point) for each cluster.
    
    Args:
        dist: Distance matrix [K, K]
        labels: Cluster labels [K]
        
    Returns:
        Array of medoid indices, one per cluster
    """
    unique_clusters = np.unique(labels)
    medoids = []
    
    for cluster_id in unique_clusters:
        # Get indices of points in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 1:
            # Single point cluster - it's the medoid
            medoids.append(cluster_indices[0])
        else:
            # Find point with minimum sum of distances to other cluster members
            cluster_dist = dist[np.ix_(cluster_indices, cluster_indices)]
            sum_distances = cluster_dist.sum(axis=1)
            local_medoid_idx = np.argmin(sum_distances)
            global_medoid_idx = cluster_indices[local_medoid_idx]
            medoids.append(global_medoid_idx)
    
    return np.array(sorted(medoids), dtype=int)


def merge_matrix_from_labels(labels: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """
    Create merge matrix mapping original latents to medoid representatives.
    
    Args:
        labels: Cluster labels [K]
        medoids: Medoid indices [K']
        
    Returns:
        Merge matrix [K, K'] where M[i,j] = 1 if latent i maps to medoid j
    """
    K = len(labels)
    K_prime = len(medoids)
    
    # Create mapping from medoid index to column index
    medoid_to_col = {medoid_idx: col for col, medoid_idx in enumerate(medoids)}
    
    # Build merge matrix
    M = np.zeros((K, K_prime), dtype=np.float32)
    
    for latent_idx, cluster_label in enumerate(labels):
        # Find which medoid represents this cluster
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_medoids = [idx for idx in cluster_indices if idx in medoid_to_col]
        
        if cluster_medoids:
            medoid_idx = cluster_medoids[0]  # Should be exactly one
            col_idx = medoid_to_col[medoid_idx]
            M[latent_idx, col_idx] = 1.0
    
    return M


def cluster_stats(labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering statistics.
    
    Args:
        labels: Cluster labels
        
    Returns:
        Dictionary with clustering statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        "n_clusters": len(unique_labels),
        "mean_cluster_size": float(counts.mean()),
        "max_cluster_size": int(counts.max()),
        "min_cluster_size": int(counts.min()),
        "compression_ratio": float(len(unique_labels) / len(labels)),
    }
    
    return stats
