"""
Statistical analysis utilities.
"""

import numpy as np
from typing import Dict


def mean_pool_by_title(z: np.ndarray, title_idx: np.ndarray) -> np.ndarray:
    """
    Mean-pool latent activations by title/document.
    
    Args:
        z: Latent activations [N_tokens, K]
        title_idx: Title indices for each token [N_tokens]
        
    Returns:
        Mean-pooled activations [N_titles, K]
    """
    unique_titles = np.unique(title_idx)
    K = z.shape[1]
    
    pooled = np.zeros((len(unique_titles), K), dtype=np.float32)
    
    for i, title_id in enumerate(unique_titles):
        mask = title_idx == title_id
        pooled[i] = z[mask].mean(axis=0)
    
    return pooled


def trim_outliers(X2: np.ndarray, p: float = 99.0) -> np.ndarray:
    """
    Remove outliers from 2D embedding for cleaner visualization.
    
    Args:
        X2: 2D coordinates [N, 2]
        p: Percentile threshold for distance-based filtering
        
    Returns:
        Filtered 2D coordinates
    """
    if len(X2) == 0:
        return X2
    
    # Compute distances from centroid
    center = X2.mean(axis=0)
    distances = np.linalg.norm(X2 - center, axis=1)
    
    # Keep points within percentile threshold
    threshold = np.percentile(distances, p)
    mask = distances <= threshold
    
    return X2[mask]


def compute_activation_stats(
    z_vals: np.ndarray,
    min_activation: float = 0.0
) -> Dict[str, float]:
    """
    Compute basic statistics for latent activations.
    
    Args:
        z_vals: Latent activation values
        min_activation: Minimum activation threshold
        
    Returns:
        Dictionary with activation statistics
    """
    active_mask = z_vals > min_activation
    active_vals = z_vals[active_mask]
    
    stats = {
        "total_activations": len(z_vals),
        "active_count": int(active_mask.sum()),
        "sparsity": 1.0 - (active_mask.sum() / len(z_vals)),
        "mean_activation": float(z_vals.mean()),
        "mean_active": float(active_vals.mean()) if len(active_vals) > 0 else 0.0,
        "max_activation": float(z_vals.max()),
        "std_activation": float(z_vals.std()),
    }
    
    return stats
