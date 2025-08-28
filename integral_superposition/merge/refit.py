"""
Decoder refitting after merging latents.
"""

import torch
import numpy as np
from typing import Callable, Iterator, Tuple

from ..sae.model import SparseAutoencoder


def merge_and_refit_decoder(
    X_iter: Callable[[], Iterator[np.ndarray]],
    sae: SparseAutoencoder,
    M: np.ndarray,
    l2: float = 1e-3,
    device: str = "cpu"
) -> np.ndarray:
    """
    Merge latents using matrix M and refit decoder weights.
    
    Args:
        X_iter: Function returning iterator over activation batches [N, d_model]
        sae: Trained SAE model
        M: Merge matrix [K, K'] mapping old to new latents
        l2: L2 regularization strength
        device: Device for computation
        
    Returns:
        New decoder weights [d_model, K']
    """
    device = torch.device(device)
    sae = sae.to(device)
    sae.eval()
    
    M_tensor = torch.as_tensor(M, dtype=torch.float32, device=device)
    
    # Accumulate normal equation components
    G = None  # Z'^T @ Z' 
    C = None  # Z'^T @ X
    
    with torch.no_grad():
        for X_batch in X_iter():
            if X_batch.shape[0] == 0:
                continue
                
            X_tensor = torch.from_numpy(X_batch.astype(np.float32)).to(device)
            
            # Get original latents
            Z = sae.encode(X_tensor)  # [N, K]
            
            # Merge to new latents
            Z_prime = Z @ M_tensor  # [N, K']
            
            # Optional mean centering (for no-bias decoder)
            X_centered = X_tensor - X_tensor.mean(dim=0, keepdim=True)
            Z_prime_centered = Z_prime - Z_prime.mean(dim=0, keepdim=True)
            
            # Accumulate Gram matrix and covariance
            batch_G = Z_prime_centered.T @ Z_prime_centered  # [K', K']
            batch_C = Z_prime_centered.T @ X_centered        # [K', d_model]
            
            if G is None:
                G = batch_G
                C = batch_C
            else:
                G += batch_G
                C += batch_C
    
    if G is None:
        raise ValueError("No data processed - check X_iter")
    
    # Add L2 regularization
    if l2 > 0:
        G = G + l2 * torch.eye(G.shape[0], device=device, dtype=G.dtype)
    
    # Solve normal equations: G @ W'^T = C
    W_prime_T = torch.linalg.solve(G, C)  # [K', d_model]
    W_prime = W_prime_T.T.contiguous()    # [d_model, K']
    
    # Optional: normalize decoder columns to unit norm
    col_norms = W_prime.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W_prime_unit = W_prime / col_norms
    
    return W_prime_unit.cpu().numpy()


def streaming_refit(
    XZ_iter: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]],
    M: np.ndarray,
    l2: float = 1e-3,
    device: str = "cpu"
) -> np.ndarray:
    """
    Refit decoder from pre-computed (X, Z) pairs.
    
    Args:
        XZ_iter: Iterator yielding (X_batch, Z_batch) tuples
        M: Merge matrix [K, K']
        l2: L2 regularization
        device: Computation device
        
    Returns:
        Refitted decoder weights [d_model, K']
    """
    device = torch.device(device)
    M_tensor = torch.as_tensor(M, dtype=torch.float32, device=device)
    
    G = None
    C = None
    
    for X_batch, Z_batch in XZ_iter():
        if X_batch.shape[0] == 0:
            continue
            
        X_tensor = torch.from_numpy(X_batch.astype(np.float32)).to(device)
        Z_tensor = torch.from_numpy(Z_batch.astype(np.float32)).to(device)
        
        # Merge latents
        Z_prime = Z_tensor @ M_tensor
        
        # Center for no-bias solution
        X_c = X_tensor - X_tensor.mean(dim=0, keepdim=True)
        Z_c = Z_prime - Z_prime.mean(dim=0, keepdim=True)
        
        # Accumulate
        batch_G = Z_c.T @ Z_c
        batch_C = Z_c.T @ X_c
        
        if G is None:
            G = batch_G
            C = batch_C
        else:
            G += batch_G
            C += batch_C
    
    if G is None:
        raise ValueError("No data processed")
    
    # Ridge regression
    if l2 > 0:
        G = G + l2 * torch.eye(G.shape[0], device=device, dtype=G.dtype)
    
    W_T = torch.linalg.solve(G, C)
    W = W_T.T.contiguous()
    
    # Unit norm columns
    W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    
    return W.cpu().numpy()
