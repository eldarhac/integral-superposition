"""
SAE inference utilities.
"""

import torch
import numpy as np

from .model import SparseAutoencoder


def latents_from_shard(
    sae: SparseAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch: int = 16384
) -> np.ndarray:
    """
    Compute latent activations for a shard of data.
    
    Args:
        sae: Trained SAE model
        X: Input activations [N, d_model]
        device: Device to run inference on
        batch: Batch size for inference
        
    Returns:
        Latent activations [N, k]
    """
    sae.eval()
    sae = sae.to(device)
    
    latents = []
    
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            end = min(start + batch, X.shape[0])
            x_batch = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
            
            z = sae.encode(x_batch)
            latents.append(z.cpu().numpy())
    
    return np.concatenate(latents, axis=0)
