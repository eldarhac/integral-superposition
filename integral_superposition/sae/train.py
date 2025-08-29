"""
SAE training utilities.
"""

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterator
from tqdm.auto import tqdm

from .model import SparseAutoencoder
from ..config import SAEConfig


def train_sae(
    sae: SparseAutoencoder,
    shard_iter: Callable[[], Iterator[torch.Tensor]],
    cfg: SAEConfig,
    device: torch.device
) -> None:
    """
    Train a Sparse Autoencoder on activation shards.
    
    Args:
        sae: SparseAutoencoder model
        shard_iter: Function returning iterator over activation batches
        cfg: Training configuration
        device: Device to train on
    """
    sae = sae.to(device)
    sae.train()
    
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr)
    
    for epoch in range(cfg.epochs):
        running_recon = 0.0
        running_l1 = 0.0
        n_samples = 0
        
        # Create fresh iterator for each epoch
        data_iter = shard_iter()
        
        for batch in tqdm(data_iter, desc=f"SAE epoch {epoch+1}/{cfg.epochs}"):
            if batch.shape[0] == 0:
                continue
                
            # Limit batch size if needed
            if batch.shape[0] > cfg.batch_size:
                batch = batch[:cfg.batch_size]
            
            # Convert numpy array to tensor and move to device
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch.astype(np.float32))
            batch = batch.to(device, non_blocking=True)
            
            # Forward pass
            x_hat, z = sae(batch)
            
            # Losses
            recon_loss = ((x_hat - batch) ** 2).mean()
            l1_loss = z.abs().mean()
            total_loss = recon_loss + cfg.l1 * l1_loss
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            batch_size = batch.size(0)
            running_recon += recon_loss.item() * batch_size
            running_l1 += l1_loss.item() * batch_size
            n_samples += batch_size
        
        if n_samples > 0:
            avg_recon = running_recon / n_samples
            avg_l1 = running_l1 / n_samples
            print(f"  recon MSE: {avg_recon:.6f} | L1(z): {avg_l1:.6f} | lam={cfg.l1}")
        else:
            print(f"  No samples processed in epoch {epoch+1}")


def save_sae(sae: SparseAutoencoder, path: str) -> None:
    """
    Save SAE state dict to disk.
    
    Args:
        sae: Trained SAE model
        path: Path to save weights
    """
    torch.save(sae.state_dict(), path)


def load_sae(
    path: str,
    d_model: int = None,
    k: int = None,
    map_location: str = "cpu"
) -> SparseAutoencoder:
    """
    Load SAE from saved weights.
    
    Args:
        path: Path to saved weights
        d_model: Model input/output dimension (inferred if None)
        k: Number of latent dimensions (inferred if None)
        map_location: Device to load weights to
        
    Returns:
        Loaded SAE model
    """
    state_dict = torch.load(path, map_location=map_location)
    
    # Infer dimensions from state dict if not provided
    if d_model is None:
        d_model = state_dict['enc.weight'].shape[1]
    if k is None:
        k = state_dict['enc.weight'].shape[0]
    
    sae = SparseAutoencoder(d_model, k)
    sae.load_state_dict(state_dict)
    sae.eval()
    
    return sae
