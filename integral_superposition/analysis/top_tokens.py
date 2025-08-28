"""
Top token analysis for latent activations.
"""

import numpy as np
import torch
from typing import List, Tuple, Set
from tqdm.auto import tqdm

from ..sae.model import SparseAutoencoder
from ..backends.io_store import load_npy


def topk_sets_per_latent(
    sae: SparseAutoencoder,
    act_files: List[str], 
    device: torch.device,
    k: int = 200,
    batch: int = 16384
) -> List[Set[int]]:
    """
    Compute top-k token sets for each latent across all shards.
    
    Args:
        sae: Trained SAE model
        act_files: List of activation shard file paths
        device: Device for inference
        k: Number of top tokens to keep per latent
        batch: Batch size for processing
        
    Returns:
        List of sets, one per latent, containing top-activating token indices
    """
    sae.eval()
    sae = sae.to(device)
    
    K_LATENTS = sae.k
    offsets = [0]
    
    # Compute shard offsets
    for af in act_files:
        size = load_npy(af, mmap=True).shape[0]
        offsets.append(offsets[-1] + size)
    
    # Initialize storage for top-k tracking
    top_vals = [np.empty((0,), dtype=np.float32) for _ in range(K_LATENTS)]
    top_idx = [np.empty((0,), dtype=np.int64) for _ in range(K_LATENTS)]
    
    with torch.no_grad():
        for si, af in enumerate(tqdm(act_files, desc="Processing shards")):
            acts = load_npy(af, mmap=True)
            shard_offset = offsets[si]
            
            for start in range(0, acts.shape[0], batch):
                end = min(start + batch, acts.shape[0])
                x_batch = torch.from_numpy(acts[start:end].astype(np.float32)).to(device)
                
                z = sae.encode(x_batch)  # [batch, K]
                z_np = z.cpu().numpy()
                
                # Process each latent
                for j in range(K_LATENTS):
                    z_j = z_np[:, j]
                    active_mask = z_j > 0.0
                    
                    if not active_mask.any():
                        continue
                    
                    active_indices = np.where(active_mask)[0]
                    active_vals = z_j[active_indices]
                    global_indices = shard_offset + start + active_indices
                    
                    # Merge with existing top-k for this latent
                    all_vals = np.concatenate([top_vals[j], active_vals])
                    all_idx = np.concatenate([top_idx[j], global_indices.astype(np.int64)])
                    
                    # Keep only top-k
                    if len(all_vals) > k:
                        keep = np.argpartition(all_vals, -k)[-k:]
                        top_vals[j] = all_vals[keep]
                        top_idx[j] = all_idx[keep]
                    else:
                        top_vals[j] = all_vals
                        top_idx[j] = all_idx
    
    # Convert to sets
    return [set(idx_array.tolist()) for idx_array in top_idx]


def top1_tokens_for_latents(
    sae: SparseAutoencoder,
    medoids: np.ndarray,
    act_files: List[str],
    tok_files: List[str], 
    device: torch.device,
    batch: int = 16384
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the single most activating token for each medoid latent.
    
    Args:
        sae: Trained SAE model
        medoids: Array of medoid latent indices
        act_files: List of activation shard file paths
        tok_files: List of corresponding token files
        device: Device for inference
        batch: Batch size for processing
        
    Returns:
        Tuple of (token_ids, activation_values) for each medoid
    """
    sae.eval()
    sae = sae.to(device)
    
    # Track best token for each medoid
    best_vals = np.full(len(medoids), -1.0, dtype=np.float32)
    best_toks = np.full(len(medoids), -1, dtype=np.int32)
    
    offsets = [0]
    for af in act_files:
        size = load_npy(af, mmap=True).shape[0]
        offsets.append(offsets[-1] + size)
    
    with torch.no_grad():
        for si, (af, tf) in enumerate(zip(act_files, tok_files)):
            acts = load_npy(af, mmap=True)
            toks = load_npy(tf, mmap=True)
            
            for start in range(0, acts.shape[0], batch):
                end = min(start + batch, acts.shape[0])
                x_batch = torch.from_numpy(acts[start:end].astype(np.float32)).to(device)
                
                z = sae.encode(x_batch)  # [batch, K]
                z_medoids = z[:, medoids].cpu().numpy()  # [batch, M]
                
                tok_batch = toks[start:end]
                
                # Check each medoid
                for m_idx, latent_j in enumerate(medoids):
                    z_vals = z_medoids[:, m_idx]
                    max_idx = np.argmax(z_vals)
                    max_val = z_vals[max_idx]
                    
                    if max_val > best_vals[m_idx]:
                        best_vals[m_idx] = max_val
                        best_toks[m_idx] = tok_batch[max_idx]
    
    return best_toks, best_vals
