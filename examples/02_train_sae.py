#!/usr/bin/env python3
"""
Example: Train a Sparse Autoencoder on dumped activations.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from integral_superposition import (
    SAEConfig, Paths,
    sae, backends, dump
)


def main():
    """Train SAE example."""
    
    # Configuration
    cfg = SAEConfig(
        k=2048,
        lr=3e-4,
        l1=1e-3,
        epochs=2,
        batch_size=4096
    )
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt",
        summary_csv="latent_summary.csv"
    )
    
    # Check if activation shards exist
    act_files = backends.io_store.shard_paths(paths.shards_dir, "acts")
    if not act_files:
        print(f"No activation shards found in {paths.shards_dir}")
        print("Run 01_dump_activations.py first")
        return
    
    print(f"Found {len(act_files)} activation shards")
    
    # Get model dimensions from first shard
    sample_acts = backends.io_store.load_npy(act_files[0], mmap=True)
    d_model = sample_acts.shape[1]
    print(f"d_model: {d_model}, k_latents: {cfg.k}")
    
    # Create SAE model
    model = sae.SparseAutoencoder(d_model, cfg.k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create data iterator
    def shard_iterator():
        """Iterator over activation shards."""
        return dump.stream_shards(act_files, batch=cfg.batch_size)
    
    # Train SAE
    print("Training SAE...")
    sae.train_sae(
        sae=model,
        shard_iter=shard_iterator,
        cfg=cfg,
        device=device
    )
    
    # Save trained model
    print(f"Saving SAE to {paths.sae_path}")
    sae.save_sae(model, paths.sae_path)
    
    print("SAE training complete!")


if __name__ == "__main__":
    main()
