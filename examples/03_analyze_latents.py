#!/usr/bin/env python3
"""
Example: Analyze latent activations and compute enrichment statistics.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from integral_superposition import (
    Paths, sae, backends, analysis
)


def main():
    """Analyze latents example."""
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt", 
        summary_csv="latent_summary.csv"
    )
    
    # Load trained SAE
    if not os.path.exists(paths.sae_path):
        print(f"SAE weights not found at {paths.sae_path}")
        print("Run 02_train_sae.py first")
        return
    
    print("Loading trained SAE...")
    model = sae.load_sae(paths.sae_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"SAE loaded: {model.d_model} -> {model.k}")
    
    # Get shard files
    act_files = backends.io_store.shard_paths(paths.shards_dir, "acts")
    lab_files = backends.io_store.shard_paths(paths.shards_dir, "labels")
    
    if not act_files or not lab_files:
        print("Activation or label shards not found")
        return
    
    print(f"Found {len(act_files)} shards")
    
    # Build global labels array
    print("Building global labels array...")
    offsets = [0]
    labels_parts = []
    
    for lf in lab_files:
        labels_shard = backends.io_store.load_npy(lf, mmap=True).astype(np.int64)
        labels_parts.append(labels_shard)
        offsets.append(offsets[-1] + len(labels_shard))
    
    labels_global = np.concatenate(labels_parts)
    print(f"Total tokens: {len(labels_global)}")
    print("Label distribution:", dict(zip(*np.unique(labels_global, return_counts=True))))
    
    # Compute top-k token sets per latent
    print("Computing top-k activating tokens per latent...")
    top_k_sets = analysis.top_tokens.topk_sets_per_latent(
        sae=model,
        act_files=act_files,
        device=device,
        k=200
    )
    
    # Convert to dict for enrichment analysis
    top_idx_dict = {j: np.array(list(token_set), dtype=np.int64) 
                    for j, token_set in enumerate(top_k_sets)}
    
    # Compute enrichment statistics
    print("Computing enrichment statistics...")
    summary_df = analysis.enrichment.fisher_enrichment(top_idx_dict, labels_global)
    
    # Save summary
    summary_df.to_csv(paths.summary_csv, index=False)
    print(f"Summary saved to {paths.summary_csv}")
    
    # Show top results
    print("\nTop 10 most enriched latents:")
    print(summary_df.head(10)[['latent', 'topK_count', 'sports_frac_topK', 
                               'odds_ratio', 'fisher_p']])
    
    # Basic statistics
    active_latents = (summary_df['topK_count'] > 0).sum()
    print(f"\nActive latents: {active_latents}/{len(summary_df)}")
    print(f"Mean activations per latent: {summary_df['topK_count'].mean():.1f}")
    
    print("Latent analysis complete!")


if __name__ == "__main__":
    main()
