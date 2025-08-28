#!/usr/bin/env python3
"""
Example: Merge similar latents and refit decoder.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from integral_superposition import (
    Paths, sae, backends, merge, analysis
)


def main():
    """Merge and refit example."""
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt",
        summary_csv="latent_summary.csv"
    )
    
    # Load trained SAE
    if not os.path.exists(paths.sae_path):
        print(f"SAE weights not found at {paths.sae_path}")
        return
    
    print("Loading SAE...")
    model = sae.load_sae(paths.sae_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    K = model.k
    d_model = model.d_model
    print(f"Original SAE: {d_model} -> {K}")
    
    # Get decoder weights for clustering
    W = model.dec.weight.detach().cpu().numpy()  # [d_model, K]
    
    # Compute cosine distance matrix
    print("Computing cosine distances...")
    cosine_dist = merge.cluster.cosine_dist_from_decoder(W)
    
    # Determine number of clusters
    n_clusters = merge.cluster.auto_n_clusters(K, target_compression=0.125)
    print(f"Clustering {K} latents into {n_clusters} groups")
    
    # Perform clustering
    labels = merge.cluster.agglomerative_labels(cosine_dist, n_clusters)
    
    # Find medoids
    medoids = merge.medoid.medoids_from_labels(cosine_dist, labels)
    print(f"Selected {len(medoids)} medoids")
    
    # Create merge matrix
    M = merge.medoid.merge_matrix_from_labels(labels, medoids)
    print(f"Merge matrix shape: {M.shape}")
    
    # Print clustering statistics
    cluster_stats = merge.medoid.cluster_stats(labels)
    print("Clustering stats:", cluster_stats)
    
    # Get data for refitting
    act_files = backends.io_store.shard_paths(paths.shards_dir, "acts")
    if not act_files:
        print("No activation shards found")
        return
    
    # Create data iterator for refitting
    def activation_iterator():
        """Iterator over activation data."""
        for batch in backends.dump.stream_shards(act_files, batch=4096):
            if batch.shape[0] > 0:
                yield batch
    
    # Refit decoder
    print("Refitting decoder weights...")
    W_new = merge.refit.merge_and_refit_decoder(
        X_iter=activation_iterator,
        sae=model,
        M=M,
        l2=1e-3,
        device=str(device)
    )
    
    print(f"New decoder shape: {W_new.shape}")
    print(f"Compression ratio: {W_new.shape[1] / K:.3f}")
    
    # Create new SAE with merged decoder
    merged_sae = sae.SparseAutoencoder(d_model, len(medoids))
    
    # Copy encoder weights (will map to old latent space)
    merged_sae.enc.weight.data = model.enc.weight.data.clone()
    merged_sae.enc.bias.data = model.enc.bias.data.clone()
    
    # Set new decoder weights  
    merged_sae.dec.weight.data = torch.from_numpy(W_new.T)  # [K', d_model]
    
    # Save merged model
    merged_path = paths.sae_path.replace('.pt', '_merged.pt')
    sae.save_sae(merged_sae, merged_path)
    print(f"Merged SAE saved to {merged_path}")
    
    # Save merge matrix
    merge_matrix_path = "merge_matrix.npy"
    np.save(merge_matrix_path, M)
    print(f"Merge matrix saved to {merge_matrix_path}")
    
    print("Merge and refit complete!")


if __name__ == "__main__":
    main()
