#!/usr/bin/env python3
"""
Example: Create 2D visualizations of latent space.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from integral_superposition import (
    Paths, sae, backends, analysis, viz
)


def main():
    """Plot maps example."""
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt",
        summary_csv="latent_summary.csv"
    )
    
    # Load SAE
    if not os.path.exists(paths.sae_path):
        print(f"SAE not found at {paths.sae_path}")
        return
    
    model = sae.load_sae(paths.sae_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get activation files
    act_files = backends.io_store.shard_paths(paths.shards_dir, "acts")
    lab_files = backends.io_store.shard_paths(paths.shards_dir, "labels") 
    ttl_files = backends.io_store.shard_paths(paths.shards_dir, "titleidx")
    
    if not all([act_files, lab_files, ttl_files]):
        print("Missing shard files")
        return
    
    print("Computing title-level latent codes...")
    
    # Collect title-level latent codes (mean-pooled)
    per_title_zsum = {}
    per_title_counts = {}
    per_title_labels = {}
    
    with torch.no_grad():
        for af, lf, tf in zip(act_files[:3], lab_files[:3], ttl_files[:3]):  # Limit for demo
            acts = backends.io_store.load_npy(af, mmap=True)
            labs = backends.io_store.load_npy(lf, mmap=True).astype(np.int64)
            ttls = backends.io_store.load_npy(tf, mmap=True).astype(np.int64)
            
            # Process in batches
            batch_size = 2048
            for start in range(0, acts.shape[0], batch_size):
                end = min(start + batch_size, acts.shape[0])
                x_batch = torch.from_numpy(acts[start:end].astype(np.float32)).to(device)
                
                z = model.encode(x_batch).cpu().numpy()
                
                for i in range(z.shape[0]):
                    title_id = int(ttls[start + i])
                    label = int(labs[start + i])
                    
                    if title_id not in per_title_zsum:
                        per_title_zsum[title_id] = z[i] 
                        per_title_counts[title_id] = 1
                        per_title_labels[title_id] = label
                    else:
                        per_title_zsum[title_id] += z[i]
                        per_title_counts[title_id] += 1
    
    # Mean pool and create arrays
    title_ids = sorted(per_title_zsum.keys())
    Z_titles = np.stack([per_title_zsum[tid] / per_title_counts[tid] for tid in title_ids])
    y_titles = np.array([per_title_labels[tid] for tid in title_ids])
    
    print(f"Collected {len(title_ids)} titles")
    print("Label distribution:", dict(zip(*np.unique(y_titles, return_counts=True))))
    
    # Create 2D reducer
    print("Fitting 2D reducer...")
    reducer = analysis.reducer.create_reducer("pca", n_components=2, random_state=42)
    reducer.fit(Z_titles)
    
    # Plot titles map
    print("Creating titles map...")
    viz.maps.titles_map(Z_titles, y_titles, reducer, trim_p=95.0)
    
    # Load summary for medoid analysis
    if os.path.exists(paths.summary_csv):
        print("Loading latent summary...")
        summary = pd.read_csv(paths.summary_csv)
        
        # Get top sports-enriched latents as "medoids" 
        top_latents = summary.head(10)
        medoids = top_latents['latent'].values
        
        # Create labels for medoids
        labels_text = [f"L{int(lat)}" for lat in medoids]
        is_sport = (top_latents['sports_frac_topK'] > 0.6).values
        
        print("Creating medoids map...")
        viz.maps.latents_labels_map(
            reducer=reducer,
            Z_titles=Z_titles, 
            medoids=medoids,
            labels_text=labels_text,
            is_sport=is_sport
        )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
