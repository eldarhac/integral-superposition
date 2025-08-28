#!/usr/bin/env python3
"""
Example: Dump activations from a language model.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integral_superposition import (
    DumpConfig, Paths,
    data, backends, dump
)


def main():
    """Dump activations example."""
    
    # Configuration
    model_id = "google/gemma-3-270m"  # Change to available model
    
    # Option 1: Use Kaggle dataset (matches original notebook)
    kaggle_dataset_id = "kotartemiy/topic-labeled-news-dataset"
    kaggle_filename = "labelled_newscatcher_dataset.csv"
    
    # Option 2: Use local dataset file
    dataset_path = "data/sample_dataset.csv"  # Change to your dataset
    
    cfg = DumpConfig(
        layer=13,
        max_len=32,
        batch_size=64,
        dtype="bf16"
    )
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt",
        summary_csv="latent_summary.csv"
    )
    
    # Load model backend
    print(f"Loading model: {model_id}")
    backend = backends.HFCausalLM.from_pretrained(
        model_id, 
        dtype=cfg.dtype, 
        device_map="auto"
    )
    print(f"Model loaded on device: {backend.device}")
    
    # Load dataset - try Kaggle first, then local file
    print("Loading dataset...")
    try:
        # Try loading from Kaggle (matches original notebook)
        df = data.load_kaggle_dataset(
            kaggle_dataset_id, 
            kaggle_filename, 
            text_col="title", 
            label_col="topic"
        )
        print(f"Loaded {len(df)} examples from Kaggle")
    except (ImportError, RuntimeError) as e:
        print(f"Kaggle loading failed: {e}")
        print(f"Trying local dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Please either:")
            print("1. Install kagglehub: pip install kagglehub")
            print("2. Create a CSV with 'title' and 'topic' columns at", dataset_path)
            return
        
        df = data.load_titles_csv(dataset_path, text_col="title", label_col="topic")
        print(f"Loaded {len(df)} examples from local file")
    
    print("Label distribution:", df["topic"].value_counts().to_dict())
    
    # Dump activations
    print(f"Dumping activations from layer {cfg.layer}...")
    os.makedirs(paths.shards_dir, exist_ok=True)
    
    dump.dump_layer_activations(
        df=df,
        text_col="title",
        label_col="topic", 
        backend=backend,
        cfg=cfg,
        out_dir=paths.shards_dir
    )
    
    print("Activation dumping complete!")
    print(f"Shards saved to: {paths.shards_dir}")


if __name__ == "__main__":
    main()
