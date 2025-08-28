"""
Configuration classes for integral_superposition package.
"""

from pydantic import BaseModel
from typing import Optional


class DumpConfig(BaseModel):
    """Configuration for activation dumping."""
    layer: int = 13
    max_len: int = 32
    batch_size: int = 64
    dtype: str = "bf16"


class SAEConfig(BaseModel):
    """Configuration for Sparse Autoencoder training."""
    k: int = 2048  # Number of latent dimensions
    lr: float = 3e-4  # Learning rate
    l1: float = 1e-3  # L1 sparsity regularization
    epochs: int = 2
    batch_size: int = 4096


class Paths(BaseModel):
    """File paths configuration."""
    shards_dir: str = "acts_shards"
    sae_path: str = "sae_weights.pt"
    summary_csv: str = "latent_summary.csv"


class AnalysisConfig(BaseModel):
    """Configuration for latent analysis."""
    topk: int = 200  # Top-K tokens per latent
    min_activation: float = 0.0  # Minimum activation threshold
    similarity_threshold: float = 0.85  # Clustering threshold


class MergeConfig(BaseModel):
    """Configuration for latent merging and clustering."""
    n_clusters: Optional[int] = None  # Auto-determined if None
    cosine_weight: float = 0.5
    correlation_weight: float = 0.4
    jaccard_weight: float = 0.1
    l2_reg: float = 1e-3
