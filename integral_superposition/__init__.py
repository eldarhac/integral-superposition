"""
Integral Superposition: Model-agnostic Sparse Autoencoder toolkit
"""

__version__ = "0.1.0"

from . import backends, data, dump, sae, analysis, merge, steering, viz
from .config import DumpConfig, SAEConfig, Paths
from .types import ActivationShard, Latents, MergeMatrix

__all__ = [
    "backends",
    "data", 
    "dump",
    "sae",
    "analysis",
    "merge",
    "steering",
    "viz",
    "DumpConfig",
    "SAEConfig", 
    "Paths",
    "ActivationShard",
    "Latents",
    "MergeMatrix",
]
