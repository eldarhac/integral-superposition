"""
Core type definitions for integral_superposition package.
"""

import numpy as np
from typing import Protocol
from typing_extensions import TypeAlias

# Core data types
ActivationShard: TypeAlias = np.ndarray  # [N_tokens, d_model]
Latents: TypeAlias = np.ndarray          # [N_tokens, K]
MergeMatrix: TypeAlias = np.ndarray      # [K_old, K_new]


class Reducer2D(Protocol):
    """Protocol for 2D dimensionality reduction algorithms."""
    
    def fit(self, X: np.ndarray) -> "Reducer2D":
        """Fit the reducer to the data."""
        ...
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to 2D representation."""
        ...
