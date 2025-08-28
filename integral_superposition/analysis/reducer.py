"""
Dimensionality reduction utilities.
"""

import numpy as np
from sklearn.decomposition import PCA
from ..types import Reducer2D

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class PCAReducer:
    """PCA-based dimensionality reducer."""
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
    
    def fit(self, X: np.ndarray) -> "PCAReducer":
        """Fit PCA to the data."""
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensions."""
        if self.pca is None:
            raise ValueError("Must call fit() before transform()")
        return self.pca.transform(X)


if HAS_UMAP:
    class UMAPReducer:
        """UMAP-based dimensionality reducer."""
        
        def __init__(
            self,
            n_components: int = 2,
            random_state: int = 42,
            n_neighbors: int = 25,
            min_dist: float = 0.1,
            metric: str = "cosine"
        ):
            self.n_components = n_components
            self.random_state = random_state
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.metric = metric
            self.umap_model = None
        
        def fit(self, X: np.ndarray) -> "UMAPReducer":
            """Fit UMAP to the data."""
            self.umap_model = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric
            )
            self.umap_model.fit(X)
            return self
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform data to lower dimensions."""
            if self.umap_model is None:
                raise ValueError("Must call fit() before transform()")
            return self.umap_model.transform(X)


def create_reducer(
    method: str = "pca",
    n_components: int = 2, 
    random_state: int = 42,
    **kwargs
) -> Reducer2D:
    """
    Create a dimensionality reducer.
    
    Args:
        method: "pca" or "umap"
        n_components: Number of output dimensions
        random_state: Random seed
        **kwargs: Additional method-specific parameters
        
    Returns:
        Reducer implementing Reducer2D protocol
    """
    if method == "pca":
        return PCAReducer(n_components, random_state)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        return UMAPReducer(n_components, random_state, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")
