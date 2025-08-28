"""
Sparse Jaccard similarity computation.
"""

import numpy as np
from typing import List, Set
from scipy.sparse import coo_matrix, csr_matrix


def jaccard_from_sets(top_idx: List[Set[int]], K: int) -> csr_matrix:
    """
    Compute sparse Jaccard similarity matrix from token sets.
    
    Args:
        top_idx: List of token index sets, one per latent
        K: Number of latents
        
    Returns:
        Sparse Jaccard similarity matrix [K, K]
    """
    # Build sparse membership matrix A: rows = tokens, cols = latents
    # A[i,j] = 1 if token i is in latent j's top set
    
    # Collect all unique tokens
    all_tokens = set()
    for token_set in top_idx:
        all_tokens.update(token_set)
    
    if not all_tokens:
        # Return identity matrix if no tokens
        return csr_matrix(np.eye(K))
    
    # Map tokens to row indices
    token_list = sorted(all_tokens)
    token_to_row = {tok: i for i, tok in enumerate(token_list)}
    n_tokens = len(token_list)
    
    # Build COO matrix data
    rows = []
    cols = []
    data = []
    
    for j, token_set in enumerate(top_idx):
        for token in token_set:
            rows.append(token_to_row[token])
            cols.append(j)
            data.append(1)
    
    # Create sparse membership matrix A
    A = coo_matrix((data, (rows, cols)), shape=(n_tokens, K)).tocsr()
    
    # Compute co-occurrence matrix C = A.T @ A
    C = (A.T @ A).tocoo()
    
    # Compute Jaccard: J[i,j] = |S_i ∩ S_j| / |S_i ∪ S_j|
    # |S_i ∪ S_j| = |S_i| + |S_j| - |S_i ∩ S_j|
    degrees = np.asarray(A.sum(axis=0)).ravel()  # |S_j| for each latent
    
    intersection = C.data.astype(np.float32)  # |S_i ∩ S_j|
    degree_i = degrees[C.row]  # |S_i|
    degree_j = degrees[C.col]  # |S_j|
    union = degree_i + degree_j - intersection  # |S_i ∪ S_j|
    
    # Avoid division by zero
    union = np.maximum(union, 1.0)
    jaccard_data = intersection / union
    
    # Build sparse Jaccard matrix
    J = coo_matrix((jaccard_data, (C.row, C.col)), shape=(K, K)).tocsr()
    
    # Set diagonal to 1.0 (perfect self-similarity)
    J.setdiag(1.0)
    
    return J


def combine_similarity(
    cosW: np.ndarray,
    corr: np.ndarray, 
    J: csr_matrix,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1
) -> csr_matrix:
    """
    Combine cosine, correlation, and Jaccard similarities.
    
    Args:
        cosW: Cosine similarity matrix [K, K]
        corr: Correlation matrix [K, K] 
        J: Sparse Jaccard similarity matrix [K, K]
        alpha: Weight for cosine similarity
        beta: Weight for correlation
        gamma: Weight for Jaccard similarity
        
    Returns:
        Combined sparse similarity matrix
    """
    # Convert dense matrices to sparse for consistent operations
    cosW_sparse = csr_matrix(cosW)
    corr_sparse = csr_matrix(corr)
    
    # Combine with weights
    S = alpha * cosW_sparse + beta * corr_sparse + gamma * J
    
    # Ensure diagonal is 1.0
    S.setdiag(1.0)
    
    return S
