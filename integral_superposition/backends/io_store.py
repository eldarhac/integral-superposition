"""
File system and cloud storage utilities.
"""

import os
import glob
import numpy as np
from typing import List


def save_npy(path: str, arr: np.ndarray) -> None:
    """
    Save numpy array to disk.
    
    Args:
        path: File path to save to
        arr: Numpy array to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_npy(path: str, mmap: bool = True) -> np.ndarray:
    """
    Load numpy array from disk.
    
    Args:
        path: File path to load from
        mmap: Whether to use memory mapping for large files
        
    Returns:
        Loaded numpy array
    """
    if mmap:
        return np.load(path, mmap_mode="r")
    else:
        return np.load(path)


def shard_paths(dir: str, stem: str) -> List[str]:
    """
    Get sorted list of shard file paths.
    
    Args:
        dir: Directory containing shard files
        stem: File stem (e.g., "acts", "labels")
        
    Returns:
        Sorted list of shard file paths
    """
    pattern = os.path.join(dir, f"*_{stem}.npy")
    return sorted(glob.glob(pattern))
