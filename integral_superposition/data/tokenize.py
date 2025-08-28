"""
Tokenization utilities.
"""

import torch
from typing import Dict, List
from ..backends.base_model import CausalLMBackend


def collate_texts(
    texts: List[str],
    backend: CausalLMBackend, 
    max_len: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and collate a batch of texts.
    
    Args:
        texts: List of text strings
        backend: Model backend for tokenization
        max_len: Maximum sequence length
        
    Returns:
        Dictionary with tokenized batch data
    """
    return backend.tokenize(texts, max_len)
