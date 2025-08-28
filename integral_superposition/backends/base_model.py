"""
Abstract base class for causal language model backends.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, List


class CausalLMBackend(ABC):
    """Abstract backend for causal language models."""
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device where the model is located."""
        pass
    
    @abstractmethod
    def tokenize(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings to tokenize
            max_len: Maximum sequence length
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and optionally other keys
        """
        pass
    
    @abstractmethod
    def forward_hidden(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        layer: int
    ) -> torch.Tensor:
        """
        Forward pass returning hidden states at a specific layer.
        
        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T] 
            layer: Layer index to extract activations from
            
        Returns:
            Hidden states tensor [B, T, D]
        """
        pass
    
    @abstractmethod  
    def generate(self, **kwargs) -> torch.Tensor:
        """
        Generate text continuation.
        
        Args:
            **kwargs: Generation parameters (input_ids, max_new_tokens, etc.)
            
        Returns:
            Generated token IDs
        """
        pass
