"""
Sparse Autoencoder model implementation.
"""

import math
import torch
from torch import nn
from typing import Tuple


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with ReLU activation and tied/untied decoder."""
    
    def __init__(self, d_model: int, k: int):
        """
        Initialize Sparse Autoencoder.
        
        Args:
            d_model: Input/output dimensionality
            k: Number of latent dimensions
        """
        super().__init__()
        self.d_model = d_model
        self.k = k
        
        # Encoder: linear + bias
        self.enc = nn.Linear(d_model, k, bias=True)
        
        # Decoder: linear without bias (common in SAEs)
        self.dec = nn.Linear(k, d_model, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.dec.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space with ReLU activation.
        
        Args:
            x: Input tensor [*, d_model]
            
        Returns:
            Latent activations [*, k]
        """
        z_pre = self.enc(x)
        return torch.relu(z_pre)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent activations to reconstruction.
        
        Args:
            z: Latent activations [*, k]
            
        Returns:
            Reconstructed input [*, d_model]
        """
        return self.dec(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor [*, d_model]
            
        Returns:
            Tuple of (reconstruction, latent_activations)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
