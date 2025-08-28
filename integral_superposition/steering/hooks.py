"""
Model steering and intervention utilities.
"""

import torch
from typing import Callable, List, Optional
from torch.utils.hooks import RemovableHandle

from ..backends.base_model import CausalLMBackend
from ..sae.model import SparseAutoencoder


def register_prehook_last_row(
    backend: CausalLMBackend,
    layer_idx: int,
    sae: SparseAutoencoder,
    latent_map: Callable[[torch.Tensor], torch.Tensor]
) -> RemovableHandle:
    """
    Register a pre-hook that modifies the last token's hidden states using SAE.
    
    Args:
        backend: Model backend
        layer_idx: Layer index to intervene on
        sae: Trained SAE for encoding/decoding
        latent_map: Function to modify latent activations z -> z'
        
    Returns:
        Removable hook handle
    """
    # Access the appropriate layer based on backend type
    # This assumes HuggingFace-style models with .model.layers
    if hasattr(backend, 'model') and hasattr(backend.model, 'model'):
        # Gemma/LLaMA-style: backend.model.model.layers
        target_layer = backend.model.model.layers[layer_idx]
    elif hasattr(backend, 'model') and hasattr(backend.model, 'transformer'):
        # GPT-2 style: backend.model.transformer.h
        target_layer = backend.model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Unsupported model architecture for steering")
    
    def pre_hook(module, inputs):
        """Pre-hook that modifies input hidden states."""
        hidden_states = inputs[0]  # [B, T, D]
        
        if hidden_states.dim() != 3:
            return inputs  # Skip if unexpected shape
        
        B, T, D = hidden_states.shape
        
        # Extract last token activations
        last_token = hidden_states[:, -1, :]  # [B, D]
        
        # Apply SAE intervention
        with torch.no_grad():
            last_token = last_token.to(next(sae.parameters()).device).float()
            
            # Encode to latents
            z = sae.encode(last_token)  # [B, K]
            
            # Apply latent modification
            z_modified = latent_map(z)  # [B, K]
            
            # Decode back to activation space
            x_modified = sae.decode(z_modified)  # [B, D]
            
            # Convert back to original dtype and device
            x_modified = x_modified.to(hidden_states.dtype).to(hidden_states.device)
        
        # Replace last token in hidden states
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = x_modified
        
        return (hidden_states,) + inputs[1:]
    
    # Register the pre-hook
    handle = target_layer.register_forward_pre_hook(pre_hook, with_kwargs=False)
    return handle


def merge_wrapper(
    enc: Callable[[torch.Tensor], torch.Tensor],
    M: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap an encoder function to apply latent merging.
    
    Args:
        enc: Original encoder function z = enc(x)
        M: Merge matrix [K, K']
        
    Returns:
        Wrapped encoder that returns merged latents z' = z @ M
    """
    def wrapped_encoder(x: torch.Tensor) -> torch.Tensor:
        z = enc(x)  # [*, K]
        z_merged = z @ M  # [*, K']
        return z_merged
    
    return wrapped_encoder


def set_or_scale_latent(
    j: int,
    set_to: Optional[float] = None,
    scale_by: Optional[float] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a latent modification function for a specific latent index.
    
    Args:
        j: Latent index to modify
        set_to: Value to set latent to (overrides current value)
        scale_by: Factor to multiply current latent value by
        
    Returns:
        Function that modifies latent j in the input tensor
    """
    def latent_modifier(z: torch.Tensor) -> torch.Tensor:
        z = z.clone()  # Don't modify in-place
        
        if set_to is not None:
            z[:, j] = set_to
        elif scale_by is not None:
            z[:, j] = z[:, j] * scale_by
            
        return z
    
    return latent_modifier


def multi_latent_modifier(
    latent_indices: List[int],
    set_to: Optional[float] = None,
    scale_by: Optional[float] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a modification function for multiple latents.
    
    Args:
        latent_indices: List of latent indices to modify
        set_to: Value to set latents to
        scale_by: Factor to multiply latent values by
        
    Returns:
        Function that modifies specified latents
    """
    def latent_modifier(z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        
        for j in latent_indices:
            if j < z.shape[1]:  # Bounds check
                if set_to is not None:
                    z[:, j] = set_to
                elif scale_by is not None:
                    z[:, j] = z[:, j] * scale_by
        
        return z
    
    return latent_modifier
