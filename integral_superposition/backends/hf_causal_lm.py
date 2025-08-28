"""
HuggingFace Transformers implementation of CausalLMBackend.
"""

import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .base_model import CausalLMBackend


class HFCausalLM(CausalLMBackend):
    """HuggingFace Transformers backend for causal language models."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer  
        self.config = config
        self._device = next(model.parameters()).device
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dtype: str = "bf16",
        device_map: str = "auto"
    ) -> "HFCausalLM":
        """
        Load a HuggingFace model from pretrained weights.
        
        Args:
            model_id: Model identifier (e.g., "google/gemma-3-270m")
            dtype: Data type ("bf16", "fp16", "fp32")
            device_map: Device mapping strategy
            
        Returns:
            HFCausalLM instance
        """
        # Map string dtypes to torch dtypes
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16, 
            "fp32": torch.float32
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Check device capability for bfloat16
        if dtype == "bf16" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:  # Pre-Ampere GPUs don't support bf16 
                torch_dtype = torch.float16
        
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        model.eval()
        
        return cls(model, tokenizer, config)
    
    @property
    def device(self) -> torch.device:
        """Get the device where the model is located."""
        return self._device
    
    def tokenize(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
    
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        layer: int
    ) -> torch.Tensor:
        """Forward pass returning hidden states at a specific layer."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        return outputs.hidden_states[layer]
    
    def generate(self, **kwargs) -> torch.Tensor:
        """Generate text continuation."""
        with torch.no_grad():
            return self.model.generate(**kwargs)
