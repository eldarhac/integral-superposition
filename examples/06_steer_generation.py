#!/usr/bin/env python3
"""
Example: Steer text generation using SAE latent interventions.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from integral_superposition import (
    DumpConfig, Paths, sae, backends, steering
)


def main():
    """Steering generation example."""
    
    # Configuration
    model_id = "google/gemma-3-270m"  # Change to your model
    layer_idx = 13  # Should match dump configuration
    
    paths = Paths(
        shards_dir="acts_shards",
        sae_path="sae_weights.pt",
        summary_csv="latent_summary.csv"
    )
    
    # Load model backend
    print(f"Loading model: {model_id}")
    backend = backends.HFCausalLM.from_pretrained(model_id, dtype="bf16", device_map="auto")
    
    # Load SAE
    if not os.path.exists(paths.sae_path):
        print(f"SAE not found at {paths.sae_path}")
        return
    
    print("Loading SAE...")
    model = sae.load_sae(paths.sae_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load latent summary to find interesting latents
    if os.path.exists(paths.summary_csv):
        summary = pd.read_csv(paths.summary_csv)
        # Get top sports-enriched latents
        sports_latents = summary.head(5)['latent'].tolist()
        print(f"Using sports latents: {sports_latents}")
    else:
        print("No summary found, using default latents")
        sports_latents = [0, 1, 2]  # Default fallback
    
    # Test prompts
    test_prompts = [
        "The Manchester",
        "the first round of",
        "The championship game",
        "Breaking news:"
    ]
    
    def generate_with_steering(prompt: str, latents: list, intervention: float):
        """Generate text with latent steering."""
        
        # Create latent modification function
        latent_modifier = steering.multi_latent_modifier(
            latent_indices=latents,
            set_to=intervention
        )
        
        # Register steering hook
        handle = steering.register_prehook_last_row(
            backend=backend,
            layer_idx=layer_idx,
            sae=model,
            latent_map=latent_modifier
        )
        
        try:
            # Tokenize prompt
            inputs = backend.tokenize([prompt], max_len=32)
            inputs = {k: v.to(backend.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = backend.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=backend.tokenizer.eos_token_id
                )
            
            # Decode
            generated = backend.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
            
        finally:
            handle.remove()
    
    # Test different intervention strengths
    print("\\n=== STEERING EXPERIMENTS ===")
    
    for prompt in test_prompts:
        print(f"\\nPrompt: '{prompt}'")
        
        # Baseline (no intervention)
        baseline = generate_with_steering(prompt, sports_latents, 0.0)
        print(f"Baseline:     {baseline}")
        
        # Suppress sports (negative intervention)
        suppressed = generate_with_steering(prompt, sports_latents, -200.0)
        print(f"Suppressed:   {suppressed}")
        
        # Enhance sports (positive intervention)  
        enhanced = generate_with_steering(prompt, sports_latents, 200.0)
        print(f"Enhanced:     {enhanced}")
    
    print("\\n=== ITERATIVE STEERING ===")
    
    # Example of iterative steering
    prompt = "The Manchester"
    print(f"Starting prompt: '{prompt}'")
    
    for i in range(5):
        prompt = generate_with_steering(prompt, sports_latents, 200.0)
        print(f"Step {i+1}: {prompt}")
    
    print("\\nSteering experiments complete!")
    

def test_latent_activations():
    """Test function to show latent activations for sample text."""
    
    paths = Paths(sae_path="sae_weights.pt")
    
    if not os.path.exists(paths.sae_path):
        return
    
    # Load SAE
    model = sae.load_sae(paths.sae_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load backend for tokenization
    backend = backends.HFCausalLM.from_pretrained("google/gemma-3-270m", dtype="bf16")
    
    test_texts = [
        "Manchester United won the game",
        "The weather is nice today",
        "Basketball tournament finals",
        "Technology news update"
    ]
    
    print("\\n=== LATENT ACTIVATIONS ===")
    
    for text in test_texts:
        # Tokenize
        inputs = backend.tokenize([text], max_len=32)
        
        # Get activations (this would need the actual model forward pass)
        # For demo purposes, we'll just show the concept
        print(f"Text: '{text}'")
        print("  -> Would show top-activating latents here")


if __name__ == "__main__":
    main()
    # test_latent_activations()  # Uncomment to test activations
