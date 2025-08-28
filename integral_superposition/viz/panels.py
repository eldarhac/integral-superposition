"""
Interactive visualization panels for latent analysis.
"""

import numpy as np
import torch
from typing import List
from IPython.display import HTML, display

from ..sae.model import SparseAutoencoder
from ..backends.io_store import load_npy


def top_activations_panel(
    latent_j: int,
    act_files: List[str],
    tok_files: List[str], 
    sae: SparseAutoencoder,
    tokenizer,
    device: torch.device,
    top_titles: int = 12,
    max_tokens_per_title: int = 64,
    min_alpha: float = 0.0
) -> None:
    """
    Render HTML panel showing top-activating tokens for a latent.
    
    Args:
        latent_j: Latent index to analyze
        act_files: List of activation shard files
        tok_files: List of corresponding token files
        sae: Trained SAE model
        tokenizer: Tokenizer for decoding tokens
        device: Device for inference
        top_titles: Number of top titles to show
        max_tokens_per_title: Maximum tokens to display per title
        min_alpha: Minimum activation threshold for highlighting
    """
    # Get global activations and build title mapping
    z_all, tok_all, ttl_all = _compute_latent_scores_for_all_tokens(
        latent_j, act_files, tok_files, sae, device
    )
    
    # Get top titles by max activation
    ranked_titles = _titles_ranked_by_latent(z_all, ttl_all, top_titles)
    
    # Render HTML blocks
    html_blocks = []
    for title_id, max_activation in ranked_titles:
        html_block = _render_title_block(
            title_id, z_all, tok_all, ttl_all, tokenizer, 
            max_tokens_per_title, min_alpha
        )
        html_blocks.append(html_block)
    
    # Combine into final HTML
    header = f"<h3 style='margin:0 0 8px 0'>Top activations · latent #{latent_j}</h3>"
    note = "<div style='color:#666; font-size:12px; margin-bottom:8px'>Highlight intensity ∝ latent activation on token. Rows = titles with highest max activation.</div>"
    html = header + note + "".join(html_blocks)
    
    display(HTML(html))


def _compute_latent_scores_for_all_tokens(
    latent_j: int,
    act_files: List[str],
    tok_files: List[str],
    sae: SparseAutoencoder,
    device: torch.device,
    batch_size: int = 16384
) -> tuple:
    """Compute latent scores for all tokens across shards."""
    sae.eval()
    sae = sae.to(device)
    
    # Compute shard sizes and offsets
    shard_sizes = [load_npy(f, mmap=True).shape[0] for f in act_files]
    offsets = np.cumsum([0] + shard_sizes)
    total_tokens = sum(shard_sizes)
    
    # Allocate arrays
    z_all = np.zeros(total_tokens, dtype=np.float32)
    tok_all = np.zeros(total_tokens, dtype=np.int32)
    ttl_all = np.zeros(total_tokens, dtype=np.int32)
    
    cursor = 0
    with torch.no_grad():
        for si, (af, tf, ttf) in enumerate(zip(act_files, tok_files, 
                                               [f.replace("_tokids.npy", "_titleidx.npy") 
                                                for f in tok_files])):
            acts = load_npy(af, mmap=True)
            toks = load_npy(tf, mmap=True).astype(np.int32)
            ttls = load_npy(ttf, mmap=True).astype(np.int32)
            
            for start in range(0, acts.shape[0], batch_size):
                end = min(start + batch_size, acts.shape[0])
                x_batch = torch.from_numpy(acts[start:end].astype(np.float32)).to(device)
                
                z = sae.encode(x_batch)
                z_j = z[:, latent_j].cpu().numpy()
                
                batch_size_actual = end - start
                z_all[cursor:cursor + batch_size_actual] = z_j
                tok_all[cursor:cursor + batch_size_actual] = toks[start:end]
                ttl_all[cursor:cursor + batch_size_actual] = ttls[start:end]
                cursor += batch_size_actual
    
    return z_all, tok_all, ttl_all


def _titles_ranked_by_latent(z_all: np.ndarray, ttl_all: np.ndarray, top_k: int) -> List[tuple]:
    """Get top-k titles ranked by maximum latent activation."""
    unique_titles = np.unique(ttl_all)
    title_max_acts = []
    
    for title_id in unique_titles:
        mask = ttl_all == title_id
        max_act = z_all[mask].max()
        title_max_acts.append((int(title_id), float(max_act)))
    
    # Sort by max activation descending
    title_max_acts.sort(key=lambda x: x[1], reverse=True)
    return title_max_acts[:top_k]


def _render_title_block(
    title_id: int,
    z_all: np.ndarray,
    tok_all: np.ndarray, 
    ttl_all: np.ndarray,
    tokenizer,
    max_tokens: int,
    min_alpha: float
) -> str:
    """Render HTML for a single title with token highlighting."""
    mask = ttl_all == title_id
    title_toks = tok_all[mask][:max_tokens]
    title_zs = z_all[mask][:max_tokens]
    
    # Normalize activations for this title
    z_max = float(title_zs.max()) if len(title_zs) else 0.0
    z_min = float(title_zs[title_zs > 0].min()) if (title_zs > 0).any() else 0.0
    
    # Render tokens with highlighting
    token_spans = []
    for tok_id, z_val in zip(title_toks, title_zs):
        token_str = _decode_token_safe(tokenizer, tok_id)
        
        if z_val <= min_alpha or z_max == 0:
            token_spans.append(f"<span>{token_str}</span>")
        else:
            style = _html_color_style(z_val, z_min if z_min > 0 else 0.0, z_max)
            token_spans.append(f"<span style='{style}'>{token_str}</span>")
    
    block = f"<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.6; margin:10px 0; padding:8px; border-left:4px solid #eee;'>{''.join(token_spans)}</div>"
    return block


def _decode_token_safe(tokenizer, tok_id: int) -> str:
    """Safely decode token ID to string."""
    try:
        s = tokenizer.decode([int(tok_id)], skip_special_tokens=True)
        # Escape HTML characters
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return s
    except Exception:
        return str(tok_id)


def _html_color_style(val: float, vmin: float, vmax: float) -> str:
    """Generate CSS style for color-coding activation intensity."""
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    
    # Color gradient: pale yellow -> orange -> red
    if t < 0.33:
        # Pale yellow to orange  
        u = t / 0.33
        r, g, b = 255, int(246 - 96 * u), int(204 - 153 * u)
    elif t < 0.66:
        # Orange to coral
        u = (t - 0.33) / 0.33
        r, g, b = 255, int(184 - 176 * u), int(77 + 35 * u)
    else:
        # Coral to red
        u = (t - 0.66) / 0.34
        r, g, b = 255, int(112 - 112 * u), int(77 - 77 * u)
    
    return f"background-color: rgb({r}, {g}, {b}); padding:2px 1px; border-radius:3px;"
