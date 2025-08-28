"""
Activation dumping functionality.
"""

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Iterator
from tqdm.auto import tqdm

from ..backends.base_model import CausalLMBackend
from ..backends.io_store import save_npy, load_npy, shard_paths
from ..config import DumpConfig


class TitlesDataset(Dataset):
    """Dataset for title texts and labels."""
    
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return str(self.texts[idx]), int(self.labels[idx])


def dump_layer_activations(
    df: pd.DataFrame,
    text_col: str,
    label_col: str, 
    backend: CausalLMBackend,
    cfg: DumpConfig,
    out_dir: str
) -> None:
    """
    Dump hidden activations from a specific layer to disk shards.
    
    Args:
        df: DataFrame with text and label columns
        text_col: Name of text column
        label_col: Name of label column  
        backend: Model backend
        cfg: Dumping configuration
        out_dir: Output directory for shards
    """
    os.makedirs(out_dir, exist_ok=True)
    
    def collate_fn(batch):
        texts, labels = zip(*batch)
        tokenized = backend.tokenize(list(texts), cfg.max_len)
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        return tokenized
    
    dataset = TitlesDataset(
        df[text_col].tolist(),
        df[label_col].tolist()
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    device = backend.device
    all_count = 0
    shard_i = 0
    
    # Map dtype string to torch dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    
    for batch_i, batch in enumerate(tqdm(loader, desc="Dumping activations")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Use autocast if available and appropriate
        if torch.cuda.is_available() and dtype in [torch.bfloat16, torch.float16]:
            with torch.amp.autocast('cuda', dtype=dtype):
                hidden_states = backend.forward_hidden(
                    input_ids, attention_mask, cfg.layer
                )
        else:
            hidden_states = backend.forward_hidden(
                input_ids, attention_mask, cfg.layer
            )
        
        B, T, D = hidden_states.shape
        
        # Keep only valid tokens per attention mask
        mask = attention_mask.bool()
        hs_flat = hidden_states[mask].to("cpu").to(torch.float32)  # [N_tokens, D]
        labels_flat = labels.unsqueeze(1).expand(B, T)[mask].to("cpu").numpy().astype("int16")
        tokids = input_ids.to("cpu").numpy().astype("int32")[mask.cpu().numpy()]
        
        # Create title indices for each token
        titles = np.repeat(
            np.arange(batch_i * B, batch_i * B + B, dtype=np.int32),
            repeats=attention_mask.sum(dim=1).cpu().numpy()
        )[:hs_flat.shape[0]]
        
        # Save shard
        prefix = f"shard_{shard_i:05d}"
        save_npy(os.path.join(out_dir, f"{prefix}_acts.npy"), hs_flat.numpy())
        save_npy(os.path.join(out_dir, f"{prefix}_labels.npy"), labels_flat)
        save_npy(os.path.join(out_dir, f"{prefix}_tokids.npy"), tokids)
        save_npy(os.path.join(out_dir, f"{prefix}_titleidx.npy"), titles)
        
        shard_i += 1
        all_count += hs_flat.shape[0]
        
        # Clean up GPU memory
        del hidden_states, hs_flat, labels_flat, tokids, titles
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Total tokens dumped: {all_count} | shards in {out_dir}")


def build_offsets(act_files: List[str]) -> np.ndarray:
    """
    Build cumulative offsets for shard files.
    
    Args:
        act_files: List of activation shard file paths
        
    Returns:
        Cumulative size offsets
    """
    sizes = [load_npy(f, mmap=True).shape[0] for f in act_files]
    return np.cumsum([0] + sizes)


def stream_shards(act_files: List[str], batch: int = 16384) -> Iterator[np.ndarray]:
    """
    Stream activation data from shard files in batches.
    
    Args:
        act_files: List of activation shard file paths
        batch: Batch size for streaming
        
    Yields:
        Batches of activation data
    """
    for file in act_files:
        acts = load_npy(file, mmap=True)
        for start in range(0, acts.shape[0], batch):
            end = min(start + batch, acts.shape[0])
            yield acts[start:end].astype(np.float32)
