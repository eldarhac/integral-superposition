"""
Enrichment analysis for latent activations.
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy.stats import fisher_exact


def fisher_enrichment(
    top_idx: Dict[int, np.ndarray], 
    labels_global: np.ndarray
) -> pd.DataFrame:
    """
    Compute Fisher exact test enrichment for each latent.
    
    Args:
        top_idx: Dictionary mapping latent indices to top-activating token indices
        labels_global: Global labels array aligned with token indices
        
    Returns:
        DataFrame with enrichment statistics per latent
    """
    base_sports = int((labels_global == 1).sum())
    base_nonsp = int((labels_global == 0).sum())
    
    rows = []
    
    for latent_j, token_indices in top_idx.items():
        if len(token_indices) == 0:
            sports_frac = 0.0
            odds = 0.0
            pval = 1.0
            kcount = 0
        else:
            # Get labels for top-activating tokens
            labs = labels_global[token_indices]
            kcount = len(labs)
            sports = int((labs == 1).sum())
            nonsp = int((labs == 0).sum())
            
            # Contingency table:
            # [[sports_in_topK, nonsp_in_topK], 
            #  [sports_rest, nonsp_rest]]
            sports_rest = max(0, base_sports - sports)
            nonsp_rest = max(0, base_nonsp - nonsp)
            
            try:
                odds, pval = fisher_exact(
                    [[sports, nonsp], [sports_rest, nonsp_rest]],
                    alternative="two-sided"
                )
            except Exception:
                odds, pval = (0.0, 1.0)
            
            sports_frac = float(sports) / max(1, kcount)
        
        rows.append({
            "latent": int(latent_j),
            "topK_count": int(kcount),
            "sports_frac_topK": float(sports_frac),
            "odds_ratio": float(odds),
            "fisher_p": float(pval),
        })
    
    return pd.DataFrame(rows).sort_values(
        ["topK_count", "sports_frac_topK"], 
        ascending=[False, False]
    )
