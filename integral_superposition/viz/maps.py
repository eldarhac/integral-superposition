"""
2D visualization maps for latent analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from ..types import Reducer2D
from ..analysis.stats import trim_outliers


def titles_map(
    Z_titles: np.ndarray,
    y: np.ndarray,
    reducer: Reducer2D,
    trim_p: float = 95.0,
    figsize: tuple = (8, 6)
) -> None:
    """
    Create 2D scatter plot of titles in latent space.
    
    Args:
        Z_titles: Title-level latent codes [N_titles, K]
        y: Binary labels for titles [N_titles]
        reducer: Fitted 2D reducer
        trim_p: Percentile for outlier trimming
        figsize: Figure size
    """
    # Reduce to 2D
    X2 = reducer.transform(Z_titles)
    
    # Trim outliers
    X2_trimmed = trim_outliers(X2, p=trim_p)
    
    # Find indices of kept points (simple approach - could be improved)
    if len(X2_trimmed) < len(X2):
        center = X2.mean(axis=0)
        distances = np.linalg.norm(X2 - center, axis=1)
        threshold = np.percentile(distances, trim_p)
        mask = distances <= threshold
        X2_plot = X2[mask]
        y_plot = y[mask]
    else:
        X2_plot = X2
        y_plot = y
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot non-sports (label 0) points
    mask_0 = y_plot == 0
    if mask_0.any():
        plt.scatter(
            X2_plot[mask_0, 0], X2_plot[mask_0, 1], 
            s=12, alpha=0.7, label="Non-Sports", c='lightblue'
        )
    
    # Plot sports (label 1) points  
    mask_1 = y_plot == 1
    if mask_1.any():
        plt.scatter(
            X2_plot[mask_1, 0], X2_plot[mask_1, 1],
            s=16, alpha=0.8, label="Sports", c='orange', marker='^'
        )
    
    plt.title("Titles in SAE Latent Space (2D)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2") 
    plt.legend()
    plt.tight_layout()
    plt.show()


def latents_labels_map(
    reducer: Reducer2D,
    Z_titles: np.ndarray, 
    medoids: np.ndarray,
    labels_text: List[str],
    is_sport: Optional[np.ndarray] = None,
    alpha_bg: float = 0.2,
    figsize: tuple = (10, 8)
) -> None:
    """
    Create 2D map showing medoid latents with text labels.
    
    Args:
        reducer: Fitted 2D reducer
        Z_titles: Title-level codes for background context [N_titles, K]
        medoids: Indices of medoid latents
        labels_text: Text labels for each medoid
        is_sport: Binary array indicating sports-related medoids
        alpha_bg: Alpha for background scatter
        figsize: Figure size
    """
    # Get 2D embedding for background titles
    if Z_titles.shape[0] > 0:
        X2_bg = reducer.transform(Z_titles)
        X2_bg = trim_outliers(X2_bg, p=95.0)
    else:
        X2_bg = np.empty((0, 2))
    
    # Create latent representation matrix (simplified - using identity for demo)
    # In practice, you'd use actual decoder weights or latent activations
    latent_vecs = np.eye(len(medoids))  # Placeholder - replace with actual latent vectors
    
    if latent_vecs.shape[1] == len(medoids):
        X2_latents = reducer.transform(latent_vecs.T)
    else:
        # If dimensions don't match, use random positions (fallback)
        X2_latents = np.random.randn(len(medoids), 2) * np.std(X2_bg) if len(X2_bg) > 0 else np.random.randn(len(medoids), 2)
    
    plt.figure(figsize=figsize)
    
    # Background scatter (titles)
    if len(X2_bg) > 0:
        plt.scatter(
            X2_bg[:, 0], X2_bg[:, 1], 
            s=8, alpha=alpha_bg, c='lightgray', 
            label='Background Titles'
        )
    
    # Medoid points with labels
    for i, (pos, label) in enumerate(zip(X2_latents, labels_text)):
        color = 'red' if is_sport is not None and i < len(is_sport) and is_sport[i] else 'blue'
        
        plt.scatter(pos[0], pos[1], s=100, c=color, alpha=0.8, 
                   marker='D', edgecolors='black', linewidth=1)
        
        # Add text label
        plt.annotate(
            label, (pos[0], pos[1]), 
            xytext=(5, 5), textcoords='offset points',
            fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    plt.title("Latent Medoids in 2D Space")
    plt.xlabel("Dimension 1") 
    plt.ylabel("Dimension 2")
    
    # Custom legend
    legend_elements = []
    if len(X2_bg) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='lightgray', markersize=8, 
                                        alpha=alpha_bg, label='Background Titles'))
    
    if is_sport is not None:
        if is_sport.any():
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                            markerfacecolor='red', markersize=10,
                                            markeredgecolor='black', label='Sports Latents'))
        if (~is_sport).any():
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', 
                                            markerfacecolor='blue', markersize=10,
                                            markeredgecolor='black', label='Other Latents'))
    else:
        legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                        markerfacecolor='blue', markersize=10, 
                                        markeredgecolor='black', label='Medoid Latents'))
    
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()


def activation_heatmap(
    activations: np.ndarray,
    token_labels: Optional[List[str]] = None,
    latent_labels: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    top_k: int = 50
) -> None:
    """
    Create heatmap of top activations.
    
    Args:
        activations: Activation matrix [N_tokens, K_latents] 
        token_labels: Labels for tokens (x-axis)
        latent_labels: Labels for latents (y-axis)
        figsize: Figure size
        top_k: Number of top activations to show
    """
    # Get top-k most active entries
    flat_idx = np.argpartition(activations.ravel(), -top_k)[-top_k:]
    row_idx, col_idx = np.unravel_index(flat_idx, activations.shape)
    
    # Sort by activation value
    sort_idx = np.argsort(-activations[row_idx, col_idx])
    row_idx = row_idx[sort_idx]
    col_idx = col_idx[sort_idx]
    
    # Create subset matrix
    subset_acts = activations[row_idx][:, col_idx]
    
    plt.figure(figsize=figsize)
    plt.imshow(subset_acts, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Activation')
    
    plt.title(f'Top-{top_k} Latent Activations')
    plt.xlabel('Latents')
    plt.ylabel('Tokens')
    
    # Set labels if provided
    if latent_labels:
        selected_latent_labels = [latent_labels[i] if i < len(latent_labels) 
                                 else f'L{i}' for i in col_idx]
        plt.xticks(range(len(selected_latent_labels)), selected_latent_labels, 
                  rotation=45, ha='right')
    
    if token_labels:
        selected_token_labels = [token_labels[i] if i < len(token_labels)
                                else f'T{i}' for i in row_idx]
        plt.yticks(range(len(selected_token_labels)), selected_token_labels)
    
    plt.tight_layout()
    plt.show()
