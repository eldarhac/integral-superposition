"""
Dataset loading and preprocessing utilities.
"""

import os
import pandas as pd
from typing import Tuple, Optional

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False


def load_titles_csv(
    path: str, 
    text_col: str = "title", 
    label_col: str = "topic"
) -> pd.DataFrame:
    """
    Load and preprocess a CSV dataset of titles and labels.
    
    Args:
        path: Path to CSV file
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Processed DataFrame with normalized labels
    """
    # Try different separators
    try:
        df = pd.read_csv(path, sep=";")
    except:
        df = pd.read_csv(path)
    
    # Validate columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col},{label_col} missing from CSV")
    
    # Select and clean data
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    
    # Balance dataset if needed (e.g., SPORTS vs NON-SPORTS)
    if df[label_col].dtype == object:
        # Assume binary classification with string labels
        unique_labels = df[label_col].unique()
        if len(unique_labels) == 2:
            # Balance by taking equal samples of each class
            label_counts = df[label_col].value_counts()
            min_count = label_counts.min()
            
            balanced_dfs = []
            for label in unique_labels:
                subset = df[df[label_col] == label].sample(
                    n=min_count, random_state=42
                )
                balanced_dfs.append(subset)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Convert labels to numeric (0, 1)
    if df[label_col].dtype == object:
        unique_labels = sorted(df[label_col].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        df[label_col] = df[label_col].map(label_map)
    
    df[label_col] = df[label_col].astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def split_df(
    df: pd.DataFrame, 
    train: float = 0.8, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/test sets.
    
    Args:
        df: DataFrame to split
        train: Fraction for training set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train)
    
    train_df = df_shuffled[:split_idx].reset_index(drop=True)
    test_df = df_shuffled[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def load_kaggle_dataset(
    dataset_id: str,
    filename: str,
    text_col: str = "title", 
    label_col: str = "topic"
) -> pd.DataFrame:
    """
    Load and preprocess a dataset from Kaggle.
    
    Args:
        dataset_id: Kaggle dataset identifier (e.g., "kotartemiy/topic-labeled-news-dataset")
        filename: Name of the CSV file within the dataset
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Processed DataFrame with normalized labels
        
    Raises:
        ImportError: If kagglehub is not installed
        FileNotFoundError: If the dataset or file is not found
    """
    if not HAS_KAGGLEHUB:
        raise ImportError(
            "kagglehub is required for Kaggle dataset loading. "
            "Install with: pip install kagglehub"
        )
    
    try:
        # Download dataset from Kaggle
        dataset_path = kagglehub.dataset_download(dataset_id)
        csv_path = os.path.join(dataset_path, filename)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {filename} not found in downloaded dataset at {dataset_path}")
        
        print(f"Downloaded Kaggle dataset to: {csv_path}")
        
        # Load and process using existing function
        return load_titles_csv(csv_path, text_col, label_col)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Kaggle dataset {dataset_id}: {str(e)}")


def load_dataset_auto(
    path_or_kaggle: str,
    filename: Optional[str] = None,
    text_col: str = "title", 
    label_col: str = "topic"
) -> pd.DataFrame:
    """
    Auto-detect and load dataset from local path or Kaggle.
    
    Args:
        path_or_kaggle: Either a local file path or Kaggle dataset ID
        filename: Required if path_or_kaggle is a Kaggle dataset ID
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Processed DataFrame with normalized labels
    """
    # If it's a local file path
    if os.path.exists(path_or_kaggle):
        return load_titles_csv(path_or_kaggle, text_col, label_col)
    
    # If filename is provided, assume it's a Kaggle dataset
    elif filename is not None:
        return load_kaggle_dataset(path_or_kaggle, filename, text_col, label_col)
    
    # Try as Kaggle dataset with common filename patterns
    elif "/" in path_or_kaggle and not path_or_kaggle.endswith('.csv'):
        common_filenames = [
            'labelled_newscatcher_dataset.csv',
            'dataset.csv', 
            'data.csv',
            'train.csv'
        ]
        
        for fname in common_filenames:
            try:
                return load_kaggle_dataset(path_or_kaggle, fname, text_col, label_col)
            except (FileNotFoundError, RuntimeError):
                continue
        
        raise FileNotFoundError(
            f"Could not find a suitable CSV file in Kaggle dataset {path_or_kaggle}. "
            f"Tried: {common_filenames}. Please specify filename explicitly."
        )
    
    else:
        raise ValueError(
            f"Invalid path_or_kaggle: {path_or_kaggle}. "
            "Provide either a local file path or Kaggle dataset ID with filename."
        )
