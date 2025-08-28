# Integral Superposition

A model-agnostic toolkit for Sparse Autoencoder training, analysis, and steering in mechanistic interpretability research.

## Features

- **Model-agnostic backends**: Support for HuggingFace Transformers and extensible to other architectures
- **Efficient activation dumping**: Stream activations to disk with batched processing
- **Sparse Autoencoder training**: Configurable SAE with L1 sparsity regularization
- **Rich analysis tools**: Enrichment analysis, top token extraction, clustering, and similarity metrics
- **Latent merging**: Cluster similar latents and refit decoders for dimensionality reduction
- **Steering capabilities**: Model intervention via forward hooks
- **Visualization**: 2D maps, activation panels, and heatmaps

## Installation

```bash
# Clone repository
git clone https://github.com/eldarhac/integral-superposition.git
cd integral-superposition

# Install in development mode
pip install -e .

# Optional: Install additional dependencies
pip install -e ".[viz]"      # For UMAP visualization
pip install -e ".[kaggle]"   # For Kaggle dataset loading
pip install -e ".[full]"     # Install all optional dependencies
```

## Quick Start

### 1. Dump Activations

```python
from integral_superposition import backends, data, dump, DumpConfig

# Load model
backend = backends.HFCausalLM.from_pretrained("google/gemma-3-270m")

# Load dataset - Option 1: From Kaggle (matches original notebook)
df = data.load_kaggle_dataset(
    "kotartemiy/topic-labeled-news-dataset", 
    "labelled_newscatcher_dataset.csv",
    text_col="title", 
    label_col="topic"
)

# Load dataset - Option 2: From local file
# df = data.load_titles_csv("dataset.csv", text_col="title", label_col="topic")

# Load dataset - Option 3: Auto-detect
# df = data.load_dataset_auto("kotartemiy/topic-labeled-news-dataset")

# Configure and dump
cfg = DumpConfig(layer=13, max_len=32, batch_size=64)
dump.dump_layer_activations(df, "title", "topic", backend, cfg, "acts_shards")
```

### 2. Train SAE

```python
from integral_superposition import sae, SAEConfig

cfg = SAEConfig(k=2048, lr=3e-4, l1=1e-3, epochs=2)
model = sae.SparseAutoencoder(d_model=640, k=2048)

# Train on dumped activations
sae.train_sae(model, shard_iterator, cfg, device)
sae.save_sae(model, "sae_weights.pt")
```

### 3. Analyze Latents

```python  
from integral_superposition import analysis

# Find top-activating tokens
top_sets = analysis.top_tokens.topk_sets_per_latent(model, act_files, device)

# Compute enrichment statistics
summary = analysis.enrichment.fisher_enrichment(top_sets, labels_global)
```

### 4. Merge Similar Latents

```python
from integral_superposition import merge

# Cluster by decoder similarity
dist = merge.cluster.cosine_dist_from_decoder(model.dec.weight.data.numpy())
labels = merge.cluster.agglomerative_labels(dist, n_clusters=256)
medoids = merge.medoid.medoids_from_labels(dist, labels)

# Refit decoder
M = merge.medoid.merge_matrix_from_labels(labels, medoids)
W_new = merge.refit.merge_and_refit_decoder(data_iter, model, M)
```

### 5. Steer Generation

```python
from integral_superposition import steering

# Create latent modifier
modifier = steering.set_or_scale_latent(latent_j=42, set_to=100.0)

# Register steering hook  
handle = steering.register_prehook_last_row(backend, layer_idx=13, sae=model, latent_map=modifier)

# Generate with intervention
outputs = backend.generate(**inputs, max_new_tokens=10)
handle.remove()
```

## Examples

Complete examples are provided in the `examples/` directory:

- `01_dump_activations.py` - Extract activations from language model
- `02_train_sae.py` - Train Sparse Autoencoder
- `03_analyze_latents.py` - Compute enrichment and statistics  
- `04_merge_and_refit.py` - Cluster and merge similar latents
- `05_plot_maps.py` - Create 2D visualizations
- `06_steer_generation.py` - Intervention experiments

Run examples:

```bash
cd examples
python 01_dump_activations.py
python 02_train_sae.py
python 03_analyze_latents.py
# ... etc
```

## Data Loading Options

The package supports multiple data sources:

### Kaggle Datasets (matches original notebook)
```python
from integral_superposition import data

# Direct Kaggle loading
df = data.load_kaggle_dataset(
    dataset_id="kotartemiy/topic-labeled-news-dataset",
    filename="labelled_newscatcher_dataset.csv",
    text_col="title", 
    label_col="topic"
)
```

### Local Files
```python
# CSV files with custom separators
df = data.load_titles_csv("path/to/dataset.csv", text_col="title", label_col="topic")

# Train/test split
train_df, test_df = data.split_df(df, train=0.8, seed=42)
```

### Auto-Detection
```python
# Automatically detect local file or Kaggle dataset
df = data.load_dataset_auto("kotartemiy/topic-labeled-news-dataset")
df = data.load_dataset_auto("/path/to/local/file.csv")
```

## Package Structure

```
integral_superposition/
├── backends/          # Model abstraction layer
│   ├── base_model.py     # Abstract backend interface
│   ├── hf_causal_lm.py   # HuggingFace implementation
│   └── io_store.py       # File I/O utilities
├── data/              # Dataset handling
│   ├── datasets.py       # CSV loading and preprocessing  
│   └── tokenize.py       # Tokenization utilities
├── dump/              # Activation extraction
│   └── activations.py    # Batched dumping to shards
├── sae/               # Sparse Autoencoder
│   ├── model.py          # SAE model definition
│   ├── train.py          # Training loop
│   └── load.py           # Inference utilities
├── analysis/          # Latent analysis
│   ├── enrichment.py     # Fisher exact enrichment
│   ├── top_tokens.py     # Top-k extraction
│   ├── jaccard_sparse.py # Sparse Jaccard similarity
│   ├── reducer.py        # Dimensionality reduction
│   └── stats.py          # Statistical utilities
├── merge/             # Latent clustering/merging
│   ├── cluster.py        # Agglomerative clustering
│   ├── medoid.py         # Medoid selection
│   └── refit.py          # Decoder refitting
├── steering/          # Model interventions
│   └── hooks.py          # Forward hook utilities
└── viz/               # Visualization
    ├── panels.py         # Interactive HTML panels
    └── maps.py           # 2D scatter plots
```

## Configuration

Use Pydantic models for type-safe configuration:

```python
from integral_superposition import DumpConfig, SAEConfig, Paths

dump_cfg = DumpConfig(layer=13, max_len=32, batch_size=64, dtype="bf16")
sae_cfg = SAEConfig(k=2048, lr=3e-4, l1=1e-3, epochs=2, batch_size=4096)  
paths = Paths(shards_dir="acts", sae_path="model.pt", summary_csv="results.csv")
```

## Requirements

- Python ≥3.8
- PyTorch ≥1.13
- Transformers ≥4.20
- NumPy, Pandas, Scikit-learn
- Optional: UMAP (for visualization)

## Contributing

Contributions welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License. See `LICENSE` for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{integral_superposition,
  title={Integral Superposition: Model-Agnostic SAE Toolkit},
  author={Research Team},
  year={2024},
  url={https://github.com/example/integral-superposition}
}
```
