# GNN Directory Structure

This document describes the folder structure for the Graph Neural Network (GNN) pipeline.

## Directory Layout

```
gnn/
├── data/
│   ├── raw/              # Original event logs (CSV/XES) - input files
│   ├── processed/        # Cleaned and preprocessed CSV files
│   └── graphs/           # Generated PyTorch .pkl graph files
│
├── model/                # GNN model architectures
│   └── (HeteroGAT, SEPHIGRAPH, etc. will go here)
│
├── utils/                # Helper functions
│   └── (encoders, metrics, normalizers will go here)
│
├── experiments/          # Notebooks, configurations, and tests
│   └── (Jupyter notebooks, test scripts, config files)
│
└── outputs/
    ├── checkpoints/      # Saved model checkpoints (.pt files)
    ├── logs/             # Training logs and history (.json files)
    └── results/          # Evaluation results and visualizations
```

## Purpose of Each Directory

### `data/`
- **`raw/`**: Store original event log files (CSV/XES) before any processing
- **`processed/`**: Store cleaned and preprocessed event logs (after Step 0 trace reconstruction)
- **`graphs/`**: Store graph representations (.pkl files) created from event logs

### `model/`
- Contains GNN architecture implementations:
  - HeteroGAT (Heterogeneous Graph Attention Network)
  - SEPHIGRAPH
  - Other GNN models as needed

### `utils/`
- Helper functions and utilities:
  - Encoders (one-hot, embeddings)
  - Metrics (accuracy, F1, MSE, etc.)
  - Normalizers (MinMax, Standard)
  - Graph utilities

### `experiments/`
- Experimental code and analysis:
  - Jupyter notebooks for exploration
  - Configuration files (YAML/JSON)
  - Unit tests and integration tests

### `outputs/`
- **`checkpoints/`**: Model weights saved during training
- **`logs/`**: Training history, metrics, and logs
- **`results/`**: Final evaluation results, visualizations, and analysis

## Workflow

1. **Input**: Place event logs in `data/raw/`
2. **Preprocessing**: Process logs → save to `data/processed/`
3. **Graph Creation**: Create graphs → save to `data/graphs/`
4. **Training**: Train models → save checkpoints to `outputs/checkpoints/`
5. **Evaluation**: Evaluate models → save results to `outputs/results/`

