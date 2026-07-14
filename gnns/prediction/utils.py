"""Stateless helpers shared across the GNN prediction modules.

These are deliberately model-agnostic so that future neural-network predictors
added under ``gnns/prediction`` can reuse them without depending on
``GNNPredictor``.
"""
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def signed_log1p(value):
    v = float(value)
    if not np.isfinite(v):
        raise ValueError("Non-finite trace value")
    return np.sign(v) * np.log1p(abs(v))


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
