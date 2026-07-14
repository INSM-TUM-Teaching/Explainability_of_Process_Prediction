"""Stateless helpers shared across the GNN prediction modules.

These are deliberately model-agnostic so that future neural-network predictors
added under ``gnns/prediction`` can reuse them without depending on
``GNNPredictor``.
"""
import os
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
    """Pick the fastest available device: CUDA > Apple Metal (MPS) > CPU.

    On Apple Silicon we enable ``PYTORCH_ENABLE_MPS_FALLBACK`` so any op not yet
    implemented for Metal transparently runs on CPU instead of raising. That
    keeps GPU training working across PyTorch versions and machines rather than
    crashing on an unsupported kernel.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")
