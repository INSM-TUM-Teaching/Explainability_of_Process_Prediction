# models/registry.py
"""Model Registry + factory (Strategy Pattern entry point).

Callers select a strategy by a ``"<model>:<task>"`` string instead of branching
on model/task. Importing this module does not import TensorFlow/PyTorch/best4ppm
because every wrapper imports its heavy concrete predictor lazily.
"""
from .base_predictor import BasePredictor
from .transformer_predictor import (
    TransformerNextActivityWrapper,
    TransformerEventTimeWrapper,
    TransformerRemainingTimeWrapper,
    TransformerOutcomeWrapper,
)
from .gnn_predictor import GNNWrapper
from .best_predictor import BESTWrapper


MODEL_REGISTRY = {
    # transformer (one concrete predictor class per task)
    "transformer:next_activity": TransformerNextActivityWrapper,
    "transformer:custom_activity": TransformerNextActivityWrapper,
    "transformer:event_time": TransformerEventTimeWrapper,
    "transformer:remaining_time": TransformerRemainingTimeWrapper,
    "transformer:outcome": TransformerOutcomeWrapper,
    # gnn (one wrapper, task selects loss weights / explain heads)
    "gnn:next_activity": GNNWrapper,
    "gnn:custom_activity": GNNWrapper,
    "gnn:event_time": GNNWrapper,
    "gnn:remaining_time": GNNWrapper,
    "gnn:unified": GNNWrapper,
    "gnn:outcome": GNNWrapper,
    # best (one wrapper, task maps to nap/rtp/outcome)
    "best:next_activity": BESTWrapper,
    "best:remaining_trace": BESTWrapper,
    "best:outcome": BESTWrapper,
}


def get_predictor(model_name: str, config) -> BasePredictor:
    """Return an initialized predictor strategy for ``model_name``.

    Args:
        model_name: composite key ``"<model>:<task>"`` (e.g. ``"transformer:outcome"``).
        config: dict of hyperparameters and runtime parameters. ``model_type`` and
            ``task`` are injected from ``model_name`` if not already present.

    Raises:
        ValueError: if ``model_name`` is not registered.
    """
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {sorted(MODEL_REGISTRY)}"
        )

    # Copy so injecting model_type/task never mutates the caller's dict.
    config = dict(config)
    model_type, _, task = model_name.partition(":")
    config.setdefault("model_type", model_type)
    if task:
        config.setdefault("task", task)

    return cls(config)
