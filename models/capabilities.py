# models/capabilities.py
"""Declarative capability manifest for every registered model (single source of truth).

This module is intentionally dependency-free: it imports no TensorFlow / PyTorch /
best4ppm, so both the FastAPI process (``GET /capabilities``) and the training
pipeline can import it cheaply. It describes, per model, everything the frontend
wizard needs to render itself without hardcoding model knowledge:

* which prediction ``tasks`` the model supports (and how to group / label them),
* the ``config_fields`` schema (labels, defaults, and declarative validation rules),
* any cross-field ``config_constraints`` (e.g. BEST eval pattern size <= train),
* the ``explain_methods`` available for the model.

The task ids here must stay in sync with the ``"<model>:<task>"`` keys in
``models.registry.MODEL_REGISTRY``. Config field keys/defaults must match the
concrete predictors' expected config keys (see the ``default_*_config`` helpers
in ``ppm_pipeline.py``, which now derive their values from this manifest).
"""
from typing import Any, Dict, List


# --- reusable validation rule fragments -------------------------------------
_POS_INT = {"kind": "number", "integer": True, "gt": 0, "step": 1}
_DROPOUT = {"kind": "number", "gt": 0, "lt": 1, "step": 0.05, "min": 0, "max": 1}


MODEL_CAPABILITIES: List[Dict[str, Any]] = [
    {
        "id": "transformer",
        "label": "Transformer",
        "description": (
            "State-of-the-art attention-based architecture, ideal for sequential data and "
            "complex patterns. Best for large datasets with temporal dependencies."
        ),
        "config_intro": "These options map 1:1 to the backend Transformer config.",
        "tasks": [
            {
                "id": "next_activity",
                "label": "Next Activity Prediction",
                "description": "Predict the next activity in a running process instance.",
                "category": "classification",
            },
            {
                "id": "custom_activity",
                "label": "Custom Prediction",
                "description": "Predict the next value of a selected categorical column.",
                "category": "classification",
                "needs_target_column": True,
            },
            {
                "id": "outcome",
                "label": "Outcome Prediction",
                "description": (
                    "Predict the final outcome of a running process instance based on its "
                    "activity prefix. Each case's last activity is used as its outcome label."
                ),
                "category": "classification",
            },
            {
                "id": "event_time",
                "label": "Event Time Prediction",
                "description": "Predict the time until the next event occurs.",
                "category": "regression",
            },
            {
                "id": "remaining_time",
                "label": "Remaining Time Prediction",
                "description": "Estimate the time required to complete the process.",
                "category": "regression",
            },
        ],
        "config_fields": [
            {"key": "max_len", "label": "Max sequence length", "default": 16, "placeholder": "16", **_POS_INT},
            {"key": "d_model", "label": "Model dimension", "default": 64, "placeholder": "64", **_POS_INT},
            {"key": "num_heads", "label": "Number of attention heads", "default": 4, "placeholder": "4", **_POS_INT},
            {"key": "num_blocks", "label": "Number of transformer blocks", "default": 2, "placeholder": "2", **_POS_INT},
            {"key": "dropout_rate", "label": "Dropout rate", "default": 0.1, "placeholder": "0.1", **_DROPOUT},
            {"key": "epochs", "label": "Number of epochs", "default": 5, "placeholder": "5", **_POS_INT},
            {"key": "batch_size", "label": "Batch size", "default": 128, "placeholder": "128", **_POS_INT},
            {"key": "patience", "label": "Early stopping patience", "default": 10, "placeholder": "10", **_POS_INT},
        ],
        "config_constraints": [],
        "explain_methods": [
            {"value": "none", "label": "None", "description": "Skip explainability to run faster."},
            {
                "value": "lime",
                "label": "LIME",
                "description": (
                    "Local surrogate explanations. Explains individual predictions by "
                    "approximating the model locally with an interpretable model."
                ),
            },
            {
                "value": "shap",
                "label": "SHAP",
                "description": (
                    "Shapley-value based feature attributions. Provides consistent local "
                    "explanations across features."
                ),
            },
            {"value": "all", "label": "Both (LIME + SHAP)", "description": "Run both methods (takes longer)."},
        ],
    },
    {
        "id": "gnn",
        "label": "GNN (Graph Neural Network)",
        "description": (
            "Graph-based architecture, perfect for modeling relationships and dependencies "
            "between activities. Excels at capturing process structure."
        ),
        "config_intro": "These options map 1:1 to the backend GNN config.",
        "tasks": [
            {
                "id": "next_activity",
                "label": "Next Activity Prediction",
                "description": "Predict the next activity in a running process instance.",
                "category": "classification",
            },
            {
                "id": "custom_activity",
                "label": "Custom Prediction",
                "description": "Predict the next value of a selected categorical column.",
                "category": "classification",
                "needs_target_column": True,
            },
            {
                "id": "outcome",
                "label": "Outcome Prediction",
                "description": (
                    "Predict the final outcome of a running process instance based on its "
                    "activity prefix. Each case's last activity is used as its outcome label."
                ),
                "category": "classification",
            },
            {
                "id": "event_time",
                "label": "Event Time Prediction",
                "description": "Predict the time until the next event occurs.",
                "category": "regression",
            },
            {
                "id": "remaining_time",
                "label": "Remaining Time Prediction",
                "description": "Estimate the time required to complete the process.",
                "category": "regression",
            },
        ],
        "config_fields": [
            {"key": "hidden", "label": "Hidden channels", "default": 64, "placeholder": "64", **_POS_INT},
            {"key": "dropout_rate", "label": "Dropout rate", "default": 0.1, "placeholder": "0.1", **_DROPOUT},
            {
                "key": "lr",
                "label": "Learning rate",
                "default": 4e-4,
                "placeholder": "0.0004",
                "kind": "number",
                "gt": 0,
                "min": 0,
                "step": 0.0001,
            },
            {"key": "epochs", "label": "Number of epochs", "default": 5, "placeholder": "5", **_POS_INT},
            {"key": "batch_size", "label": "Batch size", "default": 64, "placeholder": "64", **_POS_INT},
            {"key": "patience", "label": "Early stopping patience", "default": 10, "placeholder": "10", **_POS_INT},
        ],
        "config_constraints": [],
        "explain_methods": [
            {"value": "none", "label": "None", "description": "Skip explainability to run faster."},
            {
                "value": "gradient",
                "label": "Gradient-Based",
                "description": (
                    "Uses gradients to estimate which input features influence predictions "
                    "most strongly."
                ),
            },
            {
                "value": "lime",
                "label": "GraphLIME",
                "description": (
                    "Graph-specific local explanations. Identifies important "
                    "substructures/features for a prediction."
                ),
            },
            {"value": "all", "label": "Both (Gradient + GraphLIME)", "description": "Run both methods (takes longer)."},
        ],
    },
    {
        "id": "best",
        "label": "BEST (Bilaterally Expanding Subtrace Tree)",
        "description": (
            "Probabilistic tree-based model for activity sequence prediction. Builds a "
            "pattern tree directly from the event log without neural network training. "
            "Supports next activity and remaining trace prediction."
        ),
        "config_intro": (
            "BEST builds a bilaterally expanding subtrace tree from your event log (no neural "
            "training). Parameters follow the official BEST framework; defaults match the "
            "BPM 2025 paper setup."
        ),
        "tasks": [
            {
                "id": "next_activity",
                "label": "Next Activity Prediction (NAP)",
                "description": (
                    "Predict the single next activity for each running case prefix. Evaluated "
                    "with accuracy and balanced accuracy (BEST task: nap)."
                ),
                "category": "sequence",
            },
            {
                "id": "remaining_trace",
                "label": "Remaining Trace Prediction (RTP)",
                "description": (
                    "Predict the full sequence of activities until case completion. Uses the "
                    "break buffer setting from model configuration to cap trace length. "
                    "Evaluated with normalized DLS (BEST task: rtp)."
                ),
                "category": "sequence",
            },
            {
                "id": "outcome",
                "label": "Outcome Prediction",
                "description": (
                    "Predict the final outcome of a running process instance. Evaluated with "
                    "accuracy and none-share (BEST task: outcome)."
                ),
                "category": "sequence",
            },
        ],
        "config_fields": [
            {
                "key": "max_pattern_size_train",
                "label": "Training pattern size (odd)",
                "description": (
                    "Maximum subtrace pattern length while building the tree. Tree depth is "
                    "(size - 1) / 2: size 21 means up to 10 activities before and after the "
                    "center activity. Larger values capture more context but increase training "
                    "time and memory."
                ),
                "default": 21,
                "placeholder": "21",
                "kind": "number",
                "integer": True,
                "odd": True,
                "min": 3,
                "step": 2,
            },
            {
                "key": "max_pattern_size_eval",
                "label": "Evaluation pattern size (odd, ≤ training)",
                "description": (
                    "Maximum pattern length when matching test prefixes to the tree during "
                    "prediction. Can be smaller than training to limit how far the matcher "
                    "walks; must not exceed the training size."
                ),
                "default": 21,
                "placeholder": "21",
                "kind": "number",
                "integer": True,
                "odd": True,
                "min": 3,
                "step": 2,
            },
            {
                "key": "process_stage_width_percentage",
                "label": "Process stage width (0 to 1)",
                "description": (
                    "Controls how many process stages BEST uses. Stage width is this fraction "
                    "of the longest training trace. 0 splits into many narrow stages (one BEST "
                    "model per event position); 1 uses a single stage for the whole trace. "
                    "Paper experiments often use values between 0 and 1."
                ),
                "default": 0.2,
                "placeholder": "0.2",
                "kind": "number",
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            {
                "key": "min_freq",
                "label": "Minimum pattern frequency",
                "description": (
                    "Drop subtrace patterns whose frequency in the log is below this cutoff. "
                    "Values near zero (e.g. 1e-14) keep almost all patterns; higher values "
                    "prune rare patterns and speed up runs."
                ),
                "default": 1e-14,
                "placeholder": "1e-14",
                "kind": "number",
                "gt": 0,
                "min": 0,
                "step": "any",
            },
            {
                "key": "break_buffer",
                "label": "Remaining-trace stop factor (> 1)",
                "description": (
                    "Used only for remaining trace prediction (RTP). Stops extending the "
                    "predicted sequence when its length reaches break_buffer × longest "
                    "training trace length. 1.2 is the value used in the BEST paper."
                ),
                "default": 1.2,
                "placeholder": "1.2",
                "kind": "number",
                "gt": 1,
                "min": 1,
                "step": 0.1,
            },
            {
                "key": "filter_sequences",
                "label": "Filter padding tokens for evaluation",
                "description": (
                    "When enabled, removes padded dummy activities from predicted sequences "
                    "before scoring RTP/NAP metrics so evaluation matches the official BEST "
                    "benchmark."
                ),
                "default": True,
                "kind": "boolean",
            },
            {
                "key": "ncores",
                "label": "Parallel cores",
                "description": (
                    "Number of CPU cores for parallel prediction and RTP evaluation (NDLS). "
                    "Use 1 on small machines; increase on multi-core hosts to shorten "
                    "remaining-trace runs."
                ),
                "default": 1,
                "placeholder": "1",
                "kind": "number",
                "integer": True,
                "min": 1,
                "step": 1,
            },
        ],
        "config_constraints": [
            {
                "type": "lte",
                "left": "max_pattern_size_eval",
                "right": "max_pattern_size_train",
                "message": "Evaluation pattern size must not exceed the training pattern size.",
            }
        ],
        "explain_methods": [
            {"value": "none", "label": "None", "description": "Skip explainability to run faster."},
            {
                "value": "pattern_analysis",
                "label": "Pattern Analysis",
                "description": (
                    "Summarises which historical subtrace patterns BEST matched during "
                    "prediction: top patterns by frequency (with accuracy), activity "
                    "importance, high-error patterns, RPIF distance and confidence "
                    "distributions, plus CSV/JSON tables for the results view."
                ),
            },
        ],
    },
]


# Index by id for O(1) lookups.
_BY_ID: Dict[str, Dict[str, Any]] = {m["id"]: m for m in MODEL_CAPABILITIES}


def list_capabilities() -> List[Dict[str, Any]]:
    """Return the full manifest (list of model capability dicts)."""
    return MODEL_CAPABILITIES


def get_capability(model_type: str) -> Dict[str, Any]:
    """Return the capability dict for ``model_type`` or raise ``KeyError``."""
    return _BY_ID[model_type]


def default_config(model_type: str) -> Dict[str, Any]:
    """Return ``{field_key: default}`` for ``model_type`` from its config fields."""
    cap = _BY_ID[model_type]
    return {f["key"]: f["default"] for f in cap["config_fields"]}
