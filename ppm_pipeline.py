# ppm_pipeline.py
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

np.random.seed(42)

# Strategy Pattern entry point: model+task -> predictor wrapper.
# Importing this does not pull in TF/Torch/best4ppm (wrappers import lazily).
from models.registry import get_predictor

# Declarative capability manifest (single source of truth shared with the API's
# GET /capabilities and the frontend wizard). Default configs are derived from it
# so hyperparameter defaults never drift between the pipeline and the UI.
from models.capabilities import default_config as _default_config

# Explainability modules are imported lazily inside each wrapper's explain();
# the wrappers raise a clear RuntimeError if a requested method is unavailable.

BEST_AVAILABLE = True
BEST_IMPORT_ERROR = None
try:
    from best.predictor import BESTRunner
except ImportError as e:
    BEST_AVAILABLE = False
    BEST_IMPORT_ERROR = str(e)

# We already verified TF/Torch are installed, but keep these guards for robustness
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    torch.manual_seed(42)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Concrete predictor classes are imported lazily by their wrappers in models/;
# the TENSORFLOW_AVAILABLE / PYTORCH_AVAILABLE flags above still gate the shims.


def detect_and_standardize_columns(df, verbose=False):
    column_mapping = {}

    case_patterns = ['case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    activity_patterns = ['concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
    timestamp_patterns = ['time:timestamp', 'Timestamp', 'timestamp', 'time', 'Time', 'start_time', 'StartTime', 'complete_time', 'CompleteTime', 'Complete Timestamp']
    resource_patterns = ['org:resource', 'Resource', 'resource', 'user', 'User', 'org:role', 'role', 'Role', 'actor', 'Actor']

    for col in df.columns:
        if col in case_patterns and col != 'CaseID':
            column_mapping[col] = 'CaseID'
            break

    # Activity - only map if 'Activity' doesn't already exist
    if 'Activity' not in df.columns:
        for col in df.columns:
            if col in activity_patterns:
                column_mapping[col] = 'Activity'
                if verbose:
                    print(f"[COLUMN DETECT] Mapping '{col}' → 'Activity'")
                break
    else:
        if verbose:
            print(f"[COLUMN DETECT] Using existing 'Activity' column")

    # Timestamp - only map if 'Timestamp' doesn't already exist
    if 'Timestamp' not in df.columns:
        for col in df.columns:
            if col in timestamp_patterns:
                column_mapping[col] = 'Timestamp'
                break

    # Resource - only map if 'Resource' doesn't already exist  
    if 'Resource' not in df.columns:
        for col in df.columns:
            if col in resource_patterns:
                column_mapping[col] = 'Resource'
                break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ['CaseID', 'Activity', 'Timestamp']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")

    return df, column_mapping, column_mapping.keys()


def _safe_rename_columns(df, rename_map):
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    for src, tgt in rename_map.items():
        if tgt in df.columns and tgt != src and tgt not in rename_map.keys():
            df = df.drop(columns=[tgt])
    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def default_transformer_config():
    return _default_config("transformer")


def default_gnn_config():
    return _default_config("gnn")


def default_best_config():
    return _default_config("best")


def run_next_activity_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    target_column=None,
    skip_auto_mapping=False,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    
    # DEBUG 1: Raw dataset
    if 'Activity' in df.columns:
        print("\n[DEBUG 1] RAW CSV:", df['Activity'].nunique(), "unique activities")
        print("Sample:", list(df['Activity'].unique()[:5]))
    else:
        print("\n[DEBUG 1] RAW CSV: Activity column not found")
        print("Columns:", list(df.columns))
    
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    target_series = None
    if target_column:
        if target_column not in df.columns:
            raise RuntimeError(f"Target column not found: {target_column}")
        if pd.api.types.is_numeric_dtype(df[target_column]):
            raise RuntimeError("Invalid target column selected: must be categorical.")
        target_series = df[target_column].astype(str)
    
    # DEBUG 2: After standardization
    if 'Activity' in df.columns:
        print("[DEBUG 2] AFTER STANDARDIZE:", df['Activity'].nunique(), "activities")
        print("Sample:", list(df['Activity'].unique()[:5]))
    else:
        print("[DEBUG 2] ERROR: Activity column missing after standardization")
        print("Columns:", list(df.columns))

    df = _safe_rename_columns(df, {
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    if target_series is not None:
        df['concept:name'] = target_series

    model_key = "transformer:custom_activity" if target_column else "transformer:next_activity"
    cfg = {
        **config,
        "test_size": test_size,
        "val_split": val_split,
        "output_dir": output_dir,
        "explainability_method": explainability_method,
    }
    predictor = get_predictor(model_key, cfg)
    data = predictor.prepare_data(df)
    predictor.train(data)
    metrics = predictor.evaluate(data)
    predictor.explain(data)

    return metrics


def run_event_time_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    skip_auto_mapping=False,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    cfg = {
        **config,
        "test_size": test_size,
        "val_split": val_split,
        "output_dir": output_dir,
        "explainability_method": explainability_method,
    }
    predictor = get_predictor("transformer:event_time", cfg)
    data = predictor.prepare_data(df)
    predictor.train(data)
    metrics = predictor.evaluate(data)
    predictor.explain(data)

    return metrics


def run_remaining_time_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    skip_auto_mapping=False,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    cfg = {
        **config,
        "test_size": test_size,
        "val_split": val_split,
        "output_dir": output_dir,
        "explainability_method": explainability_method,
    }
    predictor = get_predictor("transformer:remaining_time", cfg)
    data = predictor.prepare_data(df)
    predictor.train(data)
    metrics = predictor.evaluate(data)
    predictor.explain(data)

    return metrics


def run_outcome_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    skip_auto_mapping=False,
):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = _safe_rename_columns(df, {
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    cfg = {
        **config,
        "test_size": test_size,
        "val_split": val_split,
        "output_dir": output_dir,
        "explainability_method": explainability_method,
    }
    predictor = get_predictor("transformer:outcome", cfg)
    data = predictor.prepare_data(df)
    predictor.train(data)
    metrics = predictor.evaluate(data)
    predictor.explain(data)

    return metrics


def run_gnn_unified_prediction(
    dataset_path,
    output_dir,
    test_size,
    val_split,
    config,
    explainability_method=None,
    task='unified',
    target_column=None,
    skip_auto_mapping=False,
):
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. GNN runs cannot execute.")

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)
    if target_column:
        if target_column not in df.columns:
            raise RuntimeError(f"Target column not found: {target_column}")
        if pd.api.types.is_numeric_dtype(df[target_column]):
            raise RuntimeError("Invalid target column selected: must be categorical.")
        df["Activity"] = df[target_column].astype(str)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)

    cfg = {
        **config,
        "test_size": test_size,
        "val_split": val_split,
        "output_dir": output_dir,
        "explainability_method": explainability_method,
    }
    predictor = get_predictor(f"gnn:{task}", cfg)
    data = predictor.prepare_data(df)
    predictor.train(data)
    metrics = predictor.evaluate(data)
    predictor.explain(data)

    return metrics

def run_best_nap_prediction(
    dataset_path,
    output_dir,
    config,
    split,
    explainability=None,
    skip_auto_mapping=False,
):
    if not BEST_AVAILABLE:
        raise RuntimeError(
            f"best4ppm not available. Install with: pip install git+https://github.com/lmu-dbs/BEST.git"
            f" ({BEST_IMPORT_ERROR})"
        )

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    test_size = float(split.get("test_size", 0.2))

    cfg = {
        **config,
        "test_size": test_size,
        "output_dir": output_dir,
        "explainability_method": explainability,
    }
    predictor = get_predictor("best:next_activity", cfg)
    print("[BEST NAP] Preparing data...")
    data = predictor.prepare_data(df)
    print("[BEST NAP] Fitting model...")
    predictor.train(data)
    print("[BEST NAP] Evaluating...")
    metrics = predictor.evaluate(data)
    print(f"[BEST NAP] Metrics: {metrics}")
    if explainability:
        print("[BEST NAP] Running explainability...")
    predictor.explain(data)

    return metrics

def run_best_rtp_prediction(
    dataset_path,
    output_dir,
    config,
    split,
    explainability=None,
    skip_auto_mapping=False,
):
    if not BEST_AVAILABLE:
        raise RuntimeError(
            f"best4ppm not available. Install with: pip install git+https://github.com/lmu-dbs/BEST.git"
            f" ({BEST_IMPORT_ERROR})"
        )

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    test_size = float(split.get("test_size", 0.2))

    cfg = {
        **config,
        "test_size": test_size,
        "output_dir": output_dir,
        "explainability_method": explainability,
    }
    predictor = get_predictor("best:remaining_trace", cfg)
    print("[BEST RTP] Preparing data...")
    data = predictor.prepare_data(df)
    print("[BEST RTP] Fitting model...")
    predictor.train(data)
    print("[BEST RTP] Evaluating...")
    metrics = predictor.evaluate(data)
    print(f"[BEST RTP] Metrics: {metrics}")
    if explainability:
        print("[BEST RTP] Running explainability...")
    predictor.explain(data)

    return metrics

def run_best_outcome_prediction(
    dataset_path,
    output_dir,
    config,
    split,
    explainability=None,
    skip_auto_mapping=False,
):
    if not BEST_AVAILABLE:
        raise RuntimeError(
            f"best4ppm not available. Install with: pip install git+https://github.com/lmu-dbs/BEST.git"
            f" ({BEST_IMPORT_ERROR})"
        )

    df = pd.read_csv(dataset_path)
    if skip_auto_mapping:
        missing = [c for c in ["CaseID", "Activity", "Timestamp"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (manual mapping): {missing}")
    else:
        df, _, _ = detect_and_standardize_columns(df, verbose=False)

    test_size = float(split.get("test_size", 0.2))

    cfg = {
        **config,
        "test_size": test_size,
        "output_dir": output_dir,
        "explainability_method": explainability,
    }
    predictor = get_predictor("best:outcome", cfg)
    print("[BEST OUTCOME] Preparing data...")
    data = predictor.prepare_data(df)
    print("[BEST OUTCOME] Fitting model...")
    predictor.train(data)
    print("[BEST OUTCOME] Evaluating...")
    metrics = predictor.evaluate(data)
    print(f"[BEST OUTCOME] Metrics: {metrics}")
    if explainability:
        print("[BEST OUTCOME] Running explainability...")
    predictor.explain(data)

    return metrics
