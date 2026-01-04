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

EXPLAINABILITY_AVAILABLE = True
try:
    from explainability.transformers import run_transformer_explainability
    from explainability.gnns import run_gnn_explainability
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

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

if TENSORFLOW_AVAILABLE:
    from transformers.prediction.next_activity import NextActivityPredictor
    from transformers.prediction.event_time import EventTimePredictor
    from transformers.prediction.remaining_time import RemainingTimePredictor

if PYTORCH_AVAILABLE:
    from gnns.prediction.gnn_predictor import GNNPredictor


def detect_and_standardize_columns(df, verbose=False):
    column_mapping = {}
    columns_to_drop = []

    case_patterns = ['case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    activity_patterns = ['concept:name', 'Action', 'activity', 'event', 'Event', 'task', 'Task']
    timestamp_patterns = ['time:timestamp', 'Timestamp', 'timestamp', 'time', 'Time', 'start_time', 'StartTime', 'complete_time', 'CompleteTime']
    resource_patterns = ['org:resource', 'Resource', 'resource', 'user', 'User', 'org:role', 'role', 'Role', 'actor', 'Actor']

    for col in df.columns:
        if col in case_patterns and col != 'CaseID':
            column_mapping[col] = 'CaseID'
            break

    for col in df.columns:
        if col in activity_patterns and col != 'Activity':
            if 'Activity' in df.columns and col != 'Activity':
                columns_to_drop.append('Activity')
            column_mapping[col] = 'Activity'
            break

    for col in df.columns:
        if col in timestamp_patterns and col != 'Timestamp':
            column_mapping[col] = 'Timestamp'
            break

    for col in df.columns:
        if col in resource_patterns and col != 'Resource':
            column_mapping[col] = 'Resource'
            break

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ['CaseID', 'Activity', 'Timestamp']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")

    return df, column_mapping, column_mapping.keys()


def default_transformer_config():
    return {
        'max_len': 16,
        'd_model': 64,
        'num_heads': 4,
        'num_blocks': 2,
        'dropout_rate': 0.1,
        'epochs': 5,
        'batch_size': 128,
        'patience': 10
    }


def default_gnn_config():
    return {
        'hidden': 64,
        'dropout_rate': 0.1,
        'lr': 4e-4,
        'epochs': 5,
        'batch_size': 64,
        'patience': 10
    }


def run_next_activity_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method=None):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = df.rename(columns={
        'CaseID': 'case:id',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    predictor = NextActivityPredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    y_pred, y_pred_probs = predictor.predict(data)
    predictor.save_results(data, y_pred, y_pred_probs, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='activity',
            num_samples=20,
            methods=explainability_method
        )

    return metrics


def run_event_time_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method=None):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    predictor = EventTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    y_pred = predictor.predict(data)
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=20,
            methods=explainability_method
        )

    return metrics


def run_remaining_time_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method=None):
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Transformer runs cannot execute.")

    df = pd.read_csv(dataset_path)
    df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })

    predictor = RemainingTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )

    metrics = predictor.evaluate(data)
    y_pred = predictor.predict(data)
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_transformer_explainability(
            predictor.model,
            data,
            explainability_dir,
            task='time',
            num_samples=20,
            methods=explainability_method
        )

    return metrics


def run_gnn_unified_prediction(dataset_path, output_dir, test_size, val_split, config, explainability_method=None, task='unified'):
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. GNN runs cannot execute.")

    df = pd.read_csv(dataset_path)
    df, _, _ = detect_and_standardize_columns(df, verbose=False)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)

    if task == 'unified':
        loss_weights = (1.0, 0.1, 0.1)
    elif task == 'next_activity':
        loss_weights = (1.0, 0.0, 0.0)
    elif task == 'event_time':
        loss_weights = (0.0, 1.0, 0.0)
    elif task == 'remaining_time':
        loss_weights = (0.0, 0.0, 1.0)
    else:
        loss_weights = (1.0, 0.1, 0.1)

    predictor = GNNPredictor(
        hidden_channels=config.get('hidden', 64),
        dropout=config.get('dropout_rate', 0.1),
        lr=config.get('lr', 4e-4),
        loss_weights=loss_weights
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)

    predictor.build_model(
        data['sample_graph'],
        batch_size=config.get('batch_size', 64),
        num_activity_classes=data.get('num_activity_classes')
    )

    predictor.train(
        data,
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 64),
        patience=config.get('patience', 10)
    )

    metrics = predictor.evaluate_test(data, batch_size=config.get('batch_size', 64))
    predictor.save_model(output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_results(metrics, output_dir)

    if explainability_method and EXPLAINABILITY_AVAILABLE:
        explainability_dir = os.path.join(output_dir, 'explainability')
        run_gnn_explainability(
            predictor.model,
            data,
            explainability_dir,
            predictor.device,
            output_dir,
            dataset_path,
            num_samples=10,
            methods=explainability_method
        )

    return metrics
