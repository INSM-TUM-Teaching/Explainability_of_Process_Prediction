# models/gnn_predictor.py
"""Adapter wrapper around the PyTorch-Geometric GNN predictor.

A single wrapper serves every GNN task; ``config['task']`` selects the
``loss_weights`` and the set of tasks to explain, exactly as
``run_gnn_unified_prediction`` did. Delegates to the existing, unmodified
``gnns.prediction.gnn_predictor.GNNPredictor`` (imported lazily so importing the
registry does not require PyTorch).
"""
import os

from .base_predictor import BasePredictor

# task -> multi-head loss weights (mirrors run_gnn_unified_prediction)
_LOSS_WEIGHTS = {
    "unified": (1.0, 0.1, 0.1),
    "next_activity": (1.0, 0.0, 0.0),
    "event_time": (0.0, 1.0, 0.0),
    "remaining_time": (0.0, 0.0, 1.0),
    "outcome": (0.0, 0.0, 0.0, 1.0),
}

# task -> explainability heads
_EXPLAIN_TASKS = {
    "unified": ["activity", "event_time", "remaining_time"],
    "next_activity": ["activity"],
    "event_time": ["event_time"],
    "remaining_time": ["remaining_time"],
    "outcome": ["outcome"],
}


class GNNWrapper(BasePredictor):

    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir")
        # custom_activity is trained as next_activity (see run_job routing)
        task = config.get("task", "unified")
        self.task = "next_activity" if task == "custom_activity" else task
        loss_weights = _LOSS_WEIGHTS.get(self.task, (1.0, 0.1, 0.1))

        from gnns.prediction.gnn_predictor import GNNPredictor
        self.predictor = GNNPredictor(
            hidden_channels=config.get("hidden", 64),
            dropout=config.get("dropout_rate", 0.1),
            lr=config.get("lr", 4e-4),
            loss_weights=loss_weights,
        )

    def prepare_data(self, df):
        return self.predictor.prepare_data(
            df,
            test_size=self.config["test_size"],
            val_split=self.config["val_split"],
        )

    def train(self, data):
        batch_size = self.config.get("batch_size", 64)
        self.predictor.build_model(
            data["sample_graph"],
            batch_size=batch_size,
            num_activity_classes=data.get("num_activity_classes"),
        )
        self.predictor.train(
            data,
            epochs=self.config.get("epochs", 50),
            batch_size=batch_size,
            patience=self.config.get("patience", 10),
        )
        return data

    def evaluate(self, test_data):
        metrics = self.predictor.evaluate_test(
            test_data, batch_size=self.config.get("batch_size", 64)
        )
        self.predictor.save_model(self.output_dir)
        self.predictor.plot_training_history(self.output_dir)
        self.predictor.save_results(metrics, self.output_dir)
        return metrics

    def explain(self, test_data):
        method = self.config.get("explainability_method")
        if not method:
            return
        try:
            from explainability.gnns import run_gnn_explainability
        except ImportError as e:
            raise RuntimeError(
                "Explainability requested, but explainability modules are "
                f"unavailable: {e or 'unknown import error'}"
            )

        explainability_dir = os.path.join(self.output_dir, "explainability")
        num_samples = self.config.get("explainability_samples", 50)
        tasks_to_explain = _EXPLAIN_TASKS.get(self.task, ["activity"])

        run_gnn_explainability(
            self.predictor.model,
            test_data,
            explainability_dir,
            self.predictor.device,
            vocabularies=test_data.get("vocabs"),
            num_samples=num_samples,
            methods=method,
            tasks=tasks_to_explain,
            run_benchmark=False,
        )
