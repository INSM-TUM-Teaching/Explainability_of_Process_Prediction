# models/best_predictor.py
"""Adapter wrapper around the BEST (best4ppm) runner.

A single wrapper serves every BEST task; ``config['task']`` selects the BEST
runner task. Delegates to the existing, unmodified ``best.predictor.BESTRunner``
(imported lazily so importing the registry does not require best4ppm).
"""
from .base_predictor import BasePredictor

# registry task -> BESTRunner task
_TASK_MAP = {
    "next_activity": "nap",
    "remaining_trace": "rtp",
    "outcome": "outcome",
}

# orchestration keys injected by the shims; stripped before reaching BESTRunner
# so they never leak into BEST's saved config/artifacts.
_RESERVED_KEYS = {
    "test_size", "val_split", "output_dir", "explainability_method",
    "explainability_samples", "model_type", "task", "target_column",
}


class BESTWrapper(BasePredictor):

    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir")
        task = config.get("task")
        self.best_task = _TASK_MAP.get(task)
        if self.best_task is None:
            raise ValueError(f"Unsupported best task: {task}")

        best_config = {k: v for k, v in config.items() if k not in _RESERVED_KEYS}

        from best.predictor import BESTRunner
        self.runner = BESTRunner(config=best_config, task=self.best_task)

    def prepare_data(self, df):
        self.runner.prepare_data(df, test_size=self.config["test_size"])
        return df

    def train(self, data):
        self.runner.fit()
        return data

    def evaluate(self, test_data):
        self.runner.predict()
        metrics = self.runner.evaluate()
        self.runner.save_results(self.output_dir)
        self.runner.plot_performance(self.output_dir)
        self.runner.save_model(self.output_dir)
        return metrics

    def explain(self, test_data):
        if not self.config.get("explainability_method"):
            return
        from explainability.best.best_explainer import BESTExplainer
        explainer = BESTExplainer(
            model=self.runner.model,
            output_dir=self.output_dir,
            task=self.best_task,
            runner=self.runner,
        )
        explainer.explain()
