# models/transformer_predictor.py
"""Adapter wrappers around the TensorFlow transformer predictors.

Each wrapper delegates to one of the existing, unmodified predictor classes in
``transformers/prediction/`` and exposes the uniform :class:`BasePredictor`
lifecycle. Concrete predictor classes are imported lazily (inside ``__init__``)
so that importing the registry does not require TensorFlow to be installed.
"""
import os

from .base_predictor import BasePredictor


class _BaseTransformerWrapper(BasePredictor):
    """Common lifecycle shared by all four transformer tasks.

    Subclasses supply: ``EXPLAIN_TASK``, ``_create_predictor``, and the
    persistence/explain hooks that differ per task.
    """

    # Explainability task string passed to run_transformer_explainability.
    EXPLAIN_TASK = None

    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir")
        self.predictor = self._create_predictor(config)

    # --- hooks -------------------------------------------------------------
    def _create_predictor(self, config):
        """Lazily import and construct the underlying predictor."""
        raise NotImplementedError

    def _hyperparams(self, config):
        return dict(
            max_len=config["max_len"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_blocks=config["num_blocks"],
            dropout_rate=config["dropout_rate"],
        )

    def _prepare_extra(self):
        """Extra kwargs for the underlying ``prepare_data`` (task-specific)."""
        return {}

    def _persist(self, data):
        """predict + save_results (+ plot_predictions), task-specific."""
        raise NotImplementedError

    def _explain_label_encoder(self):
        return self.predictor.label_encoder

    def _explain_scaler(self):
        return getattr(self.predictor, "scaler", None)

    # --- lifecycle ---------------------------------------------------------
    def prepare_data(self, df):
        return self.predictor.prepare_data(
            df,
            test_size=self.config["test_size"],
            val_split=self.config["val_split"],
            **self._prepare_extra(),
        )

    def train(self, data):
        self.predictor.build_model()
        self.predictor.train(
            data,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            patience=self.config["patience"],
        )
        return data

    def evaluate(self, test_data):
        metrics = self.predictor.evaluate(test_data)
        self._persist(test_data)
        self.predictor.plot_training_history(self.output_dir)
        self.predictor.save_model(self.output_dir)
        return metrics

    def explain(self, test_data):
        method = self.config.get("explainability_method")
        if not method:
            return
        try:
            from explainability.transformers import run_transformer_explainability
        except ImportError as e:
            raise RuntimeError(
                "Explainability requested, but explainability modules are "
                f"unavailable: {e or 'unknown import error'}"
            )

        explainability_dir = os.path.join(self.output_dir, "explainability")
        num_samples = self.config.get("explainability_samples", 50)
        feature_config = {}
        vocab_size = getattr(self.predictor, "vocab_size", None)
        if vocab_size is not None:
            feature_config["vocab_size"] = vocab_size

        run_transformer_explainability(
            self.predictor.model,
            test_data,
            explainability_dir,
            task=self.EXPLAIN_TASK,
            num_samples=num_samples,
            methods=method,
            label_encoder=self._explain_label_encoder(),
            scaler=self._explain_scaler(),
            feature_config=feature_config,
            run_benchmark=False,
        )


class TransformerNextActivityWrapper(_BaseTransformerWrapper):
    EXPLAIN_TASK = "activity"

    def _create_predictor(self, config):
        from transformers.prediction.next_activity import NextActivityPredictor
        return NextActivityPredictor(**self._hyperparams(config))

    def _prepare_extra(self):
        return dict(
            max_cases=self.config.get("max_cases"),
            max_prefixes_per_case=self.config.get("max_prefixes_per_case"),
            max_graphs=self.config.get("max_graphs"),
        )

    def prepare_data(self, df):
        data = super().prepare_data(df)
        # DEBUG 3 (relocated from run_next_activity_prediction): label encoder state
        print("[DEBUG 3] LABEL ENCODER:", len(self.predictor.label_encoder.classes_), "classes")
        print("Classes:", list(self.predictor.label_encoder.classes_))
        return data

    def _persist(self, data):
        y_pred, y_pred_probs = self.predictor.predict(data)
        self.predictor.save_results(data, y_pred, y_pred_probs, self.output_dir)


class TransformerEventTimeWrapper(_BaseTransformerWrapper):
    EXPLAIN_TASK = "time"

    def _create_predictor(self, config):
        from transformers.prediction.event_time import EventTimePredictor
        return EventTimePredictor(**self._hyperparams(config))

    def _persist(self, data):
        y_pred = self.predictor.predict(data)
        self.predictor.save_results(data, y_pred, self.output_dir)
        self.predictor.plot_predictions(data, y_pred, self.output_dir)

    def _explain_scaler(self):
        return self.predictor.scaler


class TransformerRemainingTimeWrapper(_BaseTransformerWrapper):
    EXPLAIN_TASK = "time"

    def _create_predictor(self, config):
        from transformers.prediction.remaining_time import RemainingTimePredictor
        return RemainingTimePredictor(**self._hyperparams(config))

    def _persist(self, data):
        y_pred = self.predictor.predict(data)
        self.predictor.save_results(data, y_pred, self.output_dir)
        self.predictor.plot_predictions(data, y_pred, self.output_dir)

    def _explain_scaler(self):
        return self.predictor.scaler


class TransformerOutcomeWrapper(_BaseTransformerWrapper):
    EXPLAIN_TASK = "outcome"

    def _create_predictor(self, config):
        from transformers.prediction.outcome import OutcomePredictor
        return OutcomePredictor(**self._hyperparams(config))

    def _prepare_extra(self):
        return dict(
            max_cases=self.config.get("max_cases"),
            max_prefixes_per_case=self.config.get("max_prefixes_per_case"),
        )

    def _persist(self, data):
        y_pred, y_pred_probs = self.predictor.predict(data)
        self.predictor.save_results(data, y_pred, y_pred_probs, self.output_dir)

    def _explain_label_encoder(self):
        return getattr(self.predictor, "activity_encoder", self.predictor.label_encoder)

    def _explain_scaler(self):
        return None
