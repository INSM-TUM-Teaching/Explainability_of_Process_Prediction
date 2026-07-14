# models/base_predictor.py
from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """Strategy contract every model wrapper must implement.

    Wrappers are *adapters*: they delegate to the existing, unmodified predictor
    classes (``NextActivityPredictor``, ``GNNPredictor``, ``BESTRunner`` ...) and
    map their heterogeneous APIs onto this uniform five-method lifecycle. The
    underlying math and optimized data-loading are never touched here.

    ``config`` is a plain ``dict`` carrying both model hyperparameters and runtime
    parameters (``test_size``, ``val_split``, ``output_dir``,
    ``explainability_method``, ``explainability_samples``, ``task``,
    ``target_column`` ...).

    The current pipeline does more than these five steps (build_model, predict,
    save_results, plots, save_model). Those side-steps are folded into the
    nearest lifecycle method so ordering and behavior are preserved:
    ``build_model`` -> ``train``; ``predict``/``save_results``/``plot_*``/
    ``save_model`` -> ``evaluate``.
    """

    @abstractmethod
    def __init__(self, config):
        """Store ``config`` and instantiate the underlying predictor, reading
        hyperparameters from ``config`` with the same defaults as before."""
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self, df):
        """Build and return the model-specific ``data`` object from ``df``
        (delegates to the underlying predictor's ``prepare_data``)."""
        raise NotImplementedError

    @abstractmethod
    def train(self, data):
        """Build the model (where applicable) and run the training loop."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate, persist predictions/plots/model, and return the metrics dict."""
        raise NotImplementedError

    @abstractmethod
    def explain(self, test_data):
        """Run explainability if requested; otherwise a no-op."""
        raise NotImplementedError
