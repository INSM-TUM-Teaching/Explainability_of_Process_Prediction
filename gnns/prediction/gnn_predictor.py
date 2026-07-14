"""``GNNPredictor`` — the orchestrator that ties the GNN pipeline together.

The implementation is split across focused modules so each concern lives in its
own file and a future neural-network predictor can reuse the model-agnostic
pieces:

- :mod:`gnns.prediction.utils`            — seeding, device detection, helpers
- :mod:`gnns.prediction.dataset`          — ``GraphFolderDataset`` (on-disk graphs)
- :mod:`gnns.prediction.data_preparation` — event log -> train/val/test graphs
- :mod:`gnns.prediction.model_builder`    — build the ``HeteroGNN`` + optimizer
- :mod:`gnns.prediction.trainer`          — loaders, training loop, evaluation
- :mod:`gnns.prediction.persistence`      — save model / plots / results

``GNNPredictor`` itself only holds state (hyperparameters + runtime objects) and
inherits its behavior from the mixins below, so its public API is unchanged.
"""
from .data_preparation import DataPreparationMixin
from .model_builder import ModelBuilderMixin
from .trainer import TrainingMixin
from .persistence import PersistenceMixin
from .utils import set_seed, detect_device

# Re-exported so the historical import paths keep working:
#   from gnns.prediction.gnn_predictor import GraphFolderDataset
from .dataset import GraphFolderDataset

__all__ = ["GNNPredictor", "GraphFolderDataset", "set_seed"]


class GNNPredictor(
    DataPreparationMixin,
    ModelBuilderMixin,
    TrainingMixin,
    PersistenceMixin,
):

    def __init__(
        self, hidden_channels=64, dropout=0.1, lr=4e-4, loss_weights=(1.0, 0.1, 0.1)
    ):
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.lr = lr
        self.loss_weights = loss_weights

        self.model = None
        self.optimizer = None
        self.device = self._detect_device()
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_mae_time": [],
            "val_mae_rem": [],
        }

        set_seed()

    def _detect_device(self):
        return detect_device()
