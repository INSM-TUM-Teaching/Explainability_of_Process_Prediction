"""Instantiates the ``HeteroGNN`` module and its optimizer.

Provides :class:`ModelBuilderMixin`. Isolating model construction here is what
makes swapping in a different neural-network architecture straightforward: a new
predictor only has to override ``build_model`` while reusing the data-prep,
training and persistence mixins unchanged.
"""
import torch

from ..model import HeteroGNN


class ModelBuilderMixin:

    def build_model(self, sample_graph, num_activity_classes=None, num_outcome_classes=None, **kwargs):
        print("\nBuilding GNN model...")

        metadata = sample_graph.metadata()
        proj_dims = {k: v.size(-1) for k, v in sample_graph.x_dict.items()}

        if num_activity_classes is None:
            num_classes = int(torch.max(sample_graph.y_activity).item()) + 1
        else:
            num_classes = num_activity_classes

        n_outcome = num_outcome_classes or getattr(self, "_num_outcome_classes", 0)

        self.model = HeteroGNN(
            metadata=metadata,
            hidden_channels=self.hidden_channels,
            proj_dims=proj_dims,
            num_activity_classes=num_classes,
            dropout=self.dropout,
            loss_weights=self.loss_weights,
            num_outcome_classes=n_outcome,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        print(f"Model built with {num_classes} activity classes, {n_outcome} outcome classes")
        print(f"Using device: {self.device}")
