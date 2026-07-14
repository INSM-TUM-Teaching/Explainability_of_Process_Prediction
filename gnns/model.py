import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data.storage import (
    BaseStorage,
    NodeStorage,
    EdgeStorage,
    GlobalStorage,
)

torch.serialization.add_safe_globals(
    {
        BaseStorage: BaseStorage,
        NodeStorage: NodeStorage,
        EdgeStorage: EdgeStorage,
        GlobalStorage: GlobalStorage,
    }
)


class InputProjector(nn.Module):
    """Projects input features to hidden dimension"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.proj = Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class HeteroGNN(nn.Module):
    """
    Fixed Heterogeneous GNN for process prediction.
    
    Key improvements over original:
    - Uses LAST activity node instead of global mean pooling
    - Preserves temporal order
    - Multiple GNN layers for better representation
    - Balanced loss weights by default
    """


    def __init__(
        self,
        metadata,
        hidden_channels: int,
        proj_dims: dict,
        num_activity_classes: int,
        dropout: float = 0.1,
        loss_weights=(1.0, 1.0, 1.0),
        num_layers: int = 2,
        num_outcome_classes: int = 0,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.loss_weights = loss_weights
        self.num_layers = num_layers
        self.num_outcome_classes = num_outcome_classes

        # Input projection for each node type
        self.proj = nn.ModuleDict(
            {
                ntype: InputProjector(proj_dims[ntype], hidden_channels)
                for ntype in node_types
            }
        )

        # Multiple GNN layers for better temporal modeling
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for src, relname, dst in edge_types:
                if src == dst:
                    conv_dict[(src, relname, dst)] = SAGEConv(hidden_channels, hidden_channels)
                else:
                    conv_dict[(src, relname, dst)] = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Linear transformation after each layer
        self.lins = nn.ModuleList([
            nn.ModuleDict({ntype: Linear(hidden_channels, hidden_channels) for ntype in node_types})
            for _ in range(num_layers)
        ])

        self.drop = nn.Dropout(dropout)

        # Output heads
        self.out_act = Linear(hidden_channels, num_activity_classes)
        self.out_time = Linear(hidden_channels, 1)
        self.out_rem = Linear(hidden_channels, 1)
        self.out_outcome = Linear(hidden_channels, num_outcome_classes) if num_outcome_classes > 0 else None

    def forward(self, data):
        """
        Forward pass using LAST activity node for sequence representation.
        
        This preserves temporal information by using the most recent state
        instead of averaging across all time steps.
        """
        # Project input features
        x_dict = {k: self.proj[k](v) for k, v in data.x_dict.items()}
        
        # Multiple GNN layers
        for layer_idx in range(self.num_layers):
            # Message passing
            x_dict = self.convs[layer_idx](x_dict, data.edge_index_dict)
            
            # Non-linearity and dropout
            x_dict = {
                k: self.drop(torch.relu(self.lins[layer_idx][k](v)))
                for k, v in x_dict.items()
            }
        
        # ========================================
        # KEY FIX: Use LAST activity node instead of global mean pool!
        # ========================================
        
        # Get activity embeddings (most informative for sequence)
        if "activity" in x_dict:
            activity_x = x_dict["activity"]
            activity_batch = getattr(data["activity"], "batch", None)
            
            if activity_batch is None:
                # Single graph - take last activity node
                pooled = activity_x[-1].unsqueeze(0)
            else:
                # Batch of graphs - take last activity node per graph
                pooled = self._get_last_node_per_graph(activity_x, activity_batch)
        
        elif "resource" in x_dict:
            # Fallback to resource if activity not available
            resource_x = x_dict["resource"]
            resource_batch = getattr(data["resource"], "batch", None)
            
            if resource_batch is None:
                pooled = resource_x[-1].unsqueeze(0)
            else:
                pooled = self._get_last_node_per_graph(resource_x, resource_batch)
        
        elif "time" in x_dict:
            # Last fallback to time
            time_x = x_dict["time"]
            time_batch = getattr(data["time"], "batch", None)
            
            if time_batch is None:
                pooled = time_x[-1].unsqueeze(0)
            else:
                pooled = self._get_last_node_per_graph(time_x, time_batch)
        
        else:
            raise ValueError("No activity, resource, or time nodes found in graph!")
        
        # Generate predictions from LAST state
        act = self.out_act(pooled)
        time = self.out_time(pooled).squeeze(-1)
        rem = self.out_rem(pooled).squeeze(-1)
        outcome = self.out_outcome(pooled) if self.out_outcome is not None else None

        return act, time, rem, outcome

    def _get_last_node_per_graph(self, node_features, batch):
        """
        Extract the last node from each graph in a batch.

        Args:
            node_features: [total_nodes, hidden_dim] - all nodes across batch
            batch: [total_nodes] - batch assignment for each node

        Returns:
            [num_graphs, hidden_dim] - last node embedding per graph

        Vectorized: PyG concatenates each graph's nodes contiguously in ascending
        graph order, so the last node of a graph is the position right before the
        batch index changes (plus the final position). This runs on every forward
        pass, so avoiding the per-graph Python loop and the GPU->CPU sync from
        ``batch.max().item()`` is a meaningful speedup during training and eval.
        """
        n = node_features.size(0)
        is_last = torch.ones(n, dtype=torch.bool, device=batch.device)
        if n > 1:
            is_last[:-1] = batch[1:] != batch[:-1]
        last_indices = is_last.nonzero(as_tuple=False).view(-1)
        return node_features[last_indices]

    def compute_loss(self, act_logits, time_pred, rem_pred, batch, outcome_logits=None):
        """
        Compute weighted multi-task loss.

        loss_weights is a 3- or 4-tuple: (act, time, rem[, outcome]).
        """
        y_act = batch.y_activity.view(-1)
        y_time = batch.y_timestamp.view(-1)
        y_rem = batch.y_remaining_time.view(-1)

        w_act = self.loss_weights[0]
        w_time = self.loss_weights[1]
        w_rem = self.loss_weights[2]
        w_outcome = self.loss_weights[3] if len(self.loss_weights) > 3 else 0.0

        loss = (
            w_act * F.cross_entropy(act_logits, y_act)
            + w_time * F.l1_loss(time_pred, y_time)
            + w_rem * F.l1_loss(rem_pred, y_rem)
        )

        if w_outcome > 0.0 and outcome_logits is not None and hasattr(batch, "y_outcome"):
            y_outcome = batch.y_outcome.view(-1)
            loss = loss + w_outcome * F.cross_entropy(outcome_logits, y_outcome)

        return loss