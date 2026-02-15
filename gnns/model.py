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
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.loss_weights = loss_weights
        self.num_layers = num_layers

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
        
        return act, time, rem

    def _get_last_node_per_graph(self, node_features, batch):
        """
        Extract the last node from each graph in a batch.
        
        Args:
            node_features: [total_nodes, hidden_dim] - all nodes across batch
            batch: [total_nodes] - batch assignment for each node
        
        Returns:
            [num_graphs, hidden_dim] - last node embedding per graph
        """
        pooled = []
        num_graphs = batch.max().item() + 1
        
        for graph_id in range(num_graphs):
            # Get all nodes belonging to this graph
            mask = (batch == graph_id)
            graph_nodes = node_features[mask]
            
            # Take the LAST node (most recent in sequence)
            last_node = graph_nodes[-1]
            pooled.append(last_node)
        
        return torch.stack(pooled)

    def compute_loss(self, act_logits, time_pred, rem_pred, batch):
        """
        Compute weighted multi-task loss.
        
        Now with balanced default weights (1.0, 1.0, 1.0) instead of (1.0, 0.1, 0.1)
        """
        y_act = batch.y_activity.view(-1)
        y_time = batch.y_timestamp.view(-1)
        y_rem = batch.y_remaining_time.view(-1)

        w_act, w_time, w_rem = self.loss_weights

        loss = (
            w_act * F.cross_entropy(act_logits, y_act)
            + w_time * F.l1_loss(time_pred, y_time)
            + w_rem * F.l1_loss(rem_pred, y_rem)
        )
        
        return loss