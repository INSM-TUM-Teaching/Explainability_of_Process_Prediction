import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool
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

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.proj = Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class HeteroGNN(nn.Module):

    def __init__(
        self,
        metadata,
        hidden_channels: int,
        proj_dims: dict,
        num_activity_classes: int,
        dropout: float = 0.1,
        loss_weights=(1.0, 0.1, 0.1),
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.loss_weights = loss_weights

        self.proj = nn.ModuleDict(
            {
                ntype: InputProjector(proj_dims[ntype], hidden_channels)
                for ntype in node_types
            }
        )

        def build_rels():
            rel = {}
            for src, relname, dst in edge_types:
                if src == dst:
                    rel[(src, relname, dst)] = SAGEConv(hidden_channels, hidden_channels)
                else:
                    rel[(src, relname, dst)] = SAGEConv((-1, -1), hidden_channels)
            return rel

        self.conv = HeteroConv(build_rels(), aggr="sum")

        self.lin = nn.ModuleDict(
            {ntype: Linear(hidden_channels, hidden_channels) for ntype in node_types}
        )
        self.drop = nn.Dropout(dropout)

        self.out_act = Linear(hidden_channels, num_activity_classes)
        self.out_time = Linear(hidden_channels, 1)
        self.out_rem = Linear(hidden_channels, 1)

    def forward(self, data):
        x_dict = {k: self.proj[k](v) for k, v in data.x_dict.items()}
        x_dict = self.conv(x_dict, data.edge_index_dict)
        x_dict = {
            k: self.drop(torch.relu(self.lin[k](v)))
            for k, v in x_dict.items()
        }

        if "trace" in x_dict:
            trace_x = x_dict["trace"]
            trace_batch = getattr(data["trace"], "batch", None)
            if trace_batch is None:
                trace_batch = torch.zeros(len(trace_x), dtype=torch.long, device=trace_x.device)
        else:
            first_type = list(x_dict.keys())[0]
            trace_x = x_dict[first_type]
            trace_batch = torch.zeros(len(trace_x), dtype=torch.long, device=trace_x.device)

        pooled = global_mean_pool(trace_x, trace_batch)

        act = self.out_act(pooled)
        time = self.out_time(pooled).squeeze(-1)
        rem = self.out_rem(pooled).squeeze(-1)
        return act, time, rem

    def compute_loss(self, act_logits, time_pred, rem_pred, batch):
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
