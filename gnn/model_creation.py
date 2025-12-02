import argparse
import os
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.data.storage import (
    BaseStorage,
    NodeStorage,
    EdgeStorage,
    GlobalStorage,
)
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool

# ---------------------------------------------------------------------------
# SAFE GLOBALS FOR PyTorch 2.x + PyG
# ---------------------------------------------------------------------------
torch.serialization.add_safe_globals(
    {
        BaseStorage: BaseStorage,
        NodeStorage: NodeStorage,
        EdgeStorage: EdgeStorage,
        GlobalStorage: GlobalStorage,
    }
)

# ---------------------------------------------------------------------------
# SEEDING FOR REPRODUCIBILITY
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ---------------------------------------------------------------------------
# DATASET LOADER (LAZY)
# ---------------------------------------------------------------------------
class GraphFolderDataset(Dataset):
    """Loads individual prefix graphs lazily from folder (dataset_builder.py output)."""

    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        self.folder = folder
        self.files: List[str] = sorted(
            [f for f in os.listdir(folder) if f.endswith(".pt")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt graphs found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        graph = torch.load(path, map_location="cpu", weights_only=False)
        return graph


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------
def resolve_dataset_folder(req: Optional[str], base: str = "gnn/data/datasets"):
    if req:
        if os.path.isdir(req):
            return req
        alt = os.path.join(base, req)
        if os.path.isdir(alt):
            return alt
        raise FileNotFoundError(f"Cannot find dataset folder: {req}")

    folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    if not folders:
        raise FileNotFoundError("No datasets found.")
    return os.path.join(base, sorted(folders)[0])


def make_splits(n: int) -> Tuple[int, int, int]:
    train_len = max(1, int(0.8 * n))
    val_len = max(1, int(0.1 * n))
    if train_len + val_len > n:
        val_len = n - train_len
    test_len = n - train_len - val_len
    if test_len <= 0 and n >= 3:
        test_len = 1
        train_len -= 1
    return train_len, val_len, test_len


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class InputProjector(nn.Module):
    """Projects input feature vectors to smaller dense embeddings."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.proj = Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class HeteroGNN(nn.Module):
    """HeteroConv-based GNN for activity and timestamp predictions."""

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

        # Projection layers (per node type)
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

        # Output heads
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

        # ---- Trace pooling ----
        if "trace" in x_dict:
            trace_x = x_dict["trace"]
            trace_batch = getattr(data["trace"], "batch", None)
            if trace_batch is None:
                trace_batch = torch.zeros(len(trace_x), dtype=torch.long)
        else:
            # fallback: take ANY node type
            first_type = list(x_dict.keys())[0]
            trace_x = x_dict[first_type]
            trace_batch = torch.zeros(len(trace_x), dtype=torch.long)

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


# ---------------------------------------------------------------------------
# TRAIN & EVAL
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        act_logits, time_pred, rem_pred = model(batch)
        loss = model.compute_loss(act_logits, time_pred, rem_pred, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    mae_time = 0.0
    mae_rem = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            act_logits, time_pred, rem_pred = model(batch)
            loss = model.compute_loss(act_logits, time_pred, rem_pred, batch)
            total_loss += loss.item()

            y_act = batch.y_activity.view(-1)
            pred = act_logits.argmax(dim=1)
            correct += (pred == y_act).sum().item()
            total += y_act.numel()

            y_time = batch.y_timestamp.view(-1)
            y_rem = batch.y_remaining_time.view(-1)
            mae_time += torch.abs(time_pred - y_time).mean().item()
            mae_rem += torch.abs(rem_pred - y_rem).mean().item()

    return (
        correct / total,
        mae_time / len(loader),
        mae_rem / len(loader),
        total_loss / len(loader),
    )


# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device()
    print(f"Using device: {device}")

    dataset_path = resolve_dataset_folder(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    dataset = GraphFolderDataset(dataset_path)
    print(f"Found {len(dataset)} graphs")

    # ---- Splits ----
    train_len, val_len, test_len = make_splits(len(dataset))
    print(f"Splits: train={train_len} | val={val_len} | test={test_len}")

    train_graphs = [dataset[i] for i in range(train_len)]
    val_graphs = [dataset[i] for i in range(train_len, train_len + val_len)]
    test_graphs = [dataset[i] for i in range(train_len + val_len, len(dataset))]

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Batch.from_data_list,
    )

    train_loader = torch.utils.data.DataLoader(train_graphs, **loader_args)
    val_loader = torch.utils.data.DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=Batch.from_data_list
    )
    test_loader = torch.utils.data.DataLoader(
        test_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=Batch.from_data_list
    )

    # ---- Model ----
    sample = dataset[0]
    metadata = sample.metadata()
    proj_dims = {k: v.size(-1) for k, v in sample.x_dict.items()}
    num_classes = int(torch.max(sample.y_activity).item()) + 1

    model = HeteroGNN(
        metadata=metadata,
        hidden_channels=args.hidden,
        proj_dims=proj_dims,
        num_activity_classes=num_classes,
        dropout=0.1,
        loss_weights=(1.0, 0.1, 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc, _, _, _ = evaluate(model, train_loader, device)
        val_acc, val_mae_t, val_mae_r, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"MAE Time: {val_mae_t:.4f} | "
            f"MAE Rem: {val_mae_r:.4f}"
        )

    print("\nTesting:")
    test_acc, test_mae_t, test_mae_r, _ = evaluate(model, test_loader, device)
    print(f"TEST — Acc: {test_acc*100:.2f}% | Time MAE={test_mae_t:.4f} | Rem MAE={test_mae_r:.4f}")

    # ---- Save model ----
    save_path = "gnn/model/model_latest.pt"
    os.makedirs("gnn/model", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved → {save_path}")


if __name__ == "__main__":
    main()
