# gnn/dataset_builder.py

import pandas as pd
import torch
from torch_geometric.data import HeteroData


# =======================================
# STEP 1 — LOAD FULL PREFIX TABLE
# =======================================

def load_prefix_table(path: str) -> pd.DataFrame:
    print(f"Loading: {path}")
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    print(f"Loaded {len(df)} rows")
    return df


# =======================================
# STEP 2 — BUILD GLOBAL VOCABULARIES
# =======================================

def build_global_vocabs(df: pd.DataFrame):
    activities = sorted(df["Activity"].unique().tolist() +
                        df["next_activity"].unique().tolist())

    resources = sorted(df["Resource"].unique().tolist())

    activity2idx = {a: i for i, a in enumerate(activities)}
    resource2idx = {r: i for i, r in enumerate(resources)}

    print("\nGlobal vocabularies:")
    print(f"  Activities: {len(activity2idx)} classes")
    print(f"  Resources:  {len(resource2idx)} classes")

    return activity2idx, resource2idx


# =======================================
# STEP 3 — BUILD ONE HETERODATA GRAPH
# =======================================

def build_graph(prefix: pd.DataFrame, activity2idx, resource2idx):
    data = HeteroData()
    k = len(prefix)

    # Activity node features
    act_ids = [activity2idx[a] for a in prefix["Activity"]]
    data["activity"].x = torch.tensor(act_ids, dtype=torch.long).unsqueeze(-1)

    # Resource node features
    res_ids = [resource2idx[r] for r in prefix["Resource"]]
    data["resource"].x = torch.tensor(res_ids, dtype=torch.long).unsqueeze(-1)

    # Time node features (POSIX seconds)
    times = (prefix["Timestamp"].astype("int64") // 10**9).tolist()
    data["time"].x = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)

    # Single trace node
    data["trace"].x = torch.tensor([[0.0]], dtype=torch.float32)

    # Sequential edges: event i → event i+1
    if k > 1:
        src = torch.arange(0, k - 1)
        dst = torch.arange(1, k)
        edge_index = torch.stack([src, dst])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data["activity", "next", "activity"].edge_index = edge_index
    data["resource", "next", "resource"].edge_index = edge_index.clone()
    data["time", "next", "time"].edge_index = edge_index.clone()

    # Trace → all event nodes
    trace_src = torch.zeros(k, dtype=torch.long)
    idxs = torch.arange(k)

    data["trace", "to_activity", "activity"].edge_index = torch.stack([trace_src, idxs])
    data["trace", "to_resource", "resource"].edge_index = torch.stack([trace_src, idxs])
    data["trace", "to_time", "time"].edge_index = torch.stack([trace_src, idxs])

    # LABEL: next activity (classification)
    next_act = prefix.iloc[0]["next_activity"]
    data.y = torch.tensor([activity2idx[next_act]], dtype=torch.long)

    return data


# =======================================
# STEP 4 — FULL DATASET BUILDER
# =======================================

def build_full_dataset(df: pd.DataFrame, activity2idx, resource2idx):

    dataset = []

    grouped = df.groupby(["CaseID", "prefix_id"])

    for (case_id, pid), prefix_rows in grouped:
        prefix_rows = prefix_rows.sort_values("prefix_pos")

        graph = build_graph(prefix_rows, activity2idx, resource2idx)
        dataset.append(graph)

    print(f"\nTotal graphs created: {len(dataset)}")
    return dataset


# =======================================
# MAIN EXECUTION
# =======================================

def main():
    path = "gnn/data/processed/BPI_2017_Log_O_Offer_prefixes_simple.csv"

    df = load_prefix_table(path)

    activity2idx, resource2idx = build_global_vocabs(df)

    dataset = build_full_dataset(df, activity2idx, resource2idx)

    print("\nDataset example:")
    print(dataset[0])
    print("Node types:", dataset[0].node_types)
    print("Edge types:", dataset[0].edge_types)
    print("Label:", dataset[0].y)


if __name__ == "__main__":
    main()
