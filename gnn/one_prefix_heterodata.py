import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ==============================================
# 1. LOAD PREFIX CSV
# ==============================================
def load_prefix_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df


# ==============================================
# 2. PICK ONE PREFIX (FIRST ONE)
# ==============================================
def pick_first_prefix(df: pd.DataFrame) -> pd.DataFrame:
    first_case = df.iloc[0]["CaseID"]
    first_prefix = df.iloc[0]["prefix_id"]

    prefix = df[(df["CaseID"] == first_case) &
                (df["prefix_id"] == first_prefix)].copy()

    prefix = prefix.sort_values("prefix_pos").reset_index(drop=True)
    return prefix


# ==============================================
# 3. CONVERT ONE PREFIX TO HeteroData (SIMPLE)
# ==============================================
def prefix_to_heterodata(prefix: pd.DataFrame) -> HeteroData:
    """
    Simplest possible HeteroData for ONE prefix.

    Node types:
      - activity  (one per event)
      - resource  (one per event)
      - time      (one per event)
      - trace     (single node)

    Features:
      - activity.x  : integer index of activity
      - resource.x  : integer index of resource
      - time.x      : timestamp as float (seconds)
      - trace.x     : single zero (placeholder)

    Edges:
      - (activity)  i -> i+1
      - (resource)  i -> i+1
      - (time)      i -> i+1
      - trace (0)   -> all activity/resource/time nodes

    Label:
      - y : index of next_activity in the activity vocabulary
    """
    data = HeteroData()

    k = len(prefix)  # prefix length

    # ------------------------------------------
    # 3.1 Build small vocabularies (JUST for this prefix)
    # ------------------------------------------
    # Activities in the prefix + the next_activity
    activities = prefix["Activity"].tolist()
    next_act = prefix.iloc[0]["next_activity"]
    if next_act not in activities:
        activities.append(next_act)
    act2idx = {a: i for i, a in enumerate(activities)}

    # Resources in the prefix
    resources = prefix["Resource"].unique().tolist()
    res2idx = {r: i for i, r in enumerate(resources)}

    # ------------------------------------------
    # 3.2 Activity node features (shape [k, 1])
    # ------------------------------------------
    act_node_ids = [act2idx[a] for a in prefix["Activity"]]
    data["activity"].x = torch.tensor(act_node_ids, dtype=torch.long).unsqueeze(-1)

    # ------------------------------------------
    # 3.3 Resource node features (shape [k, 1])
    # ------------------------------------------
    res_node_ids = [res2idx[r] for r in prefix["Resource"]]
    data["resource"].x = torch.tensor(res_node_ids, dtype=torch.long).unsqueeze(-1)

    # ------------------------------------------
    # 3.4 Time node features (timestamp as float seconds)
    # ------------------------------------------
    # convert Timestamp to POSIX seconds
    times = prefix["Timestamp"].astype("int64") // 10**9  # nanoseconds -> seconds
    times = times.to_list()
    data["time"].x = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)

    # ------------------------------------------
    # 3.5 Trace node (single node, dummy feature)
    # ------------------------------------------
    data["trace"].x = torch.tensor([[0.0]], dtype=torch.float32)  # shape [1, 1]

    # ------------------------------------------
    # 3.6 Edges: sequential inside each type
    # ------------------------------------------
    if k > 1:
        src = torch.arange(0, k - 1, dtype=torch.long)
        dst = torch.arange(1, k, dtype=torch.long)

        data["activity", "next", "activity"].edge_index = torch.stack([src, dst], dim=0)
        data["resource", "next", "resource"].edge_index = torch.stack([src, dst], dim=0)
        data["time", "next", "time"].edge_index = torch.stack([src, dst], dim=0)
    else:
        # empty edge_index if only 1 event
        empty = torch.empty((2, 0), dtype=torch.long)
        data["activity", "next", "activity"].edge_index = empty
        data["resource", "next", "resource"].edge_index = empty
        data["time", "next", "time"].edge_index = empty

    # ------------------------------------------
    # 3.7 Edges: trace -> all event nodes
    # ------------------------------------------
    trace_src = torch.zeros(k, dtype=torch.long)  # always node 0 in 'trace'
    event_idx = torch.arange(0, k, dtype=torch.long)

    data["trace", "to_activity", "activity"].edge_index = torch.stack(
        [trace_src, event_idx], dim=0
    )
    data["trace", "to_resource", "resource"].edge_index = torch.stack(
        [trace_src, event_idx], dim=0
    )
    data["trace", "to_time", "time"].edge_index = torch.stack(
        [trace_src, event_idx], dim=0
    )

    # ------------------------------------------
    # 3.8 Label: next_activity index
    # ------------------------------------------
    y = act2idx[next_act]
    data.y = torch.tensor([y], dtype=torch.long)

    return data


# ==============================================
# MAIN
# ==============================================
def main():
    path = "gnn/data/processed/BPI_2017_Log_O_Offer_prefixes_simple.csv"

    df = load_prefix_data(path)
    prefix = pick_first_prefix(df)

    print("===== PREFIX USED =====")
    print(prefix)

    data = prefix_to_heterodata(prefix)

    print("\n===== HeteroData SUMMARY =====")
    print(data)
    print("\nNode types:", data.node_types)
    print("Edge types:", data.edge_types)
    print("activity.x shape:", data["activity"].x.shape)
    print("resource.x shape:", data["resource"].x.shape)
    print("time.x shape:", data["time"].x.shape)
    print("trace.x shape:", data["trace"].x.shape)
    print("label (y):", data.y)


if __name__ == "__main__":
    main()
