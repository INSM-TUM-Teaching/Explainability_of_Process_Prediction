"""Turns an event-log DataFrame into train/val/test lists of PyG graphs.

Provides :class:`DataPreparationMixin`, the graph-construction half of
``GNNPredictor``. It is kept separate from training/evaluation so the (heavy)
data-to-graph conversion can be understood and modified in isolation.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import GraphFolderDataset
from .utils import signed_log1p


class DataPreparationMixin:

    def prepare_data(self, df, test_size=0.3, val_split=0.5, task='unified'):
        from torch_geometric.data import HeteroData

        print("\nPreparing data for GNN...")
        self._gnn_task = task
        split_col = "__split" if "__split" in df.columns else None
        if split_col:
            df = df.copy()

        # Compute case outcomes (last activity per case)
        case_outcomes = df.sort_values(["CaseID", "Timestamp"]).groupby("CaseID")["Activity"].last()
        outcome_labels = sorted(case_outcomes.unique().tolist())
        self._outcome_vocab = {v: i for i, v in enumerate(outcome_labels)}
        self._num_outcome_classes = len(outcome_labels)
        print(f"Outcome classes: {len(outcome_labels)}")

        trace_cols = []
        grouped = df.groupby("CaseID")
        for col in df.columns:
            if col in ["CaseID", "Activity", "Resource", "Timestamp", "__split"]:
                continue
            if grouped[col].nunique().max() == 1:
                trace_cols.append(col)

        print(f"Detected trace attributes: {trace_cols}")

        def build_prefix_df(input_df):
            rows = []
            total_cases = input_df["CaseID"].nunique()
            processed_cases = 0
            for case_id, group in input_df.groupby("CaseID"):
                processed_cases += 1
                group = group.sort_values("Timestamp").reset_index(drop=True)
                trace_len = len(group)
                trace_attrs = {
                    c: group[c].iloc[0] for c in trace_cols if c in group.columns
                }

                if trace_len <= 1:
                    continue

                activities = group["Activity"].to_numpy()
                timestamps = group["Timestamp"].to_numpy()
                resources = (
                    group["Resource"].to_numpy()
                    if "Resource" in group.columns
                    else None
                )

                case_end_timestamp = timestamps[-1]
                for k in range(1, trace_len):
                    label_next_activity = activities[k]
                    label_next_timestamp = timestamps[k]
                    for pos in range(k):
                        row = {
                            "CaseID": case_id,
                            "prefix_id": k,
                            "prefix_pos": pos + 1,
                            "prefix_length": k,
                            "Activity": activities[pos],
                            "Resource": (
                                resources[pos] if resources is not None else "Unknown"
                            ),
                            "Timestamp": timestamps[pos],
                            "case_end_timestamp": case_end_timestamp,
                            "next_activity": label_next_activity,
                            "next_timestamp": label_next_timestamp,
                        }
                        if trace_attrs:
                            row.update(trace_attrs)
                        rows.append(row)

                if processed_cases % 200 == 0:
                    print(
                        f"[GNN] Prefix build progress: "
                        f"{processed_cases}/{total_cases} cases, "
                        f"{len(rows):,} prefix rows"
                    )

            prefix_df = pd.DataFrame(rows)
            print(f"Generated {len(prefix_df):,} prefix rows")
            prefix_df["__ts_log"] = np.log1p(
                prefix_df["Timestamp"].astype("int64") // 1_000_000_000
            ).astype("float32")
            return prefix_df

        base_df = df.drop(columns=[split_col]) if split_col else df
        prefix_df = build_prefix_df(base_df)

        vocabs = {}
        all_activities = set(prefix_df["Activity"].unique().tolist()) | set(
            prefix_df["next_activity"].unique().tolist()
        )
        values = sorted(all_activities)
        vocabs["Activity"] = {v: i for i, v in enumerate(values)}
        res_vals = sorted(prefix_df["Resource"].unique().tolist())
        vocabs["Resource"] = {v: i for i, v in enumerate(res_vals)}

        IGNORE_COLS = {
            "CaseID",
            "prefix_id",
            "prefix_pos",
            "prefix_length",
            "Activity",
            "Resource",
            "Timestamp",
            "next_activity",
            "__ts_log",
            "case_end_timestamp",
            "next_timestamp",
        }
        trace_attributes = [col for col in prefix_df.columns if col not in IGNORE_COLS]

        for col in trace_attributes:
            if pd.api.types.is_numeric_dtype(prefix_df[col]):
                continue
            vals = sorted(prefix_df[col].fillna("NaN").unique().tolist())
            vocabs[col] = {v: i for i, v in enumerate(vals)}
        self.vocabs = vocabs
        print(
            f"Vocabularies: Activities={len(vocabs['Activity'])}, Resources={len(vocabs['Resource'])}"
        )

        def build_graphs(prefix_df):
            print("Pre-sorting dataset...")
            prefix_df = prefix_df.sort_values(by=["CaseID", "prefix_id", "prefix_pos"])

            # 1. GET GROUP SIZES INSTEAD OF DATA FRAMES
            # We just count how many rows belong to each graph
            group_sizes = (
                prefix_df.groupby(["CaseID", "prefix_id"], sort=False).size().values
            )

            print(f"Building {len(group_sizes):,} graphs...")

            act_map = vocabs["Activity"]
            res_map = vocabs["Resource"]

            # 2. EXTRACT RAW ARRAYS ONCE (Bypass Pandas completely for the loop)
            activities = prefix_df["Activity"].values
            resources = prefix_df["Resource"].values
            ts_logs = prefix_df["__ts_log"].values
            timestamps = prefix_df["Timestamp"].tolist()
            case_end_timestamps = prefix_df["case_end_timestamp"].tolist()
            next_timestamps = prefix_df["next_timestamp"].tolist()
            next_activities = prefix_df["next_activity"].values
            case_ids = prefix_df["CaseID"].values

            trace_data = {}
            for col in trace_attributes:
                if col in prefix_df.columns:
                    trace_data[col] = prefix_df[col].values

            graphs = []
            case_indexes = {}

            # 3. SLICE ARRAYS USING INDEXING
            start_idx = 0
            for size in tqdm(group_sizes, desc="Building graphs", ncols=100):
                end_idx = start_idx + size
                k = size

                # Instantly grab the data for this graph using pure array slicing
                p_activities = activities[start_idx:end_idx]
                p_resources = resources[start_idx:end_idx]
                p_ts_log = ts_logs[start_idx:end_idx]
                p_timestamps = timestamps[start_idx:end_idx]

                data = HeteroData()

                act_ids = np.array([act_map.get(a, 0) for a in p_activities])
                data["activity"].x = F.one_hot(
                    torch.tensor(act_ids, dtype=torch.long), num_classes=len(act_map)
                ).float()
                data["activity"].num_nodes = k

                res_ids = np.array([res_map.get(r, 0) for r in p_resources])
                data["resource"].x = F.one_hot(
                    torch.tensor(res_ids, dtype=torch.long), num_classes=len(res_map)
                ).float()
                data["resource"].num_nodes = k

                data["time"].x = torch.tensor(p_ts_log, dtype=torch.float32).unsqueeze(
                    1
                )
                data["time"].num_nodes = k

                trace_features = []
                for col in trace_attributes:
                    if col not in trace_data:
                        continue

                    # Fast access to the first item of this specific group
                    val = trace_data[col][start_idx]

                    if col in vocabs:
                        idx = vocabs[col].get(val, 0)
                        trace_features.append(
                            F.one_hot(
                                torch.tensor(idx), num_classes=len(self.vocabs[col])
                            ).float()
                        )
                    else:
                        try:
                            trace_features.append(
                                torch.tensor([signed_log1p(val)], dtype=torch.float32)
                            )
                        except Exception:
                            trace_features.append(torch.zeros(1))

                if not trace_features:
                    trace_features = [torch.zeros(1)]
                data["trace"].x = torch.cat(trace_features).unsqueeze(0)
                data["trace"].num_nodes = 1

                idx = torch.arange(k)
                if k > 1:
                    dfr = torch.stack([idx[:-1], idx[1:]])
                else:
                    dfr = torch.empty((2, 0), dtype=torch.long)

                data["activity", "next", "activity"].edge_index = dfr
                data["resource", "next", "resource"].edge_index = dfr.clone()
                data["time", "next", "time"].edge_index = dfr.clone()

                same_ev = torch.stack([idx, idx])
                data["activity", "same_event", "resource"].edge_index = same_ev
                data["resource", "same_event", "activity"].edge_index = same_ev.clone()
                data["activity", "same_time", "time"].edge_index = same_ev.clone()
                data["time", "same_time", "activity"].edge_index = same_ev.clone()

                trace_src = torch.zeros(k, dtype=torch.long)
                data["activity", "to_trace", "trace"].edge_index = torch.stack(
                    [idx, trace_src]
                )
                data["resource", "to_trace", "trace"].edge_index = torch.stack(
                    [idx, trace_src]
                )
                data["time", "to_trace", "trace"].edge_index = torch.stack(
                    [idx, trace_src]
                )

                # Targets mapping
                next_act_name = next_activities[start_idx]
                if next_act_name not in act_map:
                    print(
                        f"Warning: Activity '{next_act_name}' not in vocabulary. Skipping graph."
                    )
                    start_idx = (
                        end_idx  # Don't forget to advance the index before skipping!
                    )
                    continue

                next_act = act_map[next_act_name]
                data.y_activity = torch.tensor([next_act], dtype=torch.long)

                t_next_abs = next_timestamps[start_idx].timestamp()
                t_now = p_timestamps[-1].timestamp()
                time_to_next = max(0, t_next_abs - t_now)
                data.y_timestamp = torch.tensor([np.log1p(time_to_next)], dtype=torch.float32)

                t_end = case_end_timestamps[start_idx].timestamp()
                t_now = p_timestamps[-1].timestamp()
                remaining = max(0, t_end - t_now)
                data.y_remaining_time = torch.tensor(
                    [np.log1p(remaining)], dtype=torch.float32
                )

                cid = case_ids[start_idx]
                if cid not in case_indexes:
                    case_indexes[cid] = 1
                else:
                    case_indexes[cid] += 1

                data.case_id = cid
                data.case_index = case_indexes[cid]

                outcome_name = case_outcomes.get(cid)
                if outcome_name is not None and outcome_name in self._outcome_vocab:
                    data.y_outcome = torch.tensor(
                        [self._outcome_vocab[outcome_name]], dtype=torch.long
                    )
                else:
                    data.y_outcome = torch.tensor([0], dtype=torch.long)

                graphs.append(data)

                # Move the index window forward for the next graph
                start_idx = end_idx

            return graphs

        if split_col:
            split_values = set(df[split_col].dropna().unique().tolist())
            if not {"train", "val", "test"}.issubset(split_values):
                raise ValueError(
                    "Split column must include train, val, and test values."
                )

            train_df = df[df[split_col] == "train"].drop(columns=[split_col])
            val_df = df[df[split_col] == "val"].drop(columns=[split_col])
            test_df = df[df[split_col] == "test"].drop(columns=[split_col])

            train_graphs = build_graphs(build_prefix_df(train_df))
            val_graphs = build_graphs(build_prefix_df(val_df))
            test_graphs = build_graphs(build_prefix_df(test_df))
            graphs = train_graphs + val_graphs + test_graphs
        else:
            graphs = build_graphs(prefix_df)

        print(f"[OK] Built {len(graphs):,} graphs")

        if not split_col:
            n_total = len(graphs)
            n_test = int(n_total * test_size)
            n_train_val = n_total - n_test
            n_val = int(n_train_val * val_split)
            n_train = n_train_val - n_val

            train_graphs = graphs[:n_train]
            val_graphs = graphs[n_train : n_train + n_val]
            test_graphs = graphs[n_train + n_val :]

        print(
            f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}"
        )

        return {
            "train": train_graphs,
            "val": val_graphs,
            "test": test_graphs,
            "sample_graph": graphs[0],
            "num_activity_classes": len(vocabs["Activity"]),
            "num_outcome_classes": self._num_outcome_classes,
            "vocabs": vocabs,
        }

    def prepare_data_from_graphs(self, graph_folder, test_size=0.3, val_split=0.5):
        print("\nLoading graph dataset...")
        dataset = GraphFolderDataset(graph_folder)
        print(f"Found {len(dataset)} graphs")

        n = len(dataset)
        test_val_size = int(test_size * n)
        train_size = n - test_val_size
        val_size = int(val_split * test_val_size)
        test_size = test_val_size - val_size

        print(f"\nDataset splits:")
        print(f"Train: {train_size:,} samples")
        print(f"Validation: {val_size:,} samples")
        print(f"Test: {test_size:,} samples")

        train_graphs = [dataset[i] for i in range(train_size)]
        val_graphs = [dataset[i] for i in range(train_size, train_size + val_size)]
        test_graphs = [dataset[i] for i in range(train_size + val_size, n)]

        return {
            "train_graphs": train_graphs,
            "val_graphs": val_graphs,
            "test_graphs": test_graphs,
            "sample_graph": dataset[0],
        }
