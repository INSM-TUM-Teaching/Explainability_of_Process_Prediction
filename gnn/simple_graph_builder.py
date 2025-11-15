import pandas as pd

# ==============================================
# LOAD PREFIX CSV
# ==============================================
def load_prefix_data(path):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df

# ==============================================
# TAKE ONE PREFIX (FIRST ONE)
# ==============================================
def pick_first_prefix(df):
    first_case = df.iloc[0]["CaseID"]
    first_prefix = df.iloc[0]["prefix_id"]

    prefix = df[(df["CaseID"] == first_case) &
                (df["prefix_id"] == first_prefix)].copy()

    prefix = prefix.sort_values("prefix_pos")
    return prefix

# ==============================================
# BUILD A VERY SIMPLE GRAPH DICT
# ==============================================
def build_graph(prefix):
    graph = {
        "activity_nodes": [],
        "resource_nodes": [],
        "time_nodes": [],
        "trace_node": {},
        "edges": {
            "activity_next": [],
            "resource_next": [],
            "time_next": [],
            "trace_to_activity": [],
            "trace_to_resource": [],
            "trace_to_time": []
        },
        "label": prefix.iloc[0]["next_activity"]
    }

    # ----- Extract trace attributes -----
    trace_cols = [c for c in prefix.columns
                  if c not in ["CaseID", "prefix_id", "prefix_pos",
                               "prefix_length", "Activity", "Resource",
                               "Timestamp", "next_activity"]]

    trace_attrs = {c: prefix.iloc[0][c] for c in trace_cols}

    graph["trace_node"] = trace_attrs

    # ----- Build nodes -----
    for i, row in prefix.iterrows():
        pos = int(row["prefix_pos"])  # 1..k
        idx = pos - 1                  # 0..k-1

        graph["activity_nodes"].append({
            "id": idx,
            "value": row["Activity"]
        })

        graph["resource_nodes"].append({
            "id": idx,
            "value": row["Resource"]
        })

        graph["time_nodes"].append({
            "id": idx,
            "value": row["Timestamp"]
        })

    # ----- Build edges -----
    k = len(prefix)

    for i in range(k - 1):
        graph["edges"]["activity_next"].append((i, i+1))
        graph["edges"]["resource_next"].append((i, i+1))
        graph["edges"]["time_next"].append((i, i+1))

    for i in range(k):
        graph["edges"]["trace_to_activity"].append(("trace", i))
        graph["edges"]["trace_to_resource"].append(("trace", i))
        graph["edges"]["trace_to_time"].append(("trace", i))

    return graph

# ==============================================
# MAIN
# ==============================================
def main():
    path = "gnn/data/processed/BPI_2017_Log_O_Offer_prefixes_simple.csv"

    df = load_prefix_data(path)
    prefix = pick_first_prefix(df)
    graph = build_graph(prefix)

    print("\n===== PREFIX USED =====")
    print(prefix)

    print("\n===== GRAPH BUILT =====")
    print(graph)

if __name__ == "__main__":
    main()
