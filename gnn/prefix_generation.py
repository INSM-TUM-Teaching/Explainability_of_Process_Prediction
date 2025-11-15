import os
import sys
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------
# CONFIG + PATHS
# -----------------------------------------------------------

# Show which Python / cwd we are using (helps debug venv issues)
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working dir:", os.getcwd())

# Project root = folder that contains "BPI_Models" and "gnn"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("Project root resolved to:", PROJECT_ROOT)

# Choose dataset
DATASET_NAME = "BPI_2017_Log_O_Offer"

# Input = preprocessed log CSV
INPUT_CSV = (
    PROJECT_ROOT
    / "BPI_Models"/ "BPI_logs_preprocessed_csv"/ f"{DATASET_NAME}_preprocessed_log.csv"
)

# Output = prefix-level dataset
OUTPUT_CSV = (
    PROJECT_ROOT/ "gnn"/ "data"/ "processed"/ f"{DATASET_NAME}_prefixes_simple.csv"
)

# Column names in the CSV
CASE_COL = "CaseID"
TIME_COL = "Timestamp"

print("INPUT_CSV:", INPUT_CSV)
print("Does INPUT_CSV exist?", INPUT_CSV.exists())


# -----------------------------------------------------------
# STEP 0: LOAD + SORT LOG
# -----------------------------------------------------------

def load_and_sort_log() -> pd.DataFrame:
    """
    Load the preprocessed CSV and sort it by case and time.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    print(f"\nLoading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIME_COL])

    # sort by case + time so each trace is in correct order
    df = df.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)

    print(f"Events: {len(df)}, Cases: {df[CASE_COL].nunique()}")
    print("Columns:", list(df.columns))
    print("\nFirst 5 events after sorting:")
    print(df.head())

    return df


# -----------------------------------------------------------
# STEP 0.5: DETECT TRACE ATTRIBUTES
# -----------------------------------------------------------

def detect_trace_attributes(df: pd.DataFrame, case_col: str = CASE_COL):
    """
    Detect columns that are constant within each CaseID.
    These are our trace attributes (context of the case).
    """
    trace_cols = []
    grouped = df.groupby(case_col)

    for col in df.columns:
        if col == case_col:
            continue
        # max distinct values this column takes inside ANY single case
        max_unique_within_case = grouped[col].nunique().max()
        if max_unique_within_case == 1:
            trace_cols.append(col)

    return trace_cols


# -----------------------------------------------------------
# STEP 1: GENERATE PREFIX DATASET (WITH LABEL)
# -----------------------------------------------------------

def generate_prefix_dataset(
    df: pd.DataFrame,
    trace_cols,
    case_col: str = CASE_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """
    Generate a prefix-level dataset for SephiGraph.

    Each row = one event inside one prefix.

    We keep only:
      - CaseID
      - prefix_id, prefix_pos, prefix_length
      - Activity, Resource, Timestamp
      - all trace attributes
      - next_activity (label for next-activity prediction)
    """
    rows = []

    for case_id, group in df.groupby(case_col):
        # ensure this case is sorted by time
        group = group.sort_values(time_col).reset_index(drop=True)
        trace_len = len(group)

        # collect trace attributes for this case (same for all events)
        trace_attrs = {col: group[col].iloc[0] for col in trace_cols}

        # we only consider prefixes that HAVE a next event
        # so k = 1 .. trace_len-1
        for k in range(1, trace_len):
            prefix = group.iloc[:k]

            # label = activity of event at position k (0-based index k)
            label_next_activity = group.iloc[k]["Activity"]

            for pos, (_, ev) in enumerate(prefix.iterrows(), start=1):
                row = {
                    "CaseID": ev[case_col],
                    "prefix_id": k,
                    "prefix_pos": pos,
                    "prefix_length": k,
                    "Activity": ev["Activity"],
                    "Resource": ev["Resource"],
                    "Timestamp": ev[time_col],
                    "next_activity": label_next_activity,
                }

                # add all trace attributes (same for entire case)
                for tcol, tval in trace_attrs.items():
                    row[tcol] = tval

                rows.append(row)

    prefix_df = pd.DataFrame(rows)
    return prefix_df


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    # Step 0: load + sort log
    df = load_and_sort_log()

    # Step 0.5: detect trace attributes
    trace_cols = detect_trace_attributes(df, case_col=CASE_COL)
    print("\nDetected trace attributes:")
    for c in trace_cols:
        print("  -", c)

    # Step 1: generate prefix dataset
    print("\nGenerating prefix dataset...")
    prefix_df = generate_prefix_dataset(
        df,
        trace_cols,
        case_col=CASE_COL,
        time_col=TIME_COL,
    )

    print(f"Total prefix-event rows: {len(prefix_df)}")
    num_prefixes = prefix_df[["CaseID", "prefix_id"]].drop_duplicates().shape[0]
    print(f"Total prefixes: {num_prefixes}")

    # ensure output folder exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # save to CSV
    prefix_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Prefix dataset saved to: {OUTPUT_CSV}")

    # preview
    print("\nExample prefix rows:")
    print(prefix_df.head(10))


if __name__ == "__main__":
    main()
