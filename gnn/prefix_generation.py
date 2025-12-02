import os
import sys
from pathlib import Path
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("Project root:", PROJECT_ROOT)

# Folder containing all preprocessed CSVs
INPUT_FOLDER = PROJECT_ROOT / "BPI_Models" / "BPI_logs_preprocessed_csv"

# Output folder
OUTPUT_FOLDER = PROJECT_ROOT / "gnn" / "data" / "processed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CASE_COL = "CaseID"
TIME_COL = "Timestamp"


#load and sort the logs before creating the prefix by caseid and timestamp(even though it was done in the data preprocessing)
def load_and_sort_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    return df


#attributes who are constatnt(ex:caseid) through the trace and dont change are detected to be used later 
def detect_trace_attributes(df: pd.DataFrame):
    trace_cols = []
    grouped = df.groupby(CASE_COL)

    for col in df.columns:
        if col == CASE_COL:
            continue
        if grouped[col].nunique().max() == 1:
            trace_cols.append(col)

    return trace_cols



def generate_prefix_dataset(df: pd.DataFrame, trace_cols):
    rows = []

    for case_id, group in df.groupby(CASE_COL):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        trace_len = len(group)

        # constant attributes
        trace_attrs = {c: group[c].iloc[0] for c in trace_cols}

        for k in range(1, trace_len):
            label_next_activity = group.iloc[k]["Activity"]
            prefix = group.iloc[:k]

            for pos, (_, ev) in enumerate(prefix.iterrows(), start=1):
                row = {
                    "CaseID": case_id,
                    "prefix_id": k,
                    "prefix_pos": pos,
                    "prefix_length": k,
                    "Activity": ev["Activity"],
                    "Resource": ev["Resource"],
                    "Timestamp": ev[TIME_COL],
                    "next_activity": label_next_activity,
                }
                row.update(trace_attrs)
                rows.append(row)

    return pd.DataFrame(rows)


#does it for all the preprocessed data
def main():
    print("PREFIX GENERATOR STARTED")

    csv_files = sorted(INPUT_FOLDER.glob("*_preprocessed_log.csv"))

    if not csv_files:
        print("No CSV files found in:", INPUT_FOLDER)
        return

    print(f"Found {len(csv_files)} files.")

    for csv_path in csv_files:
        print(f"\n--- Processing {csv_path.name} ---")

        df = load_and_sort_log(csv_path)
        trace_cols = detect_trace_attributes(df)

        prefix_df = generate_prefix_dataset(df, trace_cols)

        output_path = OUTPUT_FOLDER / f"{csv_path.stem}_prefixes.csv"
        prefix_df.to_csv(output_path, index=False)

        print(f"✓ Saved: {output_path} ({len(prefix_df)} rows)")

    print("ALL DONE")


if __name__ == "__main__":
    main()