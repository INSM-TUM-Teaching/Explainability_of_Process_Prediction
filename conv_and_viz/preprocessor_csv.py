import pandas as pd
import pm4py
import os
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter


def detect_column_type(series):
    if series.dtype in ['object', 'string', 'category', 'bool']:
        return 'categorical'
    elif series.dtype in ['int64', 'int32', 'float64', 'float32']:
        return 'numerical'
    else:
        return 'categorical'


def preprocess_event_log(input_path, output_csv_path="preprocessed_log.csv"):
    if input_path.endswith('.xes'):
        log = pm4py.read_xes(input_path)
        df = pm4py.convert_to_dataframe(log)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} events from {input_path}")
    print(f"Original columns: {list(df.columns)}")

    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
    if timestamp_cols:
        timestamp_col = timestamp_cols[0]
        print(f"Detected timestamp column: {timestamp_col}")
    else:
        print("Warning: No timestamp column detected")
        timestamp_col = None
    
    case_cols = [col for col in df.columns if 'case' in col.lower() and ('id' in col.lower() or 'name' in col.lower())]
    if case_cols:
        case_col = case_cols[0]
        print(f"Detected case ID column: {case_col}")
    else:
        print("Warning: No case ID column detected")
        case_col = None

    if case_col and timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df.sort_values(by=[case_col, timestamp_col], inplace=True)
        print("DataFrame sorted by CaseID and Timestamp.")

    if case_col and timestamp_col:
        df['Timestamp_Sec'] = df[timestamp_col].dt.floor('s')
        df['Prev_CaseID'] = df[case_col].shift(1)
        df['Prev_Timestamp_Sec'] = df['Timestamp_Sec'].shift(1)
        
        same_second_mask = (df[case_col] == df['Prev_CaseID']) & \
                           (df['Timestamp_Sec'] == df['Prev_Timestamp_Sec'])
        
        order_critical_mask = same_second_mask & (df[timestamp_col] != df[timestamp_col].shift(1))
        cases_affected_by_milliseconds = df[order_critical_mask][case_col].nunique()
        
        print(f"Total Cases in Log: {df[case_col].nunique()}")
        print(f"Total Unique Cases where Milliseconds Determine Order: {cases_affected_by_milliseconds}")
        

        df.drop(columns=['Timestamp_Sec', 'Prev_CaseID', 'Prev_Timestamp_Sec'], inplace=True)
        
        df[timestamp_col] = df[timestamp_col].dt.floor('s')
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
        df.sort_values(by=[case_col, timestamp_col], inplace=True)
    
    print("\n--- Checking Data Types and Handling Wrong Values ---")
    
    for col in df.columns:
        if col in [case_col, timestamp_col]:
            continue
        
        detected_type = detect_column_type(df[col])
        
        if detected_type == 'categorical':
            df[col] = df[col].ffill().bfill()
        
        elif detected_type == 'numerical':
            mask = df[col].isna()
            for idx in df[mask].index:
                prev_idx = idx - 1
                next_idx = idx + 1
                
                prev_val = df.loc[prev_idx, col] if prev_idx in df.index else np.nan
                next_val = df.loc[next_idx, col] if next_idx in df.index else np.nan
                
                if pd.notna(prev_val) and pd.notna(next_val):
                    df.loc[idx, col] = (prev_val + next_val) / 2
                elif pd.notna(prev_val):
                    df.loc[idx, col] = prev_val
                elif pd.notna(next_val):
                    df.loc[idx, col] = next_val

    print("\n--- Handling Null and Zero Values ---")
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        print(f"Columns with missing values: {missing_cols}")
        missing_before = df[missing_cols].isnull().sum()
        print(missing_before[missing_before > 0])
        numeric_cols = [col for col in missing_cols if detect_column_type(df[col]) == 'numerical']
        categorical_cols = [col for col in missing_cols if detect_column_type(df[col]) == 'categorical']
        if timestamp_col and timestamp_col in missing_cols:
            rows_before = len(df)
            df = df.dropna(subset=[timestamp_col])
            rows_after = len(df)
            if rows_before > rows_after:
                print(f"Removed {rows_before - rows_after} rows with missing timestamps")
            missing_cols.remove(timestamp_col)
            if timestamp_col in numeric_cols:
                numeric_cols.remove(timestamp_col)
            if timestamp_col in categorical_cols:
                categorical_cols.remove(timestamp_col)
        for col in numeric_cols:
            df[col] = df[col].fillna(0.0)
        for col in categorical_cols:
            df[col] = df[col].fillna('N/A')
        
        print("\n--- Missing Value Count AFTER Imputation ---")
        missing_after = df[missing_cols].isnull().sum()
        print(missing_after)
    else:
        print("No missing values found.")

    initial_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    
    print(f"\n Looking for duplicates]")
    print(f"Total rows before checking: {initial_rows}")
    print(f"Total exact duplicate rows found: {duplicate_rows}")
    
    if duplicate_rows > 0:
        df.drop_duplicates(inplace=True) 
        
        final_rows = len(df)
        print(f"Removed {duplicate_rows} duplicate rows.")
        print(f"Total rows after removal: {final_rows}")
    else:
        print("No exact duplicate rows found.")
    
    df.to_csv(output_csv_path, index=False)
    print(f"\nMain file saved at: {output_csv_path}")
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(df.head())
    print(df.info())
    
    return df


def main():
    input_file = "BPI_Models\BPI_logs_csv\BPI_2020_Log_PermitLog.csv"
    output_file = "BPI_Models/BPI_logs_preprocessed_csv/BPI_2020_Log_PermitLog_preprocessed_log.csv"
    
    df = preprocess_event_log(
        input_path=input_file,
        output_csv_path=output_file
    )
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()