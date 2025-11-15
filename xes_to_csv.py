
import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter


def load_event_log(xes_path: str):
    return xes_importer.apply(xes_path)


def log_to_dataframe_preserve_all(event_log):
    # Use PM4Py's built-in converter to preserve all attributes
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df


def log_to_dataframe_manual(event_log):
    data = []
    for trace in event_log:
        trace_attrs = dict(trace.attributes)
        
        for event in trace:
            row = trace_attrs.copy()
            row.update(dict(event))
            
            data.append(row)
    
    return pd.DataFrame(data)


def main():
    xes_path = r"BPI_Models\BPI_logs_csv\BPI_2017_Log_O_Offer.csv"
    output_folder = "BPI_Models\BPI_logs_preprocessed_csv"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the event log
    print(f"Loading XES file: {xes_path}")
    event_log = load_event_log(xes_path)
    df = log_to_dataframe_preserve_all(event_log)
    
    if df.empty or len(df.columns) < 3:
        df = log_to_dataframe_manual(event_log)
    
    # Generate output CSV filename based on input XES filename
    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Log saved as CSV at: {csv_path}")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
