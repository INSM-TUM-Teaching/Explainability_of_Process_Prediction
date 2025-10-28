import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

def load_event_log(xes_path: str):
    return xes_importer.apply(xes_path)

def log_to_dataframe(event_log):
    data = []
    for trace in event_log:
        cid = trace.attributes.get("concept:name", "")
        for event in trace:
            data.append({
                "case_id": cid,
                "activity": event.get("concept:name", ""),
                "timestamp": event.get("time:timestamp", ""),
                "resource": event.get("org:resource", ""),
                "lifecycle": event.get("lifecycle:transition", "")
            })
    return pd.DataFrame(data)

def main():
    xes_path = "BPI_Challenge_2017/BPI_Challenge_2017.xes"
    output_folder = "BPI_Models/BPI_logs"
    os.makedirs(output_folder, exist_ok=True)
    df = log_to_dataframe(load_event_log(xes_path))
    csv_path = os.path.join(output_folder, "bpi2017_log.csv")
    df.to_csv(csv_path, index=False)
    print("Log saved as CSV at:", csv_path)

if __name__ == "__main__":
    main()
