import os
from typing import Tuple, List

import pandas as pd


def convert_xes_to_csv(xes_path: str, output_dir: str) -> Tuple[str, pd.DataFrame, List[str]]:
    """
    Convert a .xes file to CSV using pm4py.

    Returns: (csv_path, dataframe, columns)
    """
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.objects.conversion.log import converter as log_converter
    except Exception as e:
        raise RuntimeError(f"pm4py is required for XES conversion: {e}")

    event_log = xes_importer.apply(xes_path)

    try:
        df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    except Exception:
        # Fallback for some pm4py versions
        import pm4py

        df = pm4py.convert_to_dataframe(event_log)

    base_name = os.path.splitext(os.path.basename(xes_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    return csv_path, df, list(df.columns)

