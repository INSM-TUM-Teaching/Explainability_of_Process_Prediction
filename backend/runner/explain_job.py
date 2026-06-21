import sys
import os
import json
import traceback

def run_explanation():
    try:
        run_id = os.environ.get("RUN_ID")
        case_id = os.environ.get("CASE_ID")
        case_index = int(os.environ.get("CASE_INDEX")) if os.environ.get("CASE_INDEX") else None
        method = os.environ.get("METHOD")
        
        # Load run details to know if it's GNN or Transformer, and the dataset path
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "runs", run_id)
        # TODO implement detail loading
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_explanation()
