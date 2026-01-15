# backend/main.py
import os
import sys
import uuid
import json
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="PPM Backend", version="0.1.0")

# Allow the Vite frontend to call the API during development
app.add_middleware(
    CORSMiddleware,
    # Allow localhost dev servers on any port (Vite commonly uses 5173/5174, etc.)
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Paths / Storage
# -----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))            # .../repo/backend
REPO_ROOT = os.path.dirname(BACKEND_DIR)                            # .../repo
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # allow importing project-level modules

# Import preprocessing utilities
try:
    from conv_and_viz.preprocessor_csv import preprocess_event_log
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("[WARNING] Preprocessor not available - skipping data cleaning")

STORAGE_DIR = os.path.join(BACKEND_DIR, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
RUNS_DIR = os.path.join(STORAGE_DIR, "runs")

for d in (STORAGE_DIR, UPLOAD_DIR, DATASETS_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)

MAX_UPLOAD_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Use the same Python interpreter that runs uvicorn (your backend/.venv)
PYTHON_EXEC = sys.executable

# -----------------------------------------------------------------------------
# Small JSON helpers (atomic write)
# -----------------------------------------------------------------------------
def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# -----------------------------------------------------------------------------
# Column detection / standardization
# -----------------------------------------------------------------------------
def detect_and_standardize_columns(
    df: pd.DataFrame, verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detect typical event-log columns and standardize them to:
      - CaseID
      - Activity
      - Timestamp
      - Resource (optional)

    Returns:
      (df_renamed, mapping_old_to_new)
    """
    column_mapping: Dict[str, str] = {}

    case_patterns = [
        "case:id", "case:concept:name", "CaseID", "case_id", "caseid", "Case ID", "Case_ID"
    ]
    activity_patterns = [
        "concept:name", "Action", "activity", "event", "Event", "task", "Task", "Activity"
    ]
    timestamp_patterns = [
        "time:timestamp", "Timestamp", "timestamp", "time", "Time", "start_time",
        "StartTime", "complete_time", "CompleteTime"
    ]
    resource_patterns = [
        "org:resource", "Resource", "resource", "user", "User", "org:role",
        "role", "Role", "actor", "Actor"
    ]

    # Case
    for col in df.columns:
        if col in case_patterns and col != "CaseID":
            column_mapping[col] = "CaseID"
            break
    if "CaseID" in df.columns and "CaseID" not in column_mapping.values():
        # already ok
        pass

    # Activity
    for col in df.columns:
        if col in activity_patterns and col != "Activity":
            column_mapping[col] = "Activity"
            break

    # Timestamp
    for col in df.columns:
        if col in timestamp_patterns and col != "Timestamp":
            column_mapping[col] = "Timestamp"
            break

    # Resource (optional)
    for col in df.columns:
        if col in resource_patterns and col != "Resource":
            column_mapping[col] = "Resource"
            break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ["CaseID", "Activity", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after detection: {missing}")

    # Make timestamp parseable (do not fail hard; just best effort)
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    except Exception:
        pass

    if verbose:
        print("COLUMN DETECTION:", column_mapping)

    return df, column_mapping


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class DatasetUploadResponse(BaseModel):
    dataset_id: str
    stored_path: str
    num_events: int
    num_cases: int
    columns: List[str]
    detected_mapping: Dict[str, str]
    preview: List[Dict[str, Any]]


class DatasetMeta(BaseModel):
    dataset_id: str
    stored_path: str
    num_events: int
    num_cases: int
    columns: List[str]
    detected_mapping: Dict[str, str]
    created_at: str


class ColumnMapping(BaseModel):
    # Column names in the uploaded dataset
    case_id: str
    activity: str
    timestamp: str
    resource: Optional[str] = None


class RunCreateRequest(BaseModel):
    dataset_id: str
    model_type: str = Field(..., description="transformer | gnn")
    task: str = Field(..., description="next_activity | event_time | remaining_time | unified (gnn)")
    config: Dict[str, Any] = Field(default_factory=dict)
    split: Dict[str, float] = Field(default_factory=lambda: {"test_size": 0.2, "val_split": 0.5})
    explainability: Optional[Any] = None
    mapping_mode: Optional[str] = Field(
        default=None, description="auto | manual (optional; defaults to auto)"
    )
    column_mapping: Optional[ColumnMapping] = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def _dataset_dir(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_id)


def _dataset_meta_path(dataset_id: str) -> str:
    return os.path.join(_dataset_dir(dataset_id), "meta.json")


def _dataset_file_path(dataset_id: str) -> str:
    return os.path.join(_dataset_dir(dataset_id), "dataset.csv")


def _load_dataset_meta(dataset_id: str) -> DatasetMeta:
    meta_path = _dataset_meta_path(dataset_id)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetMeta(**_read_json(meta_path))


# -----------------------------------------------------------------------------
# Run helpers
# -----------------------------------------------------------------------------
def _run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def _run_status_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "status.json")


def _run_request_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "request.json")


def _run_artifacts_dir(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "artifacts")


def _load_run_status(run_id: str) -> Dict[str, Any]:
    status_path = _run_status_path(run_id)
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail="Run not found")
    return _read_json(status_path)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "ppm-backend"}


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV or XES dataset. Stores it on disk, converts XES→CSV when needed,
    parses it, detects column mapping, and writes metadata to backend/storage/datasets/<dataset_id>/meta.json
    """
    filename = file.filename or "dataset.csv"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in {"csv", "xes"}:
        raise HTTPException(status_code=400, detail="Only CSV or XES files are supported.")

    dataset_id = str(uuid.uuid4())
    ds_dir = _dataset_dir(dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    stored_path = _dataset_file_path(dataset_id)  # final normalized CSV path
    raw_path = stored_path if ext == "csv" else os.path.join(ds_dir, "dataset.xes")

    # Save stream to disk with size enforcement
    size = 0
    try:
        with open(raw_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
                    )
                out.write(chunk)
    finally:
        await file.close()

    # Parse / convert
    try:
        if ext == "csv":
            df = pd.read_csv(stored_path)
        else:
            try:
                from conv_and_viz.xes_to_csv import convert_xes_to_csv
            except ImportError:
                shutil.rmtree(ds_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=500,
                    detail="XES support requires pm4py; install backend dependencies.",
                )

            try:
                csv_path, df, _ = convert_xes_to_csv(raw_path, ds_dir)
            except Exception as e:
                shutil.rmtree(ds_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail=f"Failed to convert XES: {str(e)}")

            # Normalize to dataset.csv for downstream code
            if os.path.abspath(csv_path) != os.path.abspath(stored_path):
                shutil.copyfile(csv_path, stored_path)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")

    # Preprocess the CSV (clean, deduplicate, handle missing values)
    if PREPROCESSOR_AVAILABLE:
        try:
            print(f"[Preprocessing] Cleaning dataset: {stored_path}")
            df = preprocess_event_log(stored_path, stored_path)
            print(f"[Preprocessing] Complete. Events: {len(df):,}")
        except Exception as e:
            print(f"[WARNING] Preprocessing failed: {e}")
            print("[WARNING] Continuing with raw CSV...")
            # If preprocessing fails, reload the raw CSV
            df = pd.read_csv(stored_path)
    else:
        print("[WARNING] Preprocessor not available - skipping data cleaning")
        df = pd.read_csv(stored_path)

    # Save preprocessed CSV with ORIGINAL column names (no auto-detection)
    # Column mapping will happen later when user selects Auto-Detect or Manual
    try:
        df.to_csv(stored_path, index=False)
    except Exception as e:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to write preprocessed dataset: {str(e)}")

    num_events = int(len(df))
    
    # Try to detect CaseID for num_cases, but don't rename columns
    case_col = None
    case_patterns = ['case:id', 'case:concept:name', 'CaseID', 'case_id', 'caseid', 'Case ID', 'Case_ID']
    for col in df.columns:
        if col in case_patterns:
            case_col = col
            break
    
    if case_col:
        num_cases = int(df[case_col].nunique())
    else:
        num_cases = 0  # Unknown until column mapping

    preview_rows = df.head(20).to_dict(orient="records")

    meta = DatasetMeta(
        dataset_id=dataset_id,
        stored_path=stored_path,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        detected_mapping={},  # No auto-detection - user will choose later
        created_at=_utc_now(),
    )
    _write_json(_dataset_meta_path(dataset_id), meta.model_dump())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        stored_path=stored_path,
        num_events=num_events,
        num_cases=num_cases,
        columns=list(df.columns),
        detected_mapping={},  # No auto-detection - user will choose later
        preview=preview_rows,
    )


@app.get("/datasets/{dataset_id}", response_model=DatasetMeta)
def get_dataset(dataset_id: str):
    """
    Fetch dataset metadata from the registry.
    """
    return _load_dataset_meta(dataset_id)


@app.post("/runs", response_model=RunCreateResponse)
def create_run(req: RunCreateRequest):
    """
    Create a training run. Uses Option 2: spawns a subprocess job.
    Returns immediately with queued status. Poll /runs/{run_id}.
    """
    # Validate dataset exists
    _ = _load_dataset_meta(req.dataset_id)

    run_id = str(uuid.uuid4())
    rdir = _run_dir(run_id)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(_run_artifacts_dir(run_id), exist_ok=True)

    # Write request.json for the runner
    request_obj = {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "model_type": req.model_type,
        "task": req.task,
        "config": req.config,
        "split": req.split,
        "explainability": req.explainability,
        "mapping_mode": req.mapping_mode,
        "column_mapping": req.column_mapping.model_dump() if req.column_mapping else None,
        "created_at": _utc_now(),
    }
    _write_json(_run_request_path(run_id), request_obj)

    # Initialize status.json
    status_obj = {
        "run_id": run_id,
        "status": "queued",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_json(_run_status_path(run_id), status_obj)

    # Spawn subprocess job (logs to logs.txt)
    log_path = os.path.join(rdir, "logs.txt")
    with open(log_path, "a", encoding="utf-8") as log:
        # Important: run module with repo root as cwd so "backend.*" imports resolve
        proc = subprocess.Popen(
            [PYTHON_EXEC, "-m", "backend.runner.run_job", "--run-dir", rdir],
            stdout=log,
            stderr=log,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    # Save pid
    status_obj["pid"] = proc.pid
    status_obj["updated_at"] = _utc_now()
    _write_json(_run_status_path(run_id), status_obj)

    return RunCreateResponse(run_id=run_id, status="queued")


@app.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str):
    """
    Poll run status.
    """
    status = _load_run_status(run_id)
    return RunStatus(**status)


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str):
    """
    Lists artifacts generated by the run.
    """
    rdir = _run_dir(run_id)
    if not os.path.exists(rdir):
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts_dir = _run_artifacts_dir(run_id)
    if not os.path.exists(artifacts_dir):
        return {"run_id": run_id, "artifacts": []}

    artifacts = []
    for root, _, files in os.walk(artifacts_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, artifacts_dir)
            artifacts.append(rel)

    artifacts.sort()
    return {"run_id": run_id, "artifacts": artifacts}


@app.get("/runs/{run_id}/artifacts/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str):
    """
    Download/view a single artifact file.
    """
    artifacts_dir = _run_artifacts_dir(run_id)
    full = os.path.normpath(os.path.join(artifacts_dir, artifact_path))

    # Prevent path traversal
    if not full.startswith(os.path.abspath(artifacts_dir) + os.sep) and os.path.abspath(full) != os.path.abspath(artifacts_dir):
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(full)