# backend/runner/run_job.py
import argparse
import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def patch_status(status_path: str, **patch) -> None:
    status: Dict[str, Any] = {}
    if os.path.exists(status_path):
        status = read_json(status_path)
    status.update(patch)
    status["updated_at"] = utc_now()
    write_json(status_path, status)


def list_artifacts(artifacts_dir: str) -> list[str]:
    out: list[str] = []
    if not os.path.exists(artifacts_dir):
        return out
    for root, _, files in os.walk(artifacts_dir):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, artifacts_dir)
            out.append(rel)
    out.sort()
    return out


def normalize_explainability(value: Any) -> Any:
    # Frontend/Swagger often sends "none"
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    request_path = os.path.join(run_dir, "request.json")
    status_path = os.path.join(run_dir, "status.json")
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    req = read_json(request_path)
    run_id = req["run_id"]
    dataset_id = req["dataset_id"]

    # Resolve dataset path from registry
    datasets_meta_path = os.path.abspath(
        os.path.join(run_dir, "..", "..", "datasets", dataset_id, "meta.json")
    )
    if not os.path.exists(datasets_meta_path):
        patch_status(
            status_path,
            status="failed",
            finished_at=utc_now(),
            error=f"Dataset meta not found: {datasets_meta_path}",
        )
        raise RuntimeError(f"Dataset meta not found: {datasets_meta_path}")

    dataset_meta = read_json(datasets_meta_path)
    dataset_path = dataset_meta["stored_path"]

    model_type = (req.get("model_type") or "").lower().strip()
    task = (req.get("task") or "").lower().strip()

    split = req.get("split") or {"test_size": 0.2, "val_split": 0.5}
    test_size = float(split.get("test_size", 0.2))
    val_split = float(split.get("val_split", 0.5))

    config = req.get("config") or {}
    explainability = normalize_explainability(req.get("explainability", None))

    # Mark running BEFORE any heavy imports
    patch_status(status_path, status="running", started_at=utc_now(), error=None)

    # Write early summary so artifacts are never empty while running
    write_json(os.path.join(artifacts_dir, "summary.json"), {
        "run_id": run_id,
        "status": "running",
        "dataset": {
            "dataset_id": dataset_id,
            "filename": os.path.basename(dataset_path),
            "num_events": dataset_meta.get("num_events"),
            "num_cases": dataset_meta.get("num_cases"),
        },
        "request": req,
        "artifacts": list_artifacts(artifacts_dir),
        "started_at": utc_now(),
    })

    try:
        # Lazy import (prevents "queued forever" on import failure)
        from ppm_pipeline import (
            default_gnn_config,
            default_transformer_config,
            run_next_activity_prediction,
            run_event_time_prediction,
            run_remaining_time_prediction,
            run_gnn_unified_prediction,
        )

        if model_type == "transformer" and not config:
            config = default_transformer_config()
        if model_type == "gnn" and not config:
            config = default_gnn_config()

        if model_type == "transformer":
            if task == "next_activity":
                metrics = run_next_activity_prediction(
                    dataset_path, artifacts_dir, test_size, val_split, config,
                    explainability_method=explainability
                )
            elif task == "event_time":
                metrics = run_event_time_prediction(
                    dataset_path, artifacts_dir, test_size, val_split, config,
                    explainability_method=explainability
                )
            elif task == "remaining_time":
                metrics = run_remaining_time_prediction(
                    dataset_path, artifacts_dir, test_size, val_split, config,
                    explainability_method=explainability
                )
            else:
                raise RuntimeError(f"Unsupported transformer task: {task}")

        elif model_type == "gnn":
            if task not in {"next_activity", "event_time", "remaining_time", "unified"}:
                raise RuntimeError(f"Unsupported gnn task: {task}")

            metrics = run_gnn_unified_prediction(
                dataset_path,
                artifacts_dir,
                test_size,
                val_split,
                config,
                explainability_method=explainability,
                task=task,
            )
        else:
            raise RuntimeError(f"Unsupported model_type: {model_type}")

        # metrics.json for frontend rendering
        write_json(os.path.join(artifacts_dir, "metrics.json"), {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "model_type": model_type,
            "task": task,
            "split": {"test_size": test_size, "val_split": val_split},
            "config": config,
            "metrics": metrics,
            "finished_at": utc_now(),
        })

        # final summary.json
        write_json(os.path.join(artifacts_dir, "summary.json"), {
            "run_id": run_id,
            "status": "succeeded",
            "dataset": {
                "dataset_id": dataset_id,
                "filename": os.path.basename(dataset_path),
                "num_events": dataset_meta.get("num_events"),
                "num_cases": dataset_meta.get("num_cases"),
            },
            "request": req,
            "metrics": metrics,
            "artifacts": list_artifacts(artifacts_dir),
            "finished_at": utc_now(),
        })

        patch_status(status_path, status="succeeded", finished_at=utc_now(), error=None)

    except Exception as e:
        tb = traceback.format_exc()
        write_json(os.path.join(artifacts_dir, "error.json"), {"error": str(e), "traceback": tb})
        write_json(os.path.join(artifacts_dir, "summary.json"), {
            "run_id": run_id,
            "status": "failed",
            "dataset": {
                "dataset_id": dataset_id,
                "filename": os.path.basename(dataset_path),
                "num_events": dataset_meta.get("num_events"),
                "num_cases": dataset_meta.get("num_cases"),
            },
            "request": req,
            "artifacts": list_artifacts(artifacts_dir),
            "error": str(e),
            "finished_at": utc_now(),
        })
        patch_status(status_path, status="failed", finished_at=utc_now(), error=str(e))
        raise


if __name__ == "__main__":
    main()
