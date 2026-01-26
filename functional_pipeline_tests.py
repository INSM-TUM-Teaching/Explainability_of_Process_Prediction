import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


REPORT_DIR = Path("tests/test_reports")


class FunctionalTestLogger:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.report_dir / f"functional_test_log_{timestamp}.txt"
        self.results: List[Dict[str, Any]] = []

        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("FUNCTIONAL TEST RUN\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

    def log(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def add_result(self, name: str, success: bool, details: Dict[str, Any]) -> None:
        self.results.append({
            "name": name,
            "success": success,
            "details": details,
        })

    def write_report(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"functional_test_report_{timestamp}.json"
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
            "results": self.results,
            "generated_at": datetime.now().isoformat(),
        }
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return report_path


def build_sample_dataset(path: Path, cases: int = 5, events_per_case: int = 6) -> None:
    rows = []
    base_time = pd.Timestamp("2024-01-01T08:00:00")
    for c in range(1, cases + 1):
        for e in range(events_per_case):
            rows.append({
                "CaseID": f"C{c}",
                "Activity": f"A{(e % 3) + 1}",
                "Timestamp": (base_time + pd.Timedelta(minutes=10 * (c + e))).isoformat(),
                "Resource": f"R{(c % 2) + 1}",
                "CustomCategory": "X" if e % 2 == 0 else "Y",
            })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def post_json(client: TestClient, url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    response = client.post(url, json=payload)
    try:
        body = response.json()
    except Exception:
        body = {}
    return response.status_code, body


def upload_dataset(client: TestClient, dataset_path: Path, preprocessed: bool = False) -> Dict[str, Any]:
    with dataset_path.open("rb") as f:
        files = {"file": (dataset_path.name, f, "text/csv")}
        response = client.post(f"/datasets/upload?preprocessed={'true' if preprocessed else 'false'}", files=files)
    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")
    return response.json()


def generate_splits(client: TestClient, dataset_id: str) -> Dict[str, Any]:
    status, body = post_json(client, f"/datasets/{dataset_id}/splits/generate", {
        "test_size": 0.1,
        "val_split": 0.1,
    })
    if status != 200:
        raise RuntimeError(f"Split generation failed: {status} {body}")
    return body


def preprocess_dataset(client: TestClient, dataset_id: str) -> Dict[str, Any]:
    status, body = post_json(client, f"/datasets/{dataset_id}/preprocess", {
        "sort_and_normalize_timestamps": True,
        "check_millisecond_order": True,
        "impute_categorical": True,
        "impute_numeric_neighbors": True,
        "drop_missing_timestamps": True,
        "fill_remaining_missing": True,
        "remove_duplicates": True,
    })
    if status != 200:
        raise RuntimeError(f"Preprocess failed: {status} {body}")
    return body


def start_run(
    client: TestClient,
    dataset_id: str,
    model_type: str,
    task: str,
) -> Dict[str, Any]:
    if model_type == "transformer":
        config = {
            "max_len": 8,
            "d_model": 32,
            "num_heads": 2,
            "num_blocks": 1,
            "dropout_rate": 0.1,
            "epochs": 1,
            "batch_size": 32,
            "patience": 1,
        }
    else:
        config = {
            "hidden": 32,
            "dropout_rate": 0.1,
            "lr": 0.0004,
            "epochs": 1,
            "batch_size": 16,
            "patience": 1,
        }

    payload = {
        "dataset_id": dataset_id,
        "model_type": model_type,
        "task": task,
        "config": config,
        "split": {"test_size": 0.1, "val_split": 0.1},
        "explainability": "none",
        "mapping_mode": "auto",
        "column_mapping": None,
        "target_column": None,
    }
    status, body = post_json(client, "/runs", payload)
    if status != 200:
        raise RuntimeError(f"Run creation failed: {status} {body}")
    return body


def poll_run(client: TestClient, run_id: str, timeout_s: int) -> Dict[str, Any]:
    start = time.time()
    last_status = None
    while time.time() - start < timeout_s:
        response = client.get(f"/runs/{run_id}")
        if response.status_code != 200:
            raise RuntimeError(f"Run status failed: {response.status_code} {response.text}")
        status = response.json()
        last_status = status
        if status.get("status") in {"succeeded", "failed"}:
            return status
        time.sleep(2)
    raise TimeoutError(f"Run did not finish within {timeout_s} seconds. Last status: {last_status}")


def download_artifacts_zip(client: TestClient, run_id: str, out_path: Path) -> None:
    response = client.get(f"/runs/{run_id}/artifacts.zip")
    if response.status_code != 200:
        raise RuntimeError(f"Artifacts zip failed: {response.status_code} {response.text}")
    out_path.write_bytes(response.content)


def fetch_logs(client: TestClient, run_id: str) -> List[str]:
    response = client.get(f"/runs/{run_id}/logs?tail=200")
    if response.status_code != 200:
        return []
    body = response.json()
    return body.get("lines", [])


def run_functional_pipeline(
    logger: FunctionalTestLogger,
    model_type: str,
    task: str,
    preprocess: bool,
    timeout_s: int,
) -> None:
    client = TestClient(app)
    run_label = f"{model_type}:{task}"
    details: Dict[str, Any] = {"model_type": model_type, "task": task}

    try:
        dataset_path = Path("tests/test_data") / f"sample_{model_type}.csv"
        build_sample_dataset(dataset_path)
        logger.log(f"[INFO] Uploading dataset for {run_label}")
        upload_resp = upload_dataset(client, dataset_path, preprocessed=False)
        dataset_id = upload_resp["dataset_id"]
        details["dataset_id"] = dataset_id

        if preprocess:
            logger.log(f"[INFO] Preprocessing dataset for {run_label}")
            preprocess_dataset(client, dataset_id)

        logger.log(f"[INFO] Generating splits for {run_label}")
        generate_splits(client, dataset_id)

        logger.log(f"[INFO] Starting run for {run_label}")
        run_resp = start_run(client, dataset_id, model_type, task)
        run_id = run_resp["run_id"]
        details["run_id"] = run_id

        logger.log(f"[INFO] Polling run {run_id}")
        final_status = poll_run(client, run_id, timeout_s)
        details["final_status"] = final_status

        if final_status.get("status") != "succeeded":
            details["logs"] = fetch_logs(client, run_id)
            raise RuntimeError(f"Run failed: {final_status}")

        artifacts_resp = client.get(f"/runs/{run_id}/artifacts")
        if artifacts_resp.status_code == 200:
            details["artifacts"] = artifacts_resp.json().get("artifacts", [])

        zip_path = REPORT_DIR / f"artifacts_{run_id}.zip"
        download_artifacts_zip(client, run_id, zip_path)
        details["artifacts_zip"] = str(zip_path)

        logger.add_result(f"pipeline_{run_label}", True, details)
        logger.log(f"[OK] Functional pipeline succeeded: {run_label}")
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc()
        logger.add_result(f"pipeline_{run_label}", False, details)
        logger.log(f"[X] Functional pipeline failed: {run_label} -> {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1200, help="Run timeout in seconds.")
    parser.add_argument("--preprocess", action="store_true", help="Enable preprocessing step.")
    parser.add_argument(
        "--model",
        choices=["transformer", "gnn", "all"],
        default="transformer",
        help="Which model to run in functional test.",
    )
    args = parser.parse_args()

    logger = FunctionalTestLogger(REPORT_DIR)
    logger.log("Starting functional pipeline tests...")

    if args.model in {"transformer", "all"}:
        run_functional_pipeline(
            logger,
            model_type="transformer",
            task="next_activity",
            preprocess=args.preprocess,
            timeout_s=args.timeout,
        )

    if args.model in {"gnn", "all"}:
        run_functional_pipeline(
            logger,
            model_type="gnn",
            task="next_activity",
            preprocess=args.preprocess,
            timeout_s=args.timeout,
        )

    report_path = logger.write_report()
    logger.log(f"Report written to {report_path}")

    failed = [r for r in logger.results if not r["success"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
