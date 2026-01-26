import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from main import (
    EXPLAINABILITY_AVAILABLE,
    PYTORCH_AVAILABLE,
    TENSORFLOW_AVAILABLE,
    detect_and_standardize_columns,
    process_input_file,
    run_event_time_prediction,
    run_gnn_unified_prediction,
    run_next_activity_prediction,
    run_remaining_time_prediction,
)


RAW_CSV_DIR = Path("BPI_Models/BPI_logs_csv")
RAW_XES_DIR = Path("BPI_Models/BPI_logs_xes")
PREPROCESSED_CSV_DIR = Path("BPI_Models/BPI_logs_preprocessed_csv")
REPORT_ROOT = Path("tests/test_reports")

TRANSFORMER_TASKS = ["next_activity", "event_time", "remaining_time"]
TRANSFORMER_EXPLAINABILITY = ["shap", "lime", "all"]

GNN_TASKS = ["next_activity", "event_time", "remaining_time"]
GNN_EXPLAINABILITY = ["gradient", "lime", "all"]

DEFAULT_TEST_SIZE = 0.3
DEFAULT_VAL_SPLIT = 0.5

DEFAULT_TRANSFORMER_CONFIG = {
    "max_len": 16,
    "d_model": 64,
    "num_heads": 4,
    "num_blocks": 2,
    "dropout_rate": 0.1,
    "epochs": 5,
    "batch_size": 128,
    "patience": 10,
}

DEFAULT_GNN_CONFIG = {
    "hidden": 64,
    "dropout_rate": 0.1,
    "lr": 4e-4,
    "epochs": 5,
    "batch_size": 64,
    "patience": 10,
}


class FunctionalPredictionTester:
    def __init__(self, report_root: Path) -> None:
        self.report_root = report_root
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.prepared_root = self.report_root / "prepared_datasets"
        self.prepared_root.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_log = self.report_root / f"functional_prediction_log_{timestamp}.txt"
        with self.master_log.open("w", encoding="utf-8") as f:
            f.write("FUNCTIONAL PREDICTION TEST RUN\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

    def log(self, message: str) -> None:
        print(message)
        with self.master_log.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def collect_datasets(
        self,
        raw_csv_dir: Path,
        raw_xes_dir: Path,
        preprocessed_dir: Path,
    ) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []

        def add_files(source_dir: Path, ext: str, source_type: str) -> None:
            if not source_dir.exists():
                self.log(f"[WARN] Missing directory: {source_dir}")
                return
            files = sorted(source_dir.glob(f"*.{ext}"))
            if not files:
                self.log(f"[WARN] No {ext.upper()} files in: {source_dir}")
                return
            for file_path in files:
                datasets.append({"path": file_path, "source_type": source_type})

        add_files(raw_csv_dir, "csv", "raw_csv")
        add_files(raw_xes_dir, "xes", "raw_xes")
        add_files(preprocessed_dir, "csv", "preprocessed_csv")

        return datasets

    def prepare_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        source_path: Path = dataset["path"]
        source_type: str = dataset["source_type"]

        if source_type == "raw_xes":
            preprocessed_path = Path(process_input_file(str(source_path), "xes"))
        elif source_type == "raw_csv":
            preprocessed_path = Path(process_input_file(str(source_path), "csv"))
        else:
            preprocessed_path = source_path

        df = pd.read_csv(preprocessed_path)
        df_mapped, column_mapping, _ = detect_and_standardize_columns(df, verbose=False)

        safe_name = source_path.stem.replace(" ", "_")
        prep_dir = self.prepared_root / f"{safe_name}_{source_type}"
        prep_dir.mkdir(parents=True, exist_ok=True)
        final_path = prep_dir / f"{safe_name}_final.csv"
        df_mapped.to_csv(final_path, index=False)

        return {
            "source_path": str(source_path),
            "source_type": source_type,
            "preprocessed_path": str(preprocessed_path),
            "final_path": str(final_path),
            "column_mapping": column_mapping,
        }

    def run_single_test(
        self,
        test_number: int,
        total_tests: int,
        dataset_info: Dict[str, Any],
        model_type: str,
        task: str,
        explainability: Optional[str],
    ) -> None:
        start_time = time.time()
        dataset_name = Path(dataset_info["final_path"]).stem
        exp_label = explainability or "none"
        run_label = f"{dataset_name} | {model_type} | {task} | explainability={exp_label}"

        self.log("\n" + "=" * 80)
        self.log(f"TEST {test_number}/{total_tests}: {run_label}")
        self.log("=" * 80)

        safe_label = run_label.replace(" ", "_").replace("|", "-").replace(":", "_")
        run_dir = self.report_root / f"run_{test_number:04d}_{safe_label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "test_number": test_number,
                    "dataset": dataset_info,
                    "model_type": model_type,
                    "task": task,
                    "explainability": explainability,
                    "test_size": DEFAULT_TEST_SIZE,
                    "val_split": DEFAULT_VAL_SPLIT,
                    "transformer_config": DEFAULT_TRANSFORMER_CONFIG,
                    "gnn_config": DEFAULT_GNN_CONFIG,
                    "explainability_available": EXPLAINABILITY_AVAILABLE,
                    "tensorflow_available": TENSORFLOW_AVAILABLE,
                    "pytorch_available": PYTORCH_AVAILABLE,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        error_log_path = run_dir / "ERROR_LOG.txt"
        success = False
        error_message = None

        try:
            if explainability and not EXPLAINABILITY_AVAILABLE:
                raise RuntimeError("Explainability modules not available.")

            if model_type == "transformer" and not TENSORFLOW_AVAILABLE:
                raise RuntimeError("TensorFlow not available for transformer tests.")

            if model_type == "gnn" and not PYTORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available for GNN tests.")

            with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
                with redirect_stdout(out_f), redirect_stderr(err_f):
                    if model_type == "transformer":
                        if task == "next_activity":
                            run_next_activity_prediction(
                                dataset_info["final_path"],
                                str(output_dir),
                                DEFAULT_TEST_SIZE,
                                DEFAULT_VAL_SPLIT,
                                DEFAULT_TRANSFORMER_CONFIG,
                                explainability,
                            )
                        elif task == "event_time":
                            run_event_time_prediction(
                                dataset_info["final_path"],
                                str(output_dir),
                                DEFAULT_TEST_SIZE,
                                DEFAULT_VAL_SPLIT,
                                DEFAULT_TRANSFORMER_CONFIG,
                                explainability,
                            )
                        elif task == "remaining_time":
                            run_remaining_time_prediction(
                                dataset_info["final_path"],
                                str(output_dir),
                                DEFAULT_TEST_SIZE,
                                DEFAULT_VAL_SPLIT,
                                DEFAULT_TRANSFORMER_CONFIG,
                                explainability,
                            )
                        else:
                            raise ValueError(f"Unknown transformer task: {task}")
                    elif model_type == "gnn":
                        run_gnn_unified_prediction(
                            dataset_info["final_path"],
                            str(output_dir),
                            DEFAULT_TEST_SIZE,
                            DEFAULT_VAL_SPLIT,
                            DEFAULT_GNN_CONFIG,
                            explainability,
                            task=task,
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")

            success = True
            duration = time.time() - start_time
            (run_dir / "SUCCESS.txt").write_text(
                f"Test passed in {duration:.1f}s\n",
                encoding="utf-8",
            )
            self.log(f"[OK] TEST PASSED ({duration:.1f}s)")
        except Exception as exc:
            duration = time.time() - start_time
            error_message = str(exc)
            self.log(f"[X] TEST FAILED ({duration:.1f}s): {exc}")
            error_log_path.write_text(
                "\n".join(
                    [
                        "=" * 80,
                        "ERROR LOG",
                        "=" * 80,
                        f"Test Number: {test_number}",
                        f"Dataset: {dataset_name}",
                        f"Model Type: {model_type}",
                        f"Task: {task}",
                        f"Explainability: {exp_label}",
                        f"Duration: {duration:.1f}s",
                        f"Error: {exc}",
                        "",
                        "Traceback:",
                        traceback.format_exc(),
                    ]
                ),
                encoding="utf-8",
            )

        self.results.append(
            {
                "test_number": test_number,
                "dataset": dataset_name,
                "model_type": model_type,
                "task": task,
                "explainability": exp_label,
                "success": success,
                "duration_s": time.time() - start_time,
                "run_dir": str(run_dir),
                "error": error_message,
            }
        )

    def finalize(self) -> None:
        summary_path = self.report_root / "functional_prediction_summary.json"
        passed = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - passed
        summary_payload = {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(self.results) * 100) if self.results else 0.0,
            "generated_at": datetime.now().isoformat(),
            "results": self.results,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        self.log(f"[OK] Summary written to {summary_path}")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = REPORT_ROOT / f"functional_prediction_{timestamp}"
    tester = FunctionalPredictionTester(report_root)

    tester.log("Starting prediction + explainability functional tests...")
    tester.log(f"Transformer tasks: {TRANSFORMER_TASKS}")
    tester.log(f"Transformer explainability: {TRANSFORMER_EXPLAINABILITY}")
    tester.log(f"GNN tasks: {GNN_TASKS}")
    tester.log(f"GNN explainability: {GNN_EXPLAINABILITY}")

    datasets = tester.collect_datasets(RAW_CSV_DIR, RAW_XES_DIR, PREPROCESSED_CSV_DIR)
    if not datasets:
        tester.log("[X] No datasets found. Exiting.")
        return 1

    tester.log(f"Found {len(datasets)} dataset files.")
    prepared_datasets: List[Dict[str, Any]] = []

    for dataset in datasets:
        try:
            prepared = tester.prepare_dataset(dataset)
            prepared_datasets.append(prepared)
            tester.log(f"[OK] Prepared: {dataset['path']}")
        except Exception as exc:
            tester.log(f"[X] Failed to prepare {dataset['path']}: {exc}")
            tester.results.append(
                {
                    "test_number": None,
                    "dataset": str(dataset["path"]),
                    "model_type": None,
                    "task": None,
                    "explainability": None,
                    "success": False,
                    "duration_s": 0,
                    "run_dir": None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    total_tests = len(prepared_datasets) * (
        len(TRANSFORMER_TASKS) * len(TRANSFORMER_EXPLAINABILITY)
        + len(GNN_TASKS) * len(GNN_EXPLAINABILITY)
    )
    tester.log(f"Total tests to run: {total_tests}")

    test_number = 1
    for dataset_info in prepared_datasets:
        for task in TRANSFORMER_TASKS:
            for exp in TRANSFORMER_EXPLAINABILITY:
                tester.run_single_test(
                    test_number,
                    total_tests,
                    dataset_info,
                    model_type="transformer",
                    task=task,
                    explainability=exp,
                )
                test_number += 1
        for task in GNN_TASKS:
            for exp in GNN_EXPLAINABILITY:
                tester.run_single_test(
                    test_number,
                    total_tests,
                    dataset_info,
                    model_type="gnn",
                    task=task,
                    explainability=exp,
                )
                test_number += 1

    tester.finalize()
    failed = [r for r in tester.results if not r["success"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
