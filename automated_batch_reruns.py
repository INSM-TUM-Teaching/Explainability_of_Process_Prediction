import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from contextlib import contextmanager

from ppm_pipeline import (
    default_transformer_config,
    default_gnn_config,
    run_next_activity_prediction,
    run_event_time_prediction,
    run_remaining_time_prediction,
    run_gnn_unified_prediction,
)

DEFAULT_RESULTS_DIR = "automated_batch_reruns"
DEFAULT_DATASET_DIR = os.path.join("BPI_dataset", "BPI_logs_preprocessed_csv")
DEFAULT_SUMMARY_PATH = os.path.join("automated_batch_results", "batch_summary.json")
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SPLIT = 1.0 / 9.0


def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def _capture_output(path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(path, "w", encoding="utf-8") as f:
        sys.stdout = _TeeStream(original_stdout, f)
        sys.stderr = _TeeStream(original_stderr, f)
        try:
            yield f
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _normalize_run_ids(values):
    if not values:
        return None
    tokens = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    run_ids = []
    for token in tokens:
        try:
            run_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid run id: {token}") from exc
    return sorted(set(run_ids))


def _dir_has_png(path):
    if not os.path.isdir(path):
        return False
    return any(name.lower().endswith(".png") for name in os.listdir(path))


def check_required_images(run_dir, model, task):
    missing = []

    if model == "transformer":
        if task == "next_activity":
            required_files = [
                "next_activity_training_history.png",
            ]
        elif task == "event_time":
            required_files = [
                "event_time_predictions_plot.png",
                "event_time_training_history.png",
            ]
        else:
            required_files = [
                "remaining_time_predictions_plot.png",
                "remaining_time_training_history.png",
            ]

        for rel in required_files:
            if not os.path.exists(os.path.join(run_dir, rel)):
                missing.append(rel)

        shap_dir = os.path.join(run_dir, "explainability", "shap")
        if not _dir_has_png(shap_dir):
            missing.append("explainability/shap/*.png")

        lime_dir = os.path.join(run_dir, "explainability", "lime")
        if not _dir_has_png(lime_dir):
            missing.append("explainability/lime/*.png")

    else:
        if not os.path.exists(os.path.join(run_dir, "gnn_training_history.png")):
            missing.append("gnn_training_history.png")

        grad_dir = os.path.join(run_dir, "explainability", "gradient")
        if not _dir_has_png(grad_dir):
            missing.append("explainability/gradient/*.png")

        temporal_dir = os.path.join(run_dir, "explainability", "temporal")
        if not _dir_has_png(temporal_dir):
            missing.append("explainability/temporal/*.png")

        lime_dir = os.path.join(run_dir, "explainability", "graphlime")
        if not _dir_has_png(lime_dir):
            missing.append("explainability/graphlime/*.png")

    return missing


class RerunBatchRunner:
    def __init__(self, dataset_dir, output_dir, summary_path):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.summary_path = summary_path
        self.master_log = os.path.join(output_dir, f"rerun_master_log_{_now_tag()}.txt")
        self.universal_error_log = os.path.join(output_dir, "universal_error_log.txt")
        self.results = []
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_master_header()

    def _write_master_header(self):
        with open(self.master_log, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOMATED BATCH RERUN LOG\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message):
        print(message)
        with open(self.master_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_universal_error(self, message):
        with open(self.universal_error_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def load_failed_runs(self, run_ids=None):
        if not os.path.exists(self.summary_path):
            raise FileNotFoundError(f"Summary not found: {self.summary_path}")
        with open(self.summary_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        if run_ids:
            selected = [row for row in rows if row.get("run") in run_ids]
            if not selected:
                self.log("[X] No matching runs found for the requested run IDs.")
            else:
                self.log(f"Selected {len(selected)} runs by ID: {', '.join(map(str, run_ids))}")
            return selected

        failed = [row for row in rows if not row.get("success", False)]
        if not failed:
            self.log("[OK] No failed runs found in summary.")
        else:
            self.log(f"Found {len(failed)} failed runs to rerun.")
        return failed

    def build_run_specs(self, failed_rows):
        specs = []
        for row in failed_rows:
            dataset_name = row.get("dataset", "")
            if dataset_name.endswith(".csv"):
                dataset_file = dataset_name
            else:
                dataset_file = f"{dataset_name}.csv"

            dataset_path = os.path.join(self.dataset_dir, dataset_file)
            if not os.path.exists(dataset_path):
                self.log(f"[X] Dataset not found, skipping: {dataset_path}")
                continue

            model = row.get("model", "").lower()
            task = row.get("task", "").lower()

            if model == "transformer":
                runner = {
                    "next_activity": run_next_activity_prediction,
                    "event_time": run_event_time_prediction,
                    "remaining_time": run_remaining_time_prediction,
                }.get(task)
            else:
                runner = run_gnn_unified_prediction

            if runner is None:
                self.log(f"[X] Unsupported model/task, skipping: {dataset_name} | {model} | {task}")
                continue

            specs.append({
                "model": model,
                "task": task,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "runner": runner,
                "source_run": row.get("run"),
            })

        return specs

    def run_all(self, run_ids=None):
        failed_rows = self.load_failed_runs(run_ids)
        specs = self.build_run_specs(failed_rows)
        if not specs:
            self.log("[OK] No runnable failed runs found.")
            return

        self.log("\n" + "=" * 80)
        self.log("STARTING FAILED RUN RERUNS")
        self.log("=" * 80)

        for run_index, spec in enumerate(specs, 1):
            self.run_single(spec, run_index, len(specs))

        summary_path = os.path.join(self.output_dir, "rerun_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"\n[OK] Rerun summary saved: {summary_path}")

    def run_single(self, spec, run_index, total_runs):
        start = time.time()
        dataset_name = spec["dataset_name"]
        model = spec["model"]
        task = spec["task"]
        source_run = spec.get("source_run")
        description = f"{dataset_name} | {model.upper()} | {task}"

        safe_name = description.replace(" ", "_").replace("|", "-")
        run_dir = os.path.join(self.output_dir, f"rerun_{run_index:03d}_{safe_name}")
        os.makedirs(run_dir, exist_ok=True)

        self.log("\n" + "-" * 80)
        self.log(f"RERUN {run_index}/{total_runs}: {description}")
        self.log("-" * 80)

        config = default_transformer_config() if model == "transformer" else default_gnn_config()
        explainability = "all"
        test_size = DEFAULT_TEST_SIZE
        val_split = DEFAULT_VAL_SPLIT

        result = {
            "rerun": run_index,
            "source_run": source_run,
            "dataset": dataset_name,
            "model": model,
            "task": task,
            "output_dir": run_dir,
            "success": False,
            "duration_s": None,
            "missing_images": [],
            "metrics": {},
        }

        error_capture_path = os.path.join(run_dir, "error.txt")
        with _capture_output(error_capture_path) as error_log:
            error_log.write("=" * 80 + "\n")
            error_log.write("RERUN OUTPUT CAPTURE\n")
            error_log.write("=" * 80 + "\n\n")
            error_log.write(f"Rerun: {run_index}/{total_runs}\n")
            if source_run is not None:
                error_log.write(f"Source run: {source_run}\n")
            error_log.write(f"Description: {description}\n")
            error_log.write(f"Dataset: {spec['dataset_path']}\n")
            error_log.write(f"Model: {model}\n")
            error_log.write(f"Task: {task}\n\n")
            error_log.flush()

            try:
                if model == "transformer":
                    metrics = spec["runner"](
                        spec["dataset_path"],
                        run_dir,
                        test_size,
                        val_split,
                        config,
                        explainability_method=explainability,
                    )
                else:
                    metrics = spec["runner"](
                        spec["dataset_path"],
                        run_dir,
                        test_size,
                        val_split,
                        config,
                        explainability_method=explainability,
                        task=task,
                    )

                missing = check_required_images(run_dir, model=model, task=task)
                result["missing_images"] = missing
                if missing:
                    missing_path = os.path.join(run_dir, "MISSING_IMAGES.txt")
                    with open(missing_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(missing))
                    self.log(f"[X] Missing images ({len(missing)}). See: {missing_path}")
                else:
                    self.log("[OK] All required images found.")

                result["success"] = len(missing) == 0
                result["metrics"] = metrics or {}

            except Exception as exc:
                err_path = os.path.join(run_dir, "ERROR_LOG.txt")
                err_msg = f"[ERROR] {description} failed: {exc}"
                self.log(err_msg)
                self.log_universal_error(err_msg)
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("RERUN ERROR\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Rerun: {run_index}/{total_runs}\n")
                    if source_run is not None:
                        f.write(f"Source run: {source_run}\n")
                    f.write(f"Description: {description}\n")
                    f.write(f"Dataset: {spec['dataset_path']}\n")
                    f.write(f"Model: {model}\n")
                    f.write(f"Task: {task}\n\n")
                    f.write("Exception:\n")
                    f.write(str(exc) + "\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())

                traceback.print_exc()
                self.log(f"[X] Error log saved: {err_path}")

        result["duration_s"] = round(time.time() - start, 2)
        self.results.append(result)


def main():
    parser = argparse.ArgumentParser(description="Rerun failed automated batch runs only.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Path to folder with preprocessed CSVs.")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Output folder for reruns.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH, help="Path to batch_summary.json")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Specific run IDs from batch_summary.json to rerun (e.g., --run-ids 34 35 36 or --run-ids 34,35,36).",
    )
    args = parser.parse_args()

    try:
        run_ids = _normalize_run_ids(args.run_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    runner = RerunBatchRunner(args.dataset_dir, args.output_dir, args.summary)
    runner.run_all(run_ids)


if __name__ == "__main__":
    main()
