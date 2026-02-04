import argparse
import json
import os
import time
import traceback
from datetime import datetime

from ppm_pipeline import (
    default_transformer_config,
    default_gnn_config,
    run_next_activity_prediction,
    run_event_time_prediction,
    run_remaining_time_prediction,
    run_gnn_unified_prediction,
)

DEFAULT_RESULTS_DIR = "automated_batch_results"
DEFAULT_DATASET_DIR = os.path.join("BPI_dataset", "BPI_logs_preprocessed_csv")
DEFAULT_TEST_SIZE = 0.1
# val_split is a fraction of the remaining (train+val) set.
DEFAULT_VAL_SPLIT = 1.0 / 9.0  # 0.111... -> 80/10/10 overall


def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class BatchRunner:
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.master_log = os.path.join(output_dir, f"master_log_{_now_tag()}.txt")
        self.universal_error_log = os.path.join(output_dir, "universal_error_log.txt")
        self.results = []
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_master_header()

    def _write_master_header(self):
        with open(self.master_log, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOMATED BATCH TESTING LOG\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message):
        print(message)
        with open(self.master_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_universal_error(self, message):
        with open(self.universal_error_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def list_datasets(self):
        if not os.path.isdir(self.dataset_dir):
            self.log(f"[X] Dataset directory not found: {self.dataset_dir}")
            return []
        csv_files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(".csv")]
        csv_files.sort()
        self.log(f"Found {len(csv_files)} datasets in {self.dataset_dir}:")
        for i, name in enumerate(csv_files, 1):
            self.log(f"  {i}. {name}")
        return csv_files

    def _write_experiment_setup(self, datasets):
        setup = {
            "dataset_dir": self.dataset_dir,
            "datasets": datasets,
            "split": {"test_size": DEFAULT_TEST_SIZE, "val_split": DEFAULT_VAL_SPLIT},
            "mapping": "auto",
            "transformer_default_config": default_transformer_config(),
            "gnn_default_config": default_gnn_config(),
            "explainability": "all",
            "run_order": ["transformer", "gnn"],
            "timestamp": datetime.now().isoformat(),
        }
        setup_path = os.path.join(self.output_dir, "experiment_setup.json")
        with open(setup_path, "w", encoding="utf-8") as f:
            json.dump(setup, f, indent=2)
        self.log(f"[OK] Experiment setup saved: {setup_path}")

    def build_run_specs(self, dataset_name):
        base_name = os.path.splitext(dataset_name)[0]
        dataset_path = os.path.join(self.dataset_dir, dataset_name)
        specs = []

        transformer_tasks = [
            ("next_activity", run_next_activity_prediction),
            ("event_time", run_event_time_prediction),
            ("remaining_time", run_remaining_time_prediction),
        ]
        gnn_tasks = [
            ("next_activity", run_gnn_unified_prediction),
            ("event_time", run_gnn_unified_prediction),
            ("remaining_time", run_gnn_unified_prediction),
        ]

        for task_name, fn in transformer_tasks:
            specs.append({
                "model": "transformer",
                "task": task_name,
                "dataset_name": base_name,
                "dataset_path": dataset_path,
                "runner": fn,
            })

        for task_name, fn in gnn_tasks:
            specs.append({
                "model": "gnn",
                "task": task_name,
                "dataset_name": base_name,
                "dataset_path": dataset_path,
                "runner": fn,
            })

        return specs

    def run_all(self):
        datasets = self.list_datasets()
        if not datasets:
            self.log("[X] No CSV datasets found. Aborting.")
            return

        self._write_experiment_setup(datasets)

        self.log("\n" + "=" * 80)
        self.log("STARTING AUTOMATED BATCH TESTING")
        self.log("=" * 80)

        total_runs = 0
        for dataset_name in datasets:
            total_runs += len(self.build_run_specs(dataset_name))

        run_index = 0
        for dataset_name in datasets:
            specs = self.build_run_specs(dataset_name)
            for spec in specs:
                run_index += 1
                self.run_single(spec, run_index, total_runs)

        summary_path = os.path.join(self.output_dir, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"\n[OK] Summary saved: {summary_path}")

        metrics_path = os.path.join(self.output_dir, "batch_metrics.csv")
        write_metrics_csv(self.results, metrics_path)
        self.log(f"[OK] Metrics summary saved: {metrics_path}")

    def run_single(self, spec, run_index, total_runs):
        start = time.time()
        dataset_name = spec["dataset_name"]
        model = spec["model"]
        task = spec["task"]
        description = f"{dataset_name} | {model.upper()} | {task}"

        safe_name = description.replace(" ", "_").replace("|", "-")
        run_dir = os.path.join(self.output_dir, f"run_{run_index:03d}_{safe_name}")
        os.makedirs(run_dir, exist_ok=True)

        self.log("\n" + "-" * 80)
        self.log(f"RUN {run_index}/{total_runs}: {description}")
        self.log("-" * 80)

        config = default_transformer_config() if model == "transformer" else default_gnn_config()
        explainability = "all"
        test_size = DEFAULT_TEST_SIZE
        val_split = DEFAULT_VAL_SPLIT

        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": spec["dataset_path"],
                    "model": model,
                    "task": task,
                    "split": {"test_size": test_size, "val_split": val_split},
                    "explainability": explainability,
                    "config": config,
                },
                f,
                indent=2,
            )

        result = {
            "run": run_index,
            "dataset": dataset_name,
            "model": model,
            "task": task,
            "output_dir": run_dir,
            "success": False,
            "duration_s": None,
            "missing_images": [],
            "metrics": {},
        }

        try:
            metrics = None
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
                f.write("RUN ERROR\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Run: {run_index}/{total_runs}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Dataset: {spec['dataset_path']}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Task: {task}\n\n")
                f.write("Exception:\n")
                f.write(str(exc) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())

            self.log(f"[X] Error log saved: {err_path}")

        result["duration_s"] = round(time.time() - start, 2)
        self.results.append(result)


def _dir_has_png(path):
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if name.lower().endswith(".png"):
            return True
    return False


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


def write_metrics_csv(results, output_path):
    base_headers = ["run", "dataset", "model", "task", "success", "duration_s"]
    metric_keys = set()
    for row in results:
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_keys.add(key)
    metric_headers = sorted(metric_keys)
    headers = base_headers + metric_headers

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            metrics = row.get("metrics") or {}
            values = []
            for h in headers:
                if h in base_headers:
                    values.append(str(row.get(h, "")))
                else:
                    val = metrics.get(h, "")
                    values.append(str(val))
            f.write(",".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automated batch testing for PPM pipeline.")
    parser.add_argument("--dataset-dir", help="Path to folder with preprocessed CSVs.")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Output folder for runs.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR

    runner = BatchRunner(dataset_dir, args.output_dir)
    runner.run_all()


if __name__ == "__main__":
    main()
