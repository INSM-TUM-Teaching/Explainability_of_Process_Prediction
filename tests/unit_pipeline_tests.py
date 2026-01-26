import json
import os
import traceback
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from backend import main as backend_main


REPORT_DIR = Path("tests/test_reports")


class UnitTestLogger:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.report_dir / f"unit_test_log_{timestamp}.txt"
        self.results = []

        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("UNIT TEST RUN\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

    def log(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def add_result(self, name: str, success: bool, error: str | None = None) -> None:
        self.results.append({
            "name": name,
            "success": success,
            "error": error,
        })

    def write_report(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"unit_test_report_{timestamp}.json"
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
            "results": self.results,
            "generated_at": datetime.now().isoformat(),
        }
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


LOGGER = UnitTestLogger(REPORT_DIR)


class PipelineUnitTests(unittest.TestCase):
    def _record(self, name: str, ok: bool, error: str | None = None) -> None:
        LOGGER.add_result(name, ok, error)

    def test_infer_column_types(self):
        name = "infer_column_types"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "2", "3"],
                "Activity": ["A", "B", "C"],
                "Duration": [1.2, 3.4, 5.6],
            })
            result = backend_main._infer_column_types(df)
            self.assertEqual(result["CaseID"], "categorical")
            self.assertEqual(result["Activity"], "categorical")
            self.assertEqual(result["Duration"], "numerical")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_detect_and_standardize_columns(self):
        name = "detect_and_standardize_columns"
        try:
            df = pd.DataFrame({
                "case:concept:name": ["1", "1", "2"],
                "concept:name": ["A", "B", "C"],
                "time:timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            })
            df_out, mapping = backend_main.detect_and_standardize_columns(df)
            self.assertIn("CaseID", df_out.columns)
            self.assertIn("Activity", df_out.columns)
            self.assertIn("Timestamp", df_out.columns)
            self.assertEqual(mapping["case:concept:name"], "CaseID")
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_compute_split_frames(self):
        name = "compute_split_frames"
        try:
            df = pd.DataFrame({
                "CaseID": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"],
                "Activity": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                "Timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            })
            cfg = backend_main.SplitConfig(test_size=0.2, val_split=0.5)
            train_df, val_df, test_df = backend_main._compute_split_frames(df, cfg)
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
            self.assertGreater(len(test_df), 0)
            all_ids = set(train_df.index) | set(val_df.index) | set(test_df.index)
            self.assertEqual(len(all_ids), len(df))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise

    def test_validate_split_config(self):
        name = "validate_split_config"
        try:
            with self.assertRaises(Exception):
                backend_main._validate_split_config(backend_main.SplitConfig(test_size=0, val_split=0.5))
            with self.assertRaises(Exception):
                backend_main._validate_split_config(backend_main.SplitConfig(test_size=0.2, val_split=1.5))
            backend_main._validate_split_config(backend_main.SplitConfig(test_size=0.2, val_split=0.5))
            self._record(name, True)
        except Exception as exc:
            self._record(name, False, str(exc))
            raise


def run_unit_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineUnitTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    LOGGER.write_report()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        exit_code = run_unit_tests()
        LOGGER.log("Unit tests completed.")
        raise SystemExit(exit_code)
    except Exception:
        LOGGER.log("Unit test runner crashed:")
        LOGGER.log(traceback.format_exc())
        LOGGER.write_report()
        raise
