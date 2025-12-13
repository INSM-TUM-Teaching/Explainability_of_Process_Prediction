import os
import sys
import subprocess
import time
from datetime import datetime
import json
import traceback

TEST_RESULTS_DIR = "comprehensive_test_results"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TEST_LOG_FILE = os.path.join(TEST_RESULTS_DIR, f"master_log_{TIMESTAMP}.txt")
DATASET_DIR = "BPI_Models/BPI_logs_preprocessed_csv"


class ComprehensiveTester:
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
        
        with open(TEST_LOG_FILE, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE AUTOMATED TEST LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        print(message)
        with open(TEST_LOG_FILE, 'a') as f:
            f.write(message + "\n")
    
    def get_datasets(self):
        if not os.path.exists(DATASET_DIR):
            self.log(f"✗ Dataset directory not found: {DATASET_DIR}")
            return []
        
        csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        csv_files.sort()
        
        self.log(f"Found {len(csv_files)} datasets:")
        for i, f in enumerate(csv_files, 1):
            self.log(f"  {i}. {f}")
        
        return csv_files
    
    def generate_all_combinations(self, num_datasets):
        combinations = []
        
        models = [
            {"type": "transformer", "code": "1"},
            {"type": "gnn", "code": "2"}
        ]
        
        transformer_tasks = [
            {"name": "next_activity", "code": "1"},
            {"name": "event_time", "code": "2"},
            {"name": "remaining_time", "code": "3"}
        ]
        
        gnn_tasks = [
            {"name": "next_activity", "code": "1"},
            {"name": "event_time", "code": "2"},
            {"name": "remaining_time", "code": "3"}
        ]
        
        data_splits = [
            {"name": "70-15-15", "code": "1"}
        ]
        
        transformer_explainability = [
            {"name": "SHAP", "code": "1"},
            {"name": "LIME", "code": "2"},
            {"name": "All", "code": "3"}
        ]
        
        gnn_explainability = [
            {"name": "Gradient", "code": "1"},
            {"name": "GraphLIME", "code": "2"},
            {"name": "Skip", "code": "4"}
        ]
        
        for dataset_idx in range(1, num_datasets + 1):
            
            for task in transformer_tasks:
                for split in data_splits:
                    for explainability in transformer_explainability:
                        combinations.append({
                            "model": "transformer",
                            "task": task["name"],
                            "dataset_index": dataset_idx,
                            "data_split": split["name"],
                            "explainability": explainability["name"],
                            "inputs": [
                                "1",
                                task["code"],
                                str(dataset_idx),
                                split["code"],
                                explainability["code"],
                                "y"
                            ]
                        })
            
            for task in gnn_tasks:
                for split in data_splits:
                    for explainability in gnn_explainability:
                        combinations.append({
                            "model": "gnn",
                            "task": task["name"],
                            "dataset_index": dataset_idx,
                            "data_split": split["name"],
                            "explainability": explainability["name"],
                            "inputs": [
                                "2",
                                task["code"],
                                str(dataset_idx),
                                split["code"],
                                explainability["code"],
                                "y"
                            ]
                        })
        
        return combinations
    
    def generate_description(self, config):
        dataset = f"Dataset_{config['dataset_index']}"
        model = config['model'].upper()
        task = config['task'].replace('_', ' ').title()
        split = config['data_split']
        explain = config['explainability']
        
        return f"{dataset} | {model} | {task} | {split} | Explain:{explain}"
    
    def run_single_test(self, config, test_num, total_tests):
        test_start = time.time()
        description = self.generate_description(config)
        
        self.log("\n" + "="*80)
        self.log(f"TEST {test_num}/{total_tests}: {description}")
        self.log("="*80)
        
        safe_name = description.replace(" ", "_").replace("|", "-").replace(":", "_")
        test_folder = os.path.join(TEST_RESULTS_DIR, f"test_{test_num:04d}_{safe_name}")
        os.makedirs(test_folder, exist_ok=True)
        
        config_file = os.path.join(test_folder, "config.json")
        with open(config_file, 'w') as f:
            json.dump({
                "test_number": test_num,
                "description": description,
                "configuration": config,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        try:
            input_sequence = "\n".join(config["inputs"]) + "\n"
            
            self.log(f"Input sequence: {config['inputs']}")
            
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            try:
                stdout, stderr = process.communicate(input=input_sequence, timeout=600)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError("Test exceeded 10 minute timeout")
            
            test_duration = time.time() - test_start
            
            success = process.returncode == 0 and ("✓" in stdout or "All results saved" in stdout)
            
            with open(os.path.join(test_folder, "stdout.txt"), 'w', encoding='utf-8') as f:
                f.write(stdout)
            
            with open(os.path.join(test_folder, "stderr.txt"), 'w', encoding='utf-8') as f:
                f.write(stderr)
            
            result = {
                "test_number": test_num,
                "description": description,
                "success": success,
                "duration": test_duration,
                "return_code": process.returncode,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.log(f"✓ TEST PASSED ({test_duration:.1f}s)")
                self.passed_tests += 1
                
                with open(os.path.join(test_folder, "SUCCESS.txt"), 'w') as f:
                    f.write(f"Test passed successfully\n")
                    f.write(f"Duration: {test_duration:.1f}s\n")
                    f.write(f"Return code: {process.returncode}\n")
            
            else:
                self.log(f"✗ TEST FAILED ({test_duration:.1f}s)")
                self.failed_tests += 1
                
                error_log_path = os.path.join(test_folder, "ERROR_LOG.txt")
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("ERROR LOG\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Test Number: {test_num}\n")
                    f.write(f"Description: {description}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Duration: {test_duration:.1f}s\n")
                    f.write(f"Return Code: {process.returncode}\n\n")
                    
                    f.write("="*80 + "\n")
                    f.write("USER INPUTS\n")
                    f.write("="*80 + "\n")
                    f.write(f"Model: {config['model']}\n")
                    f.write(f"Task: {config['task']}\n")
                    f.write(f"Dataset Index: {config['dataset_index']}\n")
                    f.write(f"Data Split: {config['data_split']}\n")
                    f.write(f"Explainability: {config['explainability']}\n")
                    f.write(f"Use Default Config: Yes\n\n")
                    
                    f.write("Input Sequence:\n")
                    for i, inp in enumerate(config['inputs'], 1):
                        f.write(f"  {i}. {inp}\n")
                    f.write("\n")
                    
                    f.write("="*80 + "\n")
                    f.write("ERROR DETAILS\n")
                    f.write("="*80 + "\n\n")
                    
                    if stderr:
                        f.write("STDERR OUTPUT:\n")
                        f.write("-"*80 + "\n")
                        f.write(stderr)
                        f.write("\n\n")
                    
                    f.write("CRASH LOCATION (Last 1000 chars of stdout):\n")
                    f.write("-"*80 + "\n")
                    f.write(stdout[-1000:] if len(stdout) > 1000 else stdout)
                    f.write("\n\n")
                    
                    if "Error" in stdout or "Exception" in stdout or "Traceback" in stdout:
                        f.write("DETECTED ERROR IN STDOUT:\n")
                        f.write("-"*80 + "\n")
                        lines = stdout.split('\n')
                        error_lines = [l for l in lines if 'error' in l.lower() or 'exception' in l.lower()]
                        for line in error_lines[-10:]:
                            f.write(line + "\n")
                
                self.log(f"  Error log saved: {error_log_path}")
                
                result["error"] = {
                    "stderr_snippet": stderr[-500:] if stderr else "No stderr",
                    "stdout_snippet": stdout[-500:] if len(stdout) > 500 else stdout
                }
            
            self.results.append(result)
            return success
            
        except TimeoutError as e:
            self.log(f"✗ TEST TIMEOUT (10 minutes)")
            self.failed_tests += 1
            
            error_log_path = os.path.join(test_folder, "ERROR_LOG.txt")
            with open(error_log_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TIMEOUT ERROR\n")
                f.write("="*80 + "\n\n")
                f.write(f"Test exceeded 10 minute timeout\n")
                f.write(f"Test Number: {test_num}\n")
                f.write(f"Description: {description}\n\n")
                f.write("User Inputs:\n")
                f.write(json.dumps(config, indent=2))
            
            self.results.append({
                "test_number": test_num,
                "description": description,
                "success": False,
                "duration": 600,
                "error": "Timeout after 10 minutes"
            })
            return False
            
        except Exception as e:
            self.log(f"✗ TEST EXCEPTION: {str(e)}")
            self.failed_tests += 1
            
            error_log_path = os.path.join(test_folder, "ERROR_LOG.txt")
            with open(error_log_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("EXCEPTION ERROR\n")
                f.write("="*80 + "\n\n")
                f.write(f"Exception: {str(e)}\n\n")
                f.write(f"Test Number: {test_num}\n")
                f.write(f"Description: {description}\n\n")
                f.write("User Inputs:\n")
                f.write(json.dumps(config, indent=2))
                f.write("\n\nTraceback:\n")
                f.write(traceback.format_exc())
            
            self.results.append({
                "test_number": test_num,
                "description": description,
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            })
            return False
    
    def run_all_tests(self):
        self.start_time = time.time()
        
        datasets = self.get_datasets()
        if not datasets:
            self.log("✗ No datasets found. Exiting.")
            return
        
        num_datasets = len(datasets)
        
        self.log(f"\nGenerating test combinations for {num_datasets} datasets...")
        combinations = self.generate_all_combinations(num_datasets)
        total_tests = len(combinations)
        
        self.log(f"✓ Generated {total_tests} test combinations")
        self.log(f"  - Transformer tests: {len([c for c in combinations if c['model'] == 'transformer'])}")
        self.log(f"  - GNN tests: {len([c for c in combinations if c['model'] == 'gnn'])}")
        self.log(f"  - Tests per dataset: {total_tests // num_datasets}")
        
        estimated_minutes = total_tests * 2
        self.log(f"\nEstimated total time: ~{estimated_minutes} minutes ({estimated_minutes/60:.1f} hours)")
        
        self.log("\n" + "="*80)
        self.log("STARTING COMPREHENSIVE TESTING")
        self.log("="*80)
        
        for i, config in enumerate(combinations, 1):
            self.run_single_test(config, i, total_tests)
            
            if i % 10 == 0:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / i
                remaining = (total_tests - i) * avg_time
                self.log(f"\n--- Progress: {i}/{total_tests} | Passed: {self.passed_tests} | Failed: {self.failed_tests} | ETA: {remaining/60:.1f} min ---\n")
            
            time.sleep(1)
        
        self.end_time = time.time()
        self.generate_final_report()
    
    def generate_final_report(self):
        total_duration = self.end_time - self.start_time
        total_tests = self.passed_tests + self.failed_tests
        
        self.log("\n" + "="*80)
        self.log("COMPREHENSIVE TEST SUMMARY")
        self.log("="*80)
        self.log(f"\nTotal Tests Run: {total_tests}")
        self.log(f"Passed: {self.passed_tests} ✓")
        self.log(f"Failed: {self.failed_tests} ✗")
        self.log(f"Success Rate: {(self.passed_tests/total_tests*100):.1f}%")
        self.log(f"\nTotal Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes / {total_duration/3600:.2f} hours)")
        self.log(f"Average per test: {total_duration/total_tests:.1f}s")
        
        summary_file = os.path.join(TEST_RESULTS_DIR, "comprehensive_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "total_tests": total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "success_rate": (self.passed_tests/total_tests*100),
                "total_duration_seconds": total_duration,
                "total_duration_minutes": total_duration/60,
                "total_duration_hours": total_duration/3600,
                "average_test_duration": total_duration/total_tests,
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        
        failures = [r for r in self.results if not r['success']]
        if failures:
            failures_file = os.path.join(TEST_RESULTS_DIR, "failures_summary.txt")
            with open(failures_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"FAILED TESTS SUMMARY ({len(failures)} failures)\n")
                f.write("="*80 + "\n\n")
                
                for fail in failures:
                    f.write(f"Test {fail['test_number']}: {fail['description']}\n")
                    if 'error' in fail:
                        f.write(f"  Error: {fail.get('error', 'Unknown error')}\n")
                    f.write("\n")
            
            self.log(f"\n✓ Failures summary saved: {failures_file}")
        
        self.log(f"\n✓ Comprehensive summary saved: {summary_file}")
        self.log(f"✓ Master log saved: {TEST_LOG_FILE}")
        self.log(f"✓ All test results saved in: {TEST_RESULTS_DIR}/")
        self.log("\n" + "="*80)


def main():
    print("\n" + "="*80)
    print(" "*20 + "AUTOMATED TESTING")
    print("="*80)
    print("\nThis will test selected combinations for EACH dataset:")
    print("  • Transformer:")
    print("    - Tasks: Next Activity, Event Time, Remaining Time")
    print("    - Split: 70-15-15 only")
    print("    - Explainability: SHAP, LIME, All Methods")
    print("  • GNN:")
    print("    - Tasks: Next Activity, Event Time, Remaining Time")
    print("    - Split: 70-15-15 only")
    print("    - Explainability: Gradient, GraphLIME, Skip")
    print("  • Default configuration only (always 'y')")
    
    if os.path.exists(DATASET_DIR):
        num_datasets = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
        print(f"\nDetected: {num_datasets} datasets")
        
        total_tests = num_datasets * 18
        estimated_hours = (total_tests * 2) / 60
        
        print(f"Total tests: {total_tests}")
        print(f"  - Transformer: {num_datasets * 9} tests (3 tasks × 1 split × 3 explainability)")
        print(f"  - GNN: {num_datasets * 9} tests (3 tasks × 1 split × 3 explainability)")
        print(f"Estimated time: ~{estimated_hours:.1f} hours")
    else:
        print(f"\n✗ Dataset directory not found: {DATASET_DIR}")
        return
    
    print(f"\nResults will be saved to: {TEST_RESULTS_DIR}/")
    print("  • Each test gets its own folder")
    print("  • Errors saved with full details")
    print("  • Testing continues on failure")
    
    response = input("\nProceed with comprehensive testing? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("Testing cancelled.")
        return
    
    print("\n" + "="*80)
    print("Starting comprehensive testing...")
    print("="*80 + "\n")
    
    tester = ComprehensiveTester()
    tester.run_all_tests()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING COMPLETED!")
    print(f"Check {TEST_RESULTS_DIR}/ for all results")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()