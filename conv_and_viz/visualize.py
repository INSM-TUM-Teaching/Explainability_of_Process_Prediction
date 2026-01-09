import os
from datetime import datetime


def visualize_process(xes_path, output_folder=None):
    """
    Generate process model visualizations from XES file.
    
    Args:
        xes_path: Path to XES event log file
        output_folder: Output folder for visualizations. If None, creates one based on XES filename.
        
    Returns:
        dict: Paths to generated visualization files
    """
    from conv_and_viz.process_model import build_and_save_all
    
    if output_folder is None:
        base_name = os.path.splitext(os.path.basename(xes_path))[0]
        output_folder = os.path.join("results", f"process_models_{base_name}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nGenerating process model visualizations...")
    print(f"Input: {xes_path}")
    print(f"Output folder: {output_folder}")
    
    results = build_and_save_all(xes_path, output_folder)
    
    print(f"\n[OK] Visualizations generated:")
    print(f"  - Frequency DFG: {results['frequency_jpg']}")
    print(f"  - Performance DFG: {results['performance_jpg']}")
    print(f"  - Petri Net: {results['petri_jpg']}")
    
    return results


def main():
    xes_path = "BPI_Models/BPI_logs_xes/BPI_2020_Log_RequestForPayment.xes"
    output_folder = "BPI_Models/process_models/BPI_2020_Log_RequestForPayment"

    if not os.path.exists(xes_path):
        print(f"[X] XES file not found: {xes_path}")
        print("Please update the path to your XES file.")
        return

    try:
        from conv_and_viz.process_model import build_and_save_all
    except ImportError:
        from process_model import build_and_save_all

    results = build_and_save_all(xes_path, output_folder)

    print("\nVisualization complete!")
    print("Frequency DFG saved at:", results["frequency_jpg"])
    print("Performance DFG saved at:", results["performance_jpg"])
    print("Petrinet saved at:", results["petri_jpg"])


if __name__ == "__main__":
    main()