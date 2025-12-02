
import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import glob
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from transformers.prediction.next_activity import NextActivityPredictor
from transformers.prediction.event_time import EventTimePredictor
from transformers.prediction.remaining_time import RemainingTimePredictor
DATASET_DIRECTORY = "BPI_Models/BPI_logs_preprocessed_csv"  # Directory containing CSV files
BASE_OUTPUT_DIR = "results"  # Base directory for all results

def print_banner():
    print("\n" + "="*70)
    print(" "*15 + "PREDICTIVE PROCESS MONITORING")
    print(" "*20 + "Transformer Network")
    print("="*70 + "\n")


def get_user_choice(prompt, options):
    print(f"\n{prompt}")
    for num, desc in options.items():
        print(f"  {num}. {desc}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice: "))
            if choice in options:
                return choice
            else:
                print(f"Invalid choice. Please select from {list(options.keys())}")
        except ValueError:
            print("Please enter a valid number.")


def get_yes_no(prompt):
    while True:
        response = input(f"\n{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def get_file_initials(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) >= 2:
        initials = ''.join([p[0].upper() for p in parts[:3] if p and len(p) > 0])
        return initials[:5]  # Max 5 characters
    else:
        return name[:5].upper()


def get_dataset_files():
    """Get list of CSV files and let user select one or more"""
    print("\n" + "-"*70)
    print("DATASET SELECTION")
    print("-"*70)
    
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"✗ Directory not found: {DATASET_DIRECTORY}")
        print("Please update DATASET_DIRECTORY in main.py")
        sys.exit(1)
    
    csv_pattern = os.path.join(DATASET_DIRECTORY, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"✗ No CSV files found in: {DATASET_DIRECTORY}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV file(s) in {DATASET_DIRECTORY}:\n")
    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath) / (1024 * 1024) 
        print(f"  {idx}. {filename} ({filesize:.2f} MB)")
    
    print("\n" + "-"*70)
    while True:
        try:
            choice = input("\nEnter the number of the dataset to use (e.g., 1, 2, 3): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                print(f"\n✓ Selected: {os.path.basename(selected_file)}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")


def create_output_directory(dataset_path, task_name):
    initials = get_file_initials(dataset_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_short = task_name.replace(" ", "_").lower()
    folder_name = f"{initials}_{task_short}_{timestamp}"
    output_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n✓ Output directory created: {output_dir}")
    
    info_file = os.path.join(output_dir, "dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Full Path: {os.path.abspath(dataset_path)}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    return output_dir


def get_data_split():
    print("\n" + "-"*70)
    print("DATA SPLIT CONFIGURATION")
    print("-"*70)
    
    split_options = {
        1: "70-15-15 (Train-Val-Test)",
        2: "80-10-10 (Train-Val-Test)",
        3: "60-20-20 (Train-Val-Test)",
        4: "Custom split"
    }
    
    choice = get_user_choice("Select data split:", split_options)
    
    if choice == 1:
        return 0.3, 0.5  # 70-15-15
    elif choice == 2:
        return 0.2, 0.5  # 80-10-10
    elif choice == 3:
        return 0.4, 0.5  # 60-20-20
    else:
        while True:
            try:
                train = float(input("Enter training set percentage (e.g., 70 for 70%): "))
                val = float(input("Enter validation set percentage (e.g., 15 for 15%): "))
                test = float(input("Enter test set percentage (e.g., 15 for 15%): "))
                
                if train + val + test == 100 and all([train > 0, val > 0, test > 0]):
                    test_size = (val + test) / 100
                    val_split = val / (val + test)
                    return test_size, val_split
                else:
                    print("Percentages must sum to 100 and all be positive.")
            except ValueError:
                print("Please enter valid numbers.")


def get_model_config():
    """Get model configuration from user"""
    print("\n" + "-"*70)
    print("MODEL CONFIGURATION")
    print("-"*70)
    
    use_default = get_yes_no("Use default configuration?")
    
    if use_default:
        print("\nUsing default configuration:")
        config = {
            'max_len': 16,
            'd_model': 64,
            'num_heads': 4,
            'num_blocks': 2,
            'dropout_rate': 0.1,
            'epochs': 50,
            'batch_size': 128,
            'patience': 10
        }
    else:
        print("\nEnter custom configuration:")
        config = {}
        
        try:
            config['max_len'] = int(input("  Max sequence length [16]: ") or 16)
            config['d_model'] = int(input("  Model dimension [64]: ") or 64)
            config['num_heads'] = int(input("  Number of attention heads [4]: ") or 4)
            config['num_blocks'] = int(input("  Number of transformer blocks [2]: ") or 2)
            config['dropout_rate'] = float(input("  Dropout rate [0.1]: ") or 0.1)
            config['epochs'] = int(input("  Number of epochs [50]: ") or 50)
            config['batch_size'] = int(input("  Batch size [128]: ") or 128)
            config['patience'] = int(input("  Early stopping patience [10]: ") or 10)
        except ValueError:
            print("Invalid input. Using default configuration.")
            return get_model_config()
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config


def run_next_activity_prediction(dataset_path, output_dir, test_size, val_split, config):
    print("\n" + "="*70)
    print("NEXT ACTIVITY PREDICTION")
    print("="*70)
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {len(df):,} events")
    
    predictor = NextActivityPredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )
    
    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    
    predictor.build_model()
    
    predictor.train(
        data, 
        epochs=config['epochs'], 
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    metrics = predictor.evaluate(data)
    
    y_pred, y_pred_probs = predictor.predict(data)
    
    predictor.save_results(data, y_pred, y_pred_probs, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    
    print("\n" + "="*70)
    print("NEXT ACTIVITY PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test Accuracy':<30} {metrics['test_accuracy']*100:>19.2f}%")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_test']):>20,}")
    print("-"*70)
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)


def run_event_time_prediction(dataset_path, output_dir, test_size, val_split, config):
    """Run event time prediction"""
    print("\n" + "="*70)
    print("EVENT TIME PREDICTION")
    print("="*70)
    
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {len(df):,} events")
    
    predictor = EventTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )

    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    predictor.build_model()
    predictor.train(
        data, 
        epochs=config['epochs'], 
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    metrics = predictor.evaluate(data)
    y_pred = predictor.predict(data)

    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)

    print("\n" + "="*70)
    print("EVENT TIME PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test MAE (Mean Absolute Error)':<30} {metrics['test_mae']:>17.4f} days")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_seq_test']):>20,}")
    print("-"*70)
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)


def run_remaining_time_prediction(dataset_path, output_dir, test_size, val_split, config):
    """Run remaining time prediction"""
    print("\n" + "="*70)
    print("REMAINING TIME PREDICTION")
    print("="*70)
    
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {len(df):,} events")
    
    predictor = RemainingTimePredictor(
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_blocks=config['num_blocks'],
        dropout_rate=config['dropout_rate']
    )
    
    data = predictor.prepare_data(df, test_size=test_size, val_split=val_split)
    
    predictor.build_model()
    
    predictor.train(
        data, 
        epochs=config['epochs'], 
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    metrics = predictor.evaluate(data)
    
    y_pred = predictor.predict(data)
    
    predictor.save_results(data, y_pred, output_dir)
    predictor.plot_predictions(data, y_pred, output_dir)
    predictor.plot_training_history(output_dir)
    predictor.save_model(output_dir)
    
    print("\n" + "="*70)
    print("REMAINING TIME PREDICTION - FINAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-"*70)
    print(f"{'Test MAE (Mean Absolute Error)':<30} {metrics['test_mae']:>17.4f} days")
    print(f"{'Test Loss':<30} {metrics['test_loss']:>20.4f}")
    print(f"{'Number of Test Samples':<30} {len(data['X_seq_test']):>20,}")
    print("-"*70)
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)


def main():
    """Main function"""
    print_banner()
    
    model_type_options = {
        1: "Transformer",
        2: "GNN (Graph Neural Network)"
    }
    
    model_type = get_user_choice("Select model type:", model_type_options)
    
    if model_type == 2:
        print("\n" + "="*70)
        print(" "*25 + "COMING SOON")
        print("="*70)
        print("\nGNN (Graph Neural Network) implementation is under development.")
        print("Please check back later or select Transformer model.")
        sys.exit(0)
    
    task_options = {
        1: "Next Activity Prediction",
        2: "Event Time Prediction",
        3: "Remaining Time Prediction"
    }
    
    task = get_user_choice("Select prediction task:", task_options)
    task_name = task_options[task]
    dataset_path = get_dataset_files()
    output_dir = create_output_directory(dataset_path, task_name)
    test_size, val_split = get_data_split()
    config = get_model_config()
    
    config_file = os.path.join(output_dir, "configuration.txt")
    with open(config_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Test Size: {test_size*100:.1f}%\n")
        f.write(f"Val Split: {val_split*100:.1f}%\n\n")
        f.write("Model Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*50 + "\n")
    
    print(f"\n✓ Configuration saved to: {config_file}")
    try:
        if task == 1:
            run_next_activity_prediction(dataset_path, output_dir, test_size, val_split, config)
        elif task == 2:
            run_event_time_prediction(dataset_path, output_dir, test_size, val_split, config)
        elif task == 3:
            run_remaining_time_prediction(dataset_path, output_dir, test_size, val_split, config)
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        error_file = os.path.join(output_dir, "error_log.txt")
        with open(error_file, 'w') as f:
            f.write("ERROR LOG\n")
            f.write("="*50 + "\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"\n✗ Error log saved to: {error_file}")
        sys.exit(1)
    print("\n" + "="*70)
    print("Thank you for using Predictive Process Monitoring!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()