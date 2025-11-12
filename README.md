# Explainability_of_Process_Prediction

# Directory Structure
BPM_RESEARCH_APP/
│
├── pycache/ # Cached Python bytecode
│
├── BPI_Models/ # Folder containing all event log data and models
│ ├── BPI_logs_csv/ # Original BPI event logs in CSV format
│ ├── BPI_logs_preprocessed_csv/ # Cleaned/preprocessed CSV logs
│ ├── BPI_logs_preprocessed_xes/ # Cleaned/preprocessed XES format
│ ├── BPI_logs_xes/ # Original logs in XES format
│ ├── prediction_models/ # Trained prediction models (Transformers, GNNs)
│ ├── process_models/ # Process models discovered from event logs
│ 
│
├── venv_bpm/ # Python virtual environment (dependencies)
│
├── .gitignore # Git ignore configuration
├── csv_to_xes.py # Script to convert CSV event logs to XES format
├── xes_to_csv.py # Script to convert XES logs to CSV format
├── preprocessor.py # Main preprocessing module for event logs
├── preprocessor_csv.py # Extended CSV-specific preprocessing module
├── process_model.py # Script for process model discovery and visualization
├── main.py # Entry point script for the BPM Research App
├── README.md # Project documentation
├── requirements.txt # Python package dependencies
