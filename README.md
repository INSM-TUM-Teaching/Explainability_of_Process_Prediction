# Explainability_of_Process_Prediction


---

##  Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/INSM-TUM-Teaching/Explainability_of_Process_Prediction.git
cd Explainability_of_Process_Prediction

# 2. Create and activate a virtual environment
python -m venv venv_bpm
source venv_bpm/bin/activate    # On Windows: venv_bpm\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt



Directory Structure

Explainability_of_Process_Prediction/
BPM_RESEARCH_APP/
│
├── BPI_Models/
│   ├── BPI_logs_csv/                          # Raw BPI event logs
│   ├── BPI_logs_preprocessed_csv/             # Preprocessed CSV files (input datasets)
│   └── BPI_logs_xes/                          # XES format logs
│
├── conv_and_viz/
│   ├── preprocessor_csv.py                    # CSV preprocessing utilities
│   ├── visualize.py                           # Data visualization tools
│   └── xes_to_csv.py                          # XES to CSV conversion
│   └── process_model.py                       # Process model utilities
│
├── gnns/                                       # GNN Package
│   ├── __init__.py                            # Package initialization
│   ├── model.py                               # HeteroGNN architecture
│   ├── dataset_builder.py                     # Graph dataset builder
│   ├── prefix_generation.py                   # Prefix generation (standalone)
│   └── prediction/
│       ├── __init__.py
│       └── gnn_predictor.py                   # Unified GNN predictor - For all 3 tasks
│
├── transformers/                               # Transformer Package
│   ├── __init__.py                            # Package initialization
│   ├── model.py                               # Transformer architecture
│   └── prediction/
│       ├── __init__.py
│       ├── next_activity.py                   # Next activity predictor
│       ├── event_time.py                      # Event time predictor
│       └── remaining_time.py                  # Remaining time predictor
│
├── explainability/                            # Explainability Package
│   ├── __init__.py                            
│   ├── gnns/
│       └── gnn_explainer.py                   # Unified GNN predictor
│       └── __init__.py                                                   
│   └── transformers/
│       └── transformer_explainer.py            # Unified Transformer Explainer
│       └── __init__.py
│
├── results/                                    # Output directory (auto-created)
├── comprehensive_test_results/                 # Output directory for test results (auto-created)
│
├── main.py                                     # Main entry point to the pipeline
├── automated_test.py                          # Automated testing Script to the main pipeline.(INDEPENDENT Sript)
├── requirements.txt                           # Transformer dependencies
├── README.md                                  # Main documentation



Usage Steps (After executing main.py)
1. Select Model Type
   Option 1: Transformer (choose task: Next Activity / Event Time / Remaining Time)
   Option 2: GNN (predicts all 3 tasks simultaneously)

2. Select Dataset
   Choose from available CSV files in BPI_Models/BPI_logs_preprocessed_csv/

3. Configure Data Split
   Choose preset (70-15-15, 80-10-10, 60-20-20) or custom split

4. Configure Model
   Use default hyperparameters or customize

5. Training Starts Automatically
   Progress displayed in real-time

6. Results saved to results/

